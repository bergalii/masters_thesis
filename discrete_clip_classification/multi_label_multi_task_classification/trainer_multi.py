import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn
import logging
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np


class MultiTaskHead(nn.Module):
    """Classification head for each task (verb, instrument, target)"""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class MultiTaskSelfDistillationTrainer:
    def __init__(
        self,
        num_epochs: int,
        train_loader,
        val_loader,
        num_classes: dict,
        label_mappings: dict,
        device,
        logger: logging.Logger,
        dir_name: str,
        warmup_epochs: int = 5,
        temperature: float = 2.0,
        task_weights: dict = None,  # Optional weights for each task's loss
    ):
        self.num_epochs = num_epochs
        self.alpha = min(1.0, num_epochs / warmup_epochs)
        self.temperature = temperature
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.label_mappings = label_mappings
        self.device = device
        self.logger = logger
        self.dir_name = dir_name
        self.task_weights = task_weights

        self._configure_models()

    def _configure_models(self):
        # Initialize base models
        self.teacher_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        self.student_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)

        # Replace classification heads for both models
        for model in [self.teacher_model, self.student_model]:
            # Store the number of features for the heads
            in_features = model.num_features

            # Create separate heads for each task
            model.verb_head = (MultiTaskHead(in_features, self.num_classes["verb"])).to(
                self.device
            )

            model.instrument_head = MultiTaskHead(
                in_features, self.num_classes["instrument"]
            ).to(self.device)

            model.target_head = MultiTaskHead(
                in_features, self.num_classes["target"]
            ).to(self.device)

            # Remove original classification head
            model.head = nn.Identity()

        # Create optimizers and schedulers
        self.teacher_optimizer, self.teacher_scheduler = (
            self._create_optimizer_and_scheduler(
                self.teacher_model, lr=0.001, weight_decay=0.0001, patience=7
            )
        )

        self.student_optimizer, self.student_scheduler = (
            self._create_optimizer_and_scheduler(
                self.student_model, lr=0.001, weight_decay=0.0001, patience=7
            )
        )

    def _forward_pass(self, model, inputs):
        """Perform forward pass through the model and all task heads"""
        features = model(inputs)
        return {
            "verb": model.verb_head(features),
            "instrument": model.instrument_head(features),
            "target": model.target_head(features),
        }

    def _compute_loss(self, outputs, labels, soft_labels=None):
        """Compute the combined loss for all tasks"""
        total_loss = 0
        losses = {}

        for task in ["verb", "instrument", "target"]:
            # Standard cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(outputs[task], labels[task])

            # Add distillation loss if soft labels are provided
            if soft_labels is not None:
                distillation_loss = F.binary_cross_entropy_with_logits(
                    outputs[task], soft_labels[task]
                )
                loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

            # Weight the task loss
            weighted_loss = self.task_weights[task] * loss
            total_loss += weighted_loss
            losses[task] = loss.item()

        return total_loss, losses

    def _train_model(
        self, model, optimizer, lr_scheduler, is_teacher=True, soft_labels=None
    ):
        """Train either teacher or student model"""
        best_map = {task: 0.0 for task in ["verb", "instrument", "target"]}

        for epoch in range(self.num_epochs):
            model.train()
            epoch_losses = {task: 0.0 for task in ["verb", "instrument", "target"]}

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                batch_labels = {
                    task: labels[task].to(self.device)
                    for task in ["verb", "instrument", "target"]
                }

                outputs = self._forward_pass(model, inputs)

                # Get batch soft labels if training student
                batch_soft_labels = None
                if not is_teacher and soft_labels is not None:
                    batch_soft_labels = {
                        task: soft_labels[task][
                            batch_idx
                            * self.train_loader.batch_size : (batch_idx + 1)
                            * self.train_loader.batch_size
                        ].to(self.device)
                        for task in ["verb", "instrument", "target"]
                    }

                total_loss, task_losses = self._compute_loss(
                    outputs, batch_labels, batch_soft_labels
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # Update epoch losses
                for task, loss in task_losses.items():
                    epoch_losses[task] += loss

            # Validation
            self.logger.info(f"Validation Results - Epoch {epoch+1}/{self.num_epochs}:")
            val_maps = self._validate_model(model)

            # Update learning rate schedulers based on average mAP
            avg_map = sum(val_maps.values()) / len(val_maps)
            lr_scheduler.step(avg_map)

            # Log metrics
            self.logger.info("Training Losses:")
            for task in ["verb", "instrument", "target"]:
                avg_loss = epoch_losses[task] / len(self.train_loader)
                self.logger.info(
                    f"{task}: Loss={avg_loss:.4f}, mAP={val_maps[task]:.4f}"
                )
            self.logger.info("-" * 50)

            # Save best models for each task
            for task in ["verb", "instrument", "target"]:
                if val_maps[task] > best_map[task]:
                    self.logger.info(f"Saving the best model for the task {task}")
                    best_map[task] = val_maps[task]
                    torch.save(
                        model.state_dict(), f"{self.dir_name}/best_model_{task}.pth"
                    )

    def _generate_soft_labels(self):
        """Generate soft labels using trained teacher model"""
        self.teacher_model.eval()
        soft_labels = {"verb": [], "instrument": [], "target": []}

        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(self.device)
                outputs = self._forward_pass(self.teacher_model, inputs)

                for task in ["verb", "instrument", "target"]:
                    soft_labels[task].append(torch.sigmoid(outputs[task]).cpu())

        return {task: torch.cat(labels, dim=0) for task, labels in soft_labels.items()}

    def _validate_model(self, model):
        """Validate model and compute metrics for all tasks"""
        model.eval()
        outputs = {task: [] for task in ["verb", "instrument", "target"]}
        labels = {task: [] for task in ["verb", "instrument", "target"]}

        with torch.no_grad():
            for inputs, batch_labels in self.val_loader:
                inputs = inputs.to(self.device)
                model_outputs = self._forward_pass(model, inputs)

                for task in ["verb", "instrument", "target"]:
                    outputs[task].append(model_outputs[task].cpu())
                    labels[task].append(batch_labels[task].cpu())

        task_metrics = {}
        for task in ["verb", "instrument", "target"]:
            task_outputs = torch.cat(outputs[task], dim=0)
            task_labels = torch.cat(labels[task], dim=0)
            predictions = torch.sigmoid(task_outputs).numpy()
            true_labels = task_labels.numpy()

            task_metrics[task] = {
                "overall": {
                    "mAP": average_precision_score(
                        true_labels, predictions, average="micro"
                    ),
                    "macro_AP": average_precision_score(
                        true_labels, predictions, average="macro"
                    ),
                    "weighted_AP": average_precision_score(
                        true_labels, predictions, average="weighted"
                    ),
                },
                "per_class": {},
            }

            class_aps = average_precision_score(true_labels, predictions, average=None)
            for i in range(len(class_aps)):
                class_preds = predictions[:, i]
                class_labels = true_labels[:, i]
                label_name = self.label_mappings[task].get(i, f"Class_{i}")

                thresholds = np.arange(0.2, 0.7, 0.1)
                f1_scores = [
                    f1_score(class_labels, class_preds > t) for t in thresholds
                ]
                optimal_threshold = thresholds[np.argmax(f1_scores)]
                binary_preds = class_preds > 0.5

                task_metrics[task]["per_class"][label_name] = {
                    "AP": class_aps[i],
                    "precision": precision_score(class_labels, binary_preds),
                    "recall": recall_score(class_labels, binary_preds),
                    "f1": f1_score(class_labels, binary_preds),
                    "optimal_threshold": optimal_threshold,
                }
            # Log metrics for each task
            self.logger.info(f"{task.upper()} METRICS:")
            self.logger.info(
                f"  Overall mAP: {task_metrics[task]['overall']['mAP']:.4f}"
            )
            self.logger.info(
                f"  Macro AP: {task_metrics[task]['overall']['macro_AP']:.4f}"
            )
            self.logger.info(
                f"  Weighted AP: {task_metrics[task]['overall']['weighted_AP']:.4f}"
            )
            for class_name, metrics in task_metrics[task]["per_class"].items():
                self.logger.info(f" {class_name}:")
                self.logger.info(f"  AP: {metrics['AP']:.4f}")
                self.logger.info(f"  F1: {metrics['f1']:.4f}")
                self.logger.info(f"  Precision: {metrics['precision']:.4f}")
                self.logger.info(f"  Recall: {metrics['recall']:.4f}")
                self.logger.info(
                    f"  Optimal threshold: {metrics['optimal_threshold']:.2f}"
                )

        return {
            task: metrics["overall"]["mAP"] for task, metrics in task_metrics.items()
        }

    def train(self):
        """Execute full training pipeline with self-distillation"""
        # Train teacher model
        self.logger.info("Training teacher model...")
        trainable_params_teacher = sum(
            p.numel() for p in self.teacher_model.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable parameters: {trainable_params_teacher:,}")
        self.logger.info("-" * 50)
        self._train_model(
            self.teacher_model,
            self.teacher_optimizer,
            self.teacher_scheduler,
            is_teacher=True,
        )

        # Generate soft labels
        self.logger.info("Generating soft labels...")
        soft_labels = self._generate_soft_labels()

        # Train student model
        self.logger.info("Training student model...")
        self._train_model(
            self.student_model,
            self.student_optimizer,
            self.student_scheduler,
            is_teacher=False,
            soft_labels=soft_labels,
        )

    def _create_optimizer_and_scheduler(self, model, lr, weight_decay, patience):
        """Create optimizer and scheduler with parameter groups"""
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "bias" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = SGD(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            momentum=0.9,
            nesterov=True,
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=patience
        )

        return optimizer, scheduler

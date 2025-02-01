import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
import torch.nn as nn
import logging
from torchvision.models.video.swin_transformer import (
    swin3d_s,
    Swin3D_S_Weights,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np


class SelfDistillationTrainer:
    def __init__(
        self,
        num_epochs: int,
        train_loader,
        val_loader,
        num_classes,
        device,
        logger: logging.Logger,
        label_mappings: dict,
        warmup_epochs: int = 5,
        temperature: float = 2.0,
    ):
        self.num_epochs = num_epochs
        self.alpha = min(
            1.0, num_epochs / warmup_epochs
        )  # Gradually increase from 0 to 1 the impact of distill loss
        self.temperature = temperature
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.logger = logger
        self.label_mappings = label_mappings["verb"]
        # self.temperature = max(1.0, initial_temperature * (1 - epoch/num_epochs)) gradually increase temp
        self._configure_models()

    def _configure_models(self):
        # Initialize models, optimizer, scheduler
        self.teacher_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(
            self.device
        )  # Your model architecture
        self.student_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(
            self.device
        )  # Same architecture for self-distillation
        # 2. Modify their heads
        for model in [self.teacher_model, self.student_model]:
            model.head = nn.Sequential(
                nn.LayerNorm(model.num_features),
                nn.Dropout(p=0.5),
                nn.Linear(model.num_features, 512),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, self.num_classes),
            ).to(self.device)

        # Create teacher and student optimizers and schedulers
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

    def _create_optimizer_and_scheduler(self, model, lr, weight_decay, patience):
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                (no_decay_params if "bias" in name else decay_params).append(param)

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

    def train(self):
        """
        Full training pipeline with self-distillation
        """
        # First train teacher model
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

        # Generate soft labels using teacher
        self.logger.info("Generating soft labels...")
        soft_labels = self._generate_soft_labels()

        # Train student model using soft labels
        self.logger.info("Training student model...")
        trainable_params_student = sum(
            p.numel() for p in self.student_model.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable parameters: {trainable_params_student:,}")
        self.logger.info("-" * 50)
        self._train_model(
            self.student_model,
            self.student_optimizer,
            self.student_scheduler,
            is_teacher=False,
            soft_labels=soft_labels,
        )

    def _train_model(
        self,
        model,
        optimizer,
        lr_scheduler,
        is_teacher: bool,
        soft_labels: torch.Tensor = None,
    ):
        """
        Train the teacher model or the student model with distillation loss
        """
        best_map = 0.0
        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels["verb"].to(self.device)
                outputs = model(inputs)
                if is_teacher:
                    # Default Loss
                    total_loss = F.binary_cross_entropy_with_logits(outputs, labels)

                else:
                    # Distillation loss
                    soft_targets = soft_labels[
                        batch_idx
                        * self.train_loader.batch_size : (batch_idx + 1)
                        * self.train_loader.batch_size
                    ]
                    soft_targets = soft_targets.to(self.device)
                    total_loss = F.binary_cross_entropy_with_logits(outputs, labels)

                optimizer.zero_grad()
                total_loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += total_loss.item()
                self._save_checkpoint(model)
            # Validation
            self.logger.info(
                f"Validation Results - Epoch {epoch+1}/{self.num_epochs} :"
            )
            val_map = self._validate_model(model)
            self.logger.info("-" * 50)
            lr_scheduler.step(val_map)
            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss/len(self.train_loader):.4f} - mAP: {val_map:.4f}"
            )
            self.logger.info("-" * 50)

            if val_map > best_map:
                best_map = val_map
                torch.save(model, "best_model.pth")

    def _generate_soft_labels(self) -> torch.Tensor:
        """Generate soft labels using trained teacher model"""
        self.teacher_model.eval()
        all_soft_labels = []

        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(self.device)
                outputs = self.teacher_model(inputs)
                soft_labels = torch.sigmoid(outputs)
                all_soft_labels.append(soft_labels.cpu())

        return torch.cat(all_soft_labels, dim=0)

    def _validate_model(self, model) -> dict:
        """
        Validate model with proper handling of edge cases
        Returns:
            dict: Dictionary containing overall and per-class metrics
        """
        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels["verb"].cpu())

        outputs = torch.cat(all_outputs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        predictions = torch.sigmoid(outputs).numpy()
        true_labels = labels.numpy()

        # Overall metrics with continuous predictions
        metrics = {
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
            }
        }

        # Per-class metrics
        class_aps = average_precision_score(true_labels, predictions, average=None)
        metrics["per_class"] = {}

        for i in range(len(class_aps)):
            class_preds = predictions[:, i]
            class_labels = true_labels[:, i]
            label_name = self.label_mappings.get(i)
            # Calculate optimal threshold using F1 score between 0.2 - 0.7
            thresholds = np.arange(0.2, 0.7, 0.1)
            f1_scores = []
            for threshold in thresholds:
                binary_preds = class_preds > threshold
                f1 = f1_score(class_labels, binary_preds)
                f1_scores.append(f1)

            optimal_threshold = thresholds[np.argmax(f1_scores)]
            # Try having a fix threshold of 0.5
            binary_preds = class_preds > 0.5
            # binary_preds = class_preds > optimal_threshold
            metrics["per_class"][label_name] = {
                "AP": class_aps[i],
                "precision": precision_score(class_labels, binary_preds),
                "recall": recall_score(class_labels, binary_preds),
                "f1": f1_score(class_labels, binary_preds),
                "optimal_threshold": optimal_threshold,
                "status": "active",
            }

        # Print detailed report
        self.logger.info(f"Overall mAP: {metrics['overall']['mAP']:.4f}")
        self.logger.info(f"Macro AP: {metrics['overall']['macro_AP']:.4f}")
        self.logger.info(f"Weighted AP: {metrics['overall']['weighted_AP']:.4f}")
        self.logger.info("Per-class Performance:")
        for class_name, class_metrics in metrics["per_class"].items():
            self.logger.info(f"{class_name}:")
            self.logger.info(f"  AP: {class_metrics['AP']:.4f}")
            self.logger.info(f"  F1: {class_metrics['f1']:.4f}")
            self.logger.info(f"  Precision: {class_metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {class_metrics['recall']:.4f}")
            self.logger.info(
                f"  Optimal threshold: {class_metrics['optimal_threshold']:.2f}"
            )

        return metrics["overall"]["mAP"]

    def _save_checkpoint(self, model):
        """Save model checkpoint"""
        torch.save(model.state_dict(), "checkpoint.pth")

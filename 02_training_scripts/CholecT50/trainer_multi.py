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


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dims, common_dim=128):
        super().__init__()
        self.common_dim = common_dim
        self.query = nn.Linear(feature_dims["verb"], common_dim)
        self.key = nn.Linear(feature_dims["instrument"], common_dim)
        self.value = nn.Linear(feature_dims["target"], common_dim)

    def forward(self, verb_feat, inst_feat, target_feat):
        # Project features to common dimension
        Q = self.query(verb_feat)
        K = self.key(inst_feat)
        V = self.value(target_feat)

        # Compute scaled dot-product attention
        scale_factor = 1.0 / (self.common_dim**0.5)
        attention_scores = (
            torch.bmm(Q.unsqueeze(1), K.unsqueeze(1).transpose(1, 2)) * scale_factor
        )
        attention = torch.softmax(attention_scores, dim=-1)

        return torch.bmm(attention, V.unsqueeze(1)).squeeze(1)


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
        triplet_to_ivt: dict,
        label_mappings: dict,
        device,
        logger: logging.Logger,
        dir_name: str,
        learning_rate: float,
        weight_decay: float,
        hidden_layer_dim: int,
        warmup_epochs: int = 5,
        temperature: float = 2.0,
    ):
        self.num_epochs = num_epochs
        self.alpha = min(1.0, num_epochs / warmup_epochs)
        self.temperature = temperature
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.triplet_to_ivt = triplet_to_ivt
        self.label_mappings = label_mappings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_layer_dim = hidden_layer_dim

        self.device = device
        self.logger = logger
        self.dir_name = dir_name

        self.feature_dims = {k: v for k, v in num_classes.items() if k != "triplet"}
        self.cross_attention = CrossAttentionFusion(self.feature_dims).to(self.device)
        self._configure_models()

    def _configure_models(self):
        # Initialize the teacher model
        self.teacher_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        in_features = self.teacher_model.num_features
        # Teacher heads
        self.teacher_model.verb_head = MultiTaskHead(
            in_features, self.num_classes["verb"]
        ).to(self.device)
        self.teacher_model.instrument_head = MultiTaskHead(
            in_features, self.num_classes["instrument"]
        ).to(self.device)
        self.teacher_model.target_head = MultiTaskHead(
            in_features, self.num_classes["target"]
        ).to(self.device)

        # Teacher triplet head combines the ivt heads output features
        common_dim = self.cross_attention.common_dim
        total_input_size = in_features + sum(self.feature_dims.values()) + common_dim
        self.teacher_model.triplet_head = nn.Sequential(
            nn.LayerNorm(total_input_size),
            nn.Dropout(p=0.5),
            nn.Linear(total_input_size, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.num_classes["triplet"]),
        ).to(self.device)
        # We need to remove the original classification head
        self.teacher_model.head = nn.Identity()

        # Initialize student model (uses only the triplet head)
        self.student_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        in_features_student = self.student_model.num_features
        # Student triplet head (uses only the backbone features)
        self.student_model.triplet_head = nn.Sequential(
            nn.LayerNorm(in_features_student),
            nn.Dropout(p=0.5),
            nn.Linear(in_features_student, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.num_classes["triplet"]),
        ).to(self.device)
        self.student_model.head = nn.Identity()

        # Create optimizers and schedulers
        self.teacher_optimizer, self.teacher_scheduler = (
            self._create_optimizer_and_scheduler(self.teacher_model)
        )
        self.student_optimizer, self.student_scheduler = (
            self._create_optimizer_and_scheduler(self.student_model)
        )

    def _create_optimizer_and_scheduler(self, model):
        """Create optimizer and scheduler with parameter groups"""

        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (no_decay_params if "bias" in name else decay_params).append(param)

        optimizer = SGD(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True,
        )

        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=7)
        return optimizer, scheduler

    def _forward_pass(self, model, inputs, mode):
        """Perform forward pass through the model based on the model type [teacher, student]"""
        features = model(inputs)
        if mode == "teacher":
            # Get individual component predictions
            verb_logits = model.verb_head(features)
            instrument_logits = model.instrument_head(features)
            target_logits = model.target_head(features)
            # Apply cross-attention fusion
            attention_output = self.cross_attention(
                verb_logits, instrument_logits, target_logits
            )
            # Combine all features for triplet prediction
            combined_features = torch.cat(
                [
                    features,  # original features from backbone
                    verb_logits,  # verb predictions
                    instrument_logits,  # instrument predictions
                    target_logits,  # target predictions
                    attention_output,  # attention-fused features
                ],
                dim=1,
            )
            triplet_logits = model.triplet_head(combined_features)

            return {
                "verb": verb_logits,
                "instrument": instrument_logits,
                "target": target_logits,
                "triplet": triplet_logits,
            }

        elif mode == "student":
            # Student's forward pass (only triplet)
            triplet_logits = model.triplet_head(features)
            return {"triplet": triplet_logits}
        else:
            raise TypeError(
                "This method only accepts either the teacher or the student model"
            )

    def _compute_loss(self, outputs, labels, mode):
        """Compute the combined loss for all tasks"""
        total_loss = 0
        losses = {}
        task_weights = {"verb": 0.5, "instrument": 0.5, "target": 0.5, "triplet": 1.0}

        # Convert triplet_to_ivt mapping to tensor
        triplet_components = torch.tensor(
            [self.triplet_to_ivt[idx] for idx in range(len(self.triplet_to_ivt))],
            device=self.device,
        )  # Shape: (69, 3)

        for task in outputs:
            # BCE Loss
            loss = F.binary_cross_entropy_with_logits(outputs[task], labels[task])
            # # Add distillation loss for the teacher model
            # if mode == "student":
            # distillation_loss = F.binary_cross_entropy_with_logits(
            #     outputs[task], soft_labels[task]
            # )
            # # Gradually include the distillation loss
            # loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

            # Calculate the consistency loss
            if task == "triplet" and mode == "teacher":
                # First get the individual component outputs
                with torch.no_grad():
                    verb_probs = torch.sigmoid(outputs["verb"])
                    inst_probs = torch.sigmoid(outputs["instrument"])
                    target_probs = torch.sigmoid(outputs["target"])

                # Gather indices for all 69 triplets
                i_idx = triplet_components[:, 0]  # Instrument indices for all triplets
                v_idx = triplet_components[:, 1]  # Verb indices
                t_idx = triplet_components[:, 2]  # Target indices

                # Gather probabilities for each triplet component
                inst_p = inst_probs[:, i_idx]
                verb_p = verb_probs[:, v_idx]
                target_p = target_probs[:, t_idx]

                #  This product represents the expected triplet probability
                product_targets = inst_p * verb_p * target_p
                # This loss encourages the triplet predictions to align with the component products
                consistency_loss = F.binary_cross_entropy_with_logits(
                    outputs["triplet"], product_targets
                )
                # Add this along with the triplet task loss
                loss += consistency_loss * 0.5

            total_loss += task_weights[task] * loss
            losses[task] = loss.item()

        return total_loss, losses

    def _train_model(self, model, optimizer, lr_scheduler, mode, soft_labels=None):
        """Train either teacher or student model"""

        # Track best triplet mAP
        best_map = 0.0

        for epoch in range(self.num_epochs):
            model.train()
            epoch_losses = {}

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                # Only the triplet labels for the student model
                batch_labels = (
                    {"triplet": labels["triplet"].to(self.device)}
                    if mode == "student"
                    else {
                        task: labels[task].to(self.device)
                        for task in ["verb", "instrument", "target", "triplet"]
                    }
                )

                outputs = self._forward_pass(model, inputs, mode)

                # # Get batch soft labels if training student
                # batch_soft_labels = (
                #     {
                #         "triplet": soft_labels["triplet"][
                #             batch_idx
                #             * self.train_loader.batch_size : (batch_idx + 1)
                #             * self.train_loader.batch_size
                #         ].to(self.device)
                #     }
                #     if soft_labels
                #     else None
                # )

                total_loss, task_losses = self._compute_loss(
                    outputs, batch_labels, mode
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # Accumulate losses
                for task, loss in task_losses.items():
                    epoch_losses[task] = epoch_losses.get(task, 0) + loss

            # Validation
            self.logger.info(f"Validation Results - Epoch {epoch+1}/{self.num_epochs}:")
            val_metrics = self._validate_model(model, mode)
            triplet_map = val_metrics.get("triplet")
            # Update learning rate scheduler
            lr_scheduler.step(triplet_map)

            # Log metrics
            self.logger.info("Training Losses:")

            for task in epoch_losses:
                avg_loss = epoch_losses[task] / len(self.train_loader)
                self.logger.info(
                    f"{task.capitalize()} - Loss: {avg_loss:.4f}, mAP: {val_metrics.get(task, 0):.4f}"
                )

            self.logger.info("-" * 50)
            # Save best models for each task
            if triplet_map > best_map:
                best_map = triplet_map
                torch.save(
                    model.state_dict(), f"{self.dir_name}/best_model_triplet_{mode}.pth"
                )
                self.logger.info(f"New best triplet mAP: {triplet_map:.4f}")

    def _generate_soft_labels(self):
        """Generate soft labels using the trained teacher model"""
        self.teacher_model.eval()
        soft_labels = {"triplet": []}

        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(self.device)
                outputs = self._forward_pass(self.teacher_model, inputs, "teacher")
                soft_labels["triplet"].append(torch.sigmoid(outputs["triplet"]).cpu())

        return {"triplet": torch.cat(soft_labels["triplet"], dim=0)}

    def _validate_model(self, model, mode):
        """Validate model and compute metrics for all tasks"""

        model.eval()
        outputs = {}
        labels = {}

        with torch.no_grad():
            for inputs, batch_labels in self.val_loader:
                inputs = inputs.to(self.device)
                model_outputs = self._forward_pass(model, inputs, mode)
                for task in model_outputs:
                    if task not in outputs:
                        outputs[task] = []
                        labels[task] = []
                    outputs[task].append(model_outputs[task].cpu())
                    labels[task].append(batch_labels[task].cpu())

        task_metrics = {}
        for task in outputs:
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

            task_metrics[task]["per_class"] = {}
            class_aps = average_precision_score(true_labels, predictions, average=None)

            for i in range(len(class_aps)):
                class_preds = predictions[:, i]
                class_labels = true_labels[:, i]
                # Use the index_to_triplet mapping to get the original triplet ID
                if task == "triplet":
                    original_id = self.val_loader.dataset.index_to_triplet[i]
                    label_name = self.label_mappings[task].get(
                        original_id, f"Class_{original_id}"
                    )
                else:
                    label_name = self.label_mappings[task].get(i, f"Class_{i}")

                thresholds = np.arange(0.3, 0.7, 0.1)
                f1_scores = [
                    f1_score(class_labels, class_preds > t) for t in thresholds
                ]
                optimal_threshold = thresholds[np.argmax(f1_scores)]
                binary_preds = (class_preds > optimal_threshold).astype(int)
                task_metrics[task]["per_class"][label_name] = {
                    "AP": class_aps[i],
                    "precision": precision_score(class_labels, binary_preds),
                    "recall": recall_score(class_labels, binary_preds),
                    "f1": f1_score(class_labels, binary_preds),
                    "optimal_threshold": optimal_threshold,
                }

            # Log metrics
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

            if "per_class" in task_metrics[task]:
                for class_name, class_metrics in task_metrics[task][
                    "per_class"
                ].items():
                    self.logger.info(f" {class_name}:")
                    self.logger.info(f"  AP: {class_metrics['AP']:.4f}")
                    self.logger.info(f"  F1: {class_metrics['f1']:.4f}")
                    self.logger.info(f"  Precision: {class_metrics['precision']:.4f}")
                    self.logger.info(f"  Recall: {class_metrics['recall']:.4f}")
                    self.logger.info(
                        f"  Optimal threshold: {class_metrics['optimal_threshold']:.2f}"
                    )

        # Now we can safely return the overall mAP for each task
        return {task: task_metrics[task]["overall"]["mAP"] for task in task_metrics}

    def train(self):
        """Execute full training pipeline with self-distillation"""
        # Train teacher model
        self.logger.info("Training teacher model...")
        trainable_params_teacher = sum(
            p.numel() for p in self.teacher_model.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable parameters: {trainable_params_teacher:,}")
        self.logger.info("-" * 50)
        print("here")
        self._train_model(
            self.teacher_model,
            self.teacher_optimizer,
            self.teacher_scheduler,
            "teacher",
        )

        # # Train student model
        # self.logger.info("Training student model...")
        # trainable_params_student = sum(
        #     p.numel() for p in self.teacher_model.parameters() if p.requires_grad
        # )
        # self.logger.info(f"Trainable parameters: {trainable_params_student:,}")
        # self._train_model(
        #     self.student_model,
        #     self.student_optimizer,
        #     self.student_scheduler,
        #     "student",
        # )

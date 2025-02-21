import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn
import logging
from torchvision.models.video.swin_transformer import (
    swin3d_s,
    Swin3D_S_Weights,
    swin3d_t,
    Swin3D_T_Weights,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from modules import CustomSwin3D
import numpy as np


class AttentionModule(nn.Module):
    def __init__(self, feature_dims, num_layers=2, num_heads=4, common_dim=256):
        super().__init__()
        self.common_dim = common_dim
        self.component_embeddings = nn.ModuleDict(
            {k: nn.Linear(v, common_dim) for k, v in feature_dims.items()}
        )

        encoder_layer = nn.TransformerEncoderLayer(
            common_dim, num_heads, dim_feedforward=4 * common_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, verb_feat, inst_feat, target_feat):
        # Project features to common space
        verb_emb = self.component_embeddings["verb"](verb_feat).unsqueeze(1)
        inst_emb = self.component_embeddings["instrument"](inst_feat).unsqueeze(1)
        target_emb = self.component_embeddings["target"](target_feat).unsqueeze(1)

        # Concatenate as sequence tokens
        tokens = torch.cat([verb_emb, inst_emb, target_emb], dim=1)

        # Process through transformer
        transformed = self.transformer(tokens)

        # Flatten the sequence dimension
        batch_size = transformed.size(0)
        return transformed.reshape(batch_size, -1)


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
        guidance_scale: float = 0.8,
    ):
        self.num_epochs = num_epochs
        self.alpha = min(1.0, num_epochs / warmup_epochs)
        self.temperature = temperature
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.label_mappings = label_mappings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_layer_dim = hidden_layer_dim
        self.guidance_scale = guidance_scale

        self.device = device
        self.logger = logger
        self.dir_name = dir_name

        self.triplet_to_ivt = torch.tensor(
            [triplet_to_ivt[idx] for idx in range(len(triplet_to_ivt))],
            device=device,
        )
        self.MI = torch.zeros(
            (self.num_classes["instrument"], self.num_classes["triplet"])
        ).to(device)
        self.MV = torch.zeros(
            (self.num_classes["verb"], self.num_classes["triplet"])
        ).to(device)
        self.MT = torch.zeros(
            (self.num_classes["target"], self.num_classes["triplet"])
        ).to(device)
        for t, (inst, verb, target) in triplet_to_ivt.items():
            self.MI[inst, t] = 1
            self.MV[verb, t] = 1
            self.MT[target, t] = 1

        self.feature_dims = {k: v for k, v in num_classes.items() if k != "triplet"}
        self.attention_module = AttentionModule(self.feature_dims).to(self.device)
        self._configure_models()

    def _configure_models(self):
        # Initialize the teacher model
        # self.teacher_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        # in_features = self.teacher_model.num_features
        self.teacher_model = CustomSwin3D().to(self.device)
        in_features = self.teacher_model.out_channels
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
        common_dim = self.attention_module.common_dim
        total_input_size = (
            in_features + sum(self.feature_dims.values()) + 3 * common_dim
        )
        self.teacher_model.triplet_head = nn.Sequential(
            nn.LayerNorm(total_input_size),
            nn.Dropout(p=0.5),
            nn.Linear(total_input_size, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.num_classes["triplet"]),
        ).to(self.device)
        # We need to remove the original classification head
        # self.teacher_model.head = nn.Identity()

        # Initialize student model (uses only the triplet head)
        self.student_model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT).to(self.device)
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
            attention_output = self.attention_module(
                verb_logits, instrument_logits, target_logits
            )
            # Combine all features for triplet prediction
            combined_features = torch.cat(
                [
                    features,  # original features from backbone
                    verb_logits,  # verb predictions
                    instrument_logits,  # instrument predictions
                    target_logits,  # target predictions
                    attention_output,  # features from the attention module
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

    def _compute_loss(self, outputs, labels, inputs, mode):
        """Compute the combined loss for all tasks

        Args:
            outputs (dict): Model outputs for each task
            labels (dict): Ground truth labels for each task
            inputs (torch.Tensor): Input tensor needed for teacher predictions
            mode (str): Either "teacher" or "student"
        """
        total_loss = 0
        losses = {}
        task_weights = {"verb": 0.5, "instrument": 0.5, "target": 0.5, "triplet": 1.0}

        for task in outputs:
            # Standard BCE Loss
            loss = F.binary_cross_entropy_with_logits(outputs[task], labels[task])
            # Combine it with distillation loss for the student model
            if mode == "student":
                # Get teacher's predictions
                self.teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = self._forward_pass(
                        self.teacher_model, inputs, "teacher"
                    )
                    teacher_logits = teacher_outputs["triplet"]
                    # Apply temperature scaling
                    soft_targets = torch.sigmoid(teacher_logits / self.temperature)

                # Calculate distillation loss
                student_logits = outputs[task] / self.temperature
                distillation_loss = F.binary_cross_entropy_with_logits(
                    student_logits, soft_targets
                ) * (
                    self.temperature**2
                )  # Scale the loss back

                #  Gradually include the distillation loss
                total_loss = (1 - self.alpha) * loss + self.alpha * distillation_loss
                losses[task] = total_loss.item()

                return total_loss, losses

            total_loss += task_weights[task] * loss
            losses[task] = loss.item()

        return total_loss, losses

    def _train_model(self, model, optimizer, lr_scheduler, mode):
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

                total_loss, task_losses = self._compute_loss(
                    outputs, batch_labels, inputs, mode
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
                self.logger.info("-" * 50)

    def _validate_model(self, model, mode):
        """Validate model and compute metrics for all tasks"""

        model.eval()
        all_predictions = {}
        all_labels = {}

        with torch.no_grad():
            for inputs, batch_labels in self.val_loader:
                inputs = inputs.to(self.device)
                model_outputs = self._forward_pass(model, inputs, mode)

                # Convert all outputs to probabilities at once
                task_probabilities = {
                    task: torch.sigmoid(outputs)
                    for task, outputs in model_outputs.items()
                }

                # Apply guidance if it is the teacher model
                if mode == "teacher":
                    # Get guidance from individual tasks
                    guidance_inst = torch.matmul(
                        task_probabilities["instrument"], self.MI
                    )
                    guidance_verb = torch.matmul(task_probabilities["verb"], self.MV)
                    guidance_target = torch.matmul(
                        task_probabilities["target"], self.MT
                    )

                    # Combine guidance signals
                    guidance = guidance_inst * guidance_verb * guidance_target

                    # Apply guidance with a scale factor
                    guided_triplet_probs = (
                        1 - self.guidance_scale
                    ) * task_probabilities["triplet"] + self.guidance_scale * (
                        guidance * task_probabilities["triplet"]
                    )

                    # Update the triplet predictions with guided probabilities
                    task_probabilities["triplet"] = guided_triplet_probs

                # Store probabilities and labels
                for task in task_probabilities:
                    if task not in all_predictions:
                        all_predictions[task] = []
                        all_labels[task] = []
                    all_predictions[task].append(task_probabilities[task].cpu())
                    all_labels[task].append(batch_labels[task].cpu())

        task_metrics = {}
        for task in all_predictions:
            predictions = torch.cat(all_predictions[task], dim=0).numpy()
            labels = torch.cat(all_labels[task], dim=0).numpy()

            ####################
            # Calculate AP per class without averaging
            class_aps = average_precision_score(labels, predictions, average=None)
            class_aps = self._resolve_nan(class_aps)

            # Calculate mean AP excluding NaN values
            mean_ap = np.nanmean(class_aps)

            task_metrics[task] = {"mAP": mean_ap}

            # Log metrics
            self.logger.info(f"{task.upper()} METRICS:")
            self.logger.info(f"  Overall mAP: {mean_ap:.4f}")

        return {task: task_metrics[task]["mAP"] for task in task_metrics}
        ####################

        # task_metrics[task] = {
        #     "overall": {
        #         "mAP": average_precision_score(
        #             true_labels, predictions, average="micro"
        #         ),
        #         "macro_AP": average_precision_score(
        #             true_labels, predictions, average="macro"
        #         ),
        #         "weighted_AP": average_precision_score(
        #             true_labels, predictions, average="weighted"
        #         ),
        #     },
        #     "per_class": {},
        # }

        # task_metrics[task]["per_class"] = {}
        # class_aps = average_precision_score(true_labels, predictions, average=None)

        #     for i in range(len(class_aps)):
        #         class_preds = predictions[:, i]
        #         class_labels = true_labels[:, i]
        #         # Use the index_to_triplet mapping to get the original triplet ID
        #         if task == "triplet":
        #             original_id = self.val_loader.dataset.index_to_triplet[i]
        #             label_name = self.label_mappings[task].get(
        #                 original_id, f"Class_{original_id}"
        #             )
        #         else:
        #             label_name = self.label_mappings[task].get(i, f"Class_{i}")

        #         thresholds = np.arange(0.3, 0.7, 0.1)
        #         f1_scores = [
        #             f1_score(class_labels, class_preds > t) for t in thresholds
        #         ]
        #         optimal_threshold = thresholds[np.argmax(f1_scores)]
        #         binary_preds = (class_preds > optimal_threshold).astype(int)
        #         task_metrics[task]["per_class"][label_name] = {
        #             "AP": class_aps[i],
        #             "precision": precision_score(class_labels, binary_preds),
        #             "recall": recall_score(class_labels, binary_preds),
        #             "f1": f1_score(class_labels, binary_preds),
        #             "optimal_threshold": optimal_threshold,
        #         }

        #     # Log metrics
        #     self.logger.info(f"{task.upper()} METRICS:")
        #     self.logger.info(
        #         f"  Overall mAP: {task_metrics[task]['overall']['mAP']:.4f}"
        #     )
        #     self.logger.info(
        #         f"  Macro AP: {task_metrics[task]['overall']['macro_AP']:.4f}"
        #     )
        #     self.logger.info(
        #         f"  Weighted AP: {task_metrics[task]['overall']['weighted_AP']:.4f}"
        #     )

        #     if "per_class" in task_metrics[task]:
        #         for class_name, class_metrics in task_metrics[task][
        #             "per_class"
        #         ].items():
        #             self.logger.info(f" {class_name}:")
        #             self.logger.info(f"  AP: {class_metrics['AP']:.4f}")
        #             self.logger.info(f"  F1: {class_metrics['f1']:.4f}")
        #             self.logger.info(f"  Precision: {class_metrics['precision']:.4f}")
        #             self.logger.info(f"  Recall: {class_metrics['recall']:.4f}")
        #             self.logger.info(
        #                 f"  Optimal threshold: {class_metrics['optimal_threshold']:.2f}"
        #             )

        # return {task: task_metrics[task]["overall"]["mAP"] for task in task_metrics}

    def train(self):
        """Execute full training pipeline with self-distillation"""

        # Train the teacher model
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
            "teacher",
        )

        # Train the student model
        self.logger.info("Training student model...")
        trainable_params_student = sum(
            p.numel() for p in self.teacher_model.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable parameters: {trainable_params_student:,}")
        self._train_model(
            self.student_model,
            self.student_optimizer,
            self.student_scheduler,
            "student",
        )

    def _resolve_nan(self, class_aps):
        equiv_nan = ["-0", "-0.", "-0.0", "-.0"]
        class_aps = list(map(str, class_aps))
        class_aps = [np.nan if x in equiv_nan else x for x in class_aps]
        class_aps = np.array(list(map(float, class_aps)))
        return class_aps

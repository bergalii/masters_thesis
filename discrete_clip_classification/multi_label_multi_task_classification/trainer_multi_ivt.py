import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn
import logging
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from ivtmetrics import Recognition


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
        num_triplet_combinations: int,
        label_mappings: dict,
        device,
        logger: logging.Logger,
        dir_name: str,
        warmup_epochs: int = 5,
        temperature: float = 2.0,
    ):
        self.num_epochs = num_epochs
        self.alpha = min(1.0, num_epochs / warmup_epochs)
        self.temperature = temperature
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.num_triplet_combinations = num_triplet_combinations
        self.label_mappings = label_mappings
        self.device = device
        self.logger = logger
        self.dir_name = dir_name

        self.task_to_component = {
            "verb": "v",
            "instrument": "i",
            "target": "t",
            "triplet": "ivt",
        }
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

            # Triplet combination prediction head using fused features
            model.triplet_head = nn.Sequential(
                nn.LayerNorm(in_features + sum(self.num_classes.values())),
                nn.Dropout(p=0.5),
                nn.Linear(in_features + sum(self.num_classes.values()), 512),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, self.num_triplet_combinations),
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
        # Get individual component predictions
        verb_logits = model.verb_head(features)
        instrument_logits = model.instrument_head(features)
        target_logits = model.target_head(features)

        # Combine all features for triplet prediction
        combined_features = torch.cat(
            [features, verb_logits, instrument_logits, target_logits], dim=1
        )

        triplet_logits = model.triplet_head(combined_features)

        return {
            "verb": verb_logits,
            "instrument": instrument_logits,
            "target": target_logits,
            "triplet": triplet_logits,
        }

    def _compute_loss(self, outputs, labels, soft_labels=None):
        """Compute the combined loss for all tasks"""
        total_loss = 0
        losses = {}

        for task in ["verb", "instrument", "target", "triplet"]:
            # Standard cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(outputs[task], labels[task])

            # Add distillation loss if soft labels are provided
            if soft_labels is not None:
                distillation_loss = F.binary_cross_entropy_with_logits(
                    outputs[task], soft_labels[task]
                )
                loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

            total_loss += loss
            losses[task] = loss.item()

        return total_loss, losses

    def _train_model(
        self, model, optimizer, lr_scheduler, is_teacher=True, soft_labels=None
    ):
        """Train either teacher or student model"""
        best_map = {task: 0.0 for task in ["verb", "instrument", "target", "triplet"]}

        for epoch in range(self.num_epochs):
            model.train()
            epoch_losses = {
                task: 0.0 for task in ["verb", "instrument", "target", "triplet"]
            }

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                batch_labels = {
                    task: labels[task].to(self.device)
                    for task in ["verb", "instrument", "target", "triplet"]
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
                        for task in ["verb", "instrument", "target", "triplet"]
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
            for task in ["verb", "instrument", "target", "triplet"]:
                avg_loss = epoch_losses[task] / len(self.train_loader)
                self.logger.info(f"{task}: Loss={avg_loss:.4f}")
            self.logger.info("-" * 50)

            # Save best models for each task
            for task in ["verb", "instrument", "target", "triplet"]:
                if val_maps[task] > best_map[task]:
                    self.logger.info(f"Saving the best model for the task {task}")
                    best_map[task] = val_maps[task]
                    torch.save(
                        model.state_dict(), f"{self.dir_name}/best_model_{task}.pth"
                    )

    def _generate_soft_labels(self):
        """Generate soft labels using trained teacher model"""
        self.teacher_model.eval()
        soft_labels = {"verb": [], "instrument": [], "target": [], "triplet": []}

        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(self.device)
                outputs = self._forward_pass(self.teacher_model, inputs)

                for task in ["verb", "instrument", "target", "triplet"]:
                    soft_labels[task].append(torch.sigmoid(outputs[task]).cpu())

        return {task: torch.cat(labels, dim=0) for task, labels in soft_labels.items()}

    def _validate_model(self, model):
        """Validate model and compute metrics for all tasks"""
        model.eval()
        # Initialize ivtmetrics Recognition objects for each task

        recognize = Recognition(num_class=self.num_triplet_combinations)
        recognize.reset_global()

        with torch.no_grad():
            for inputs, batch_labels in self.val_loader:
                inputs = inputs.to(self.device)
                model_outputs = self._forward_pass(model, inputs)
                predictions = torch.sigmoid(model_outputs["triplet"]).cpu().numpy()
                binary_preds = (predictions > 0.5).astype(int)
                print(binary_preds[0])
                true_labels = batch_labels["triplet"].cpu().numpy()
                print(true_labels[0])
                recognize.update(true_labels, binary_preds)
                recognize.video_end()

            # Compute metrics after processing all videos
            results = {}
            # Compute component-wise metrics
            results["instrument"] = recognize.compute_video_AP("i")
            results["verb"] = recognize.compute_video_AP("v")
            results["target"] = recognize.compute_video_AP("t")
            results["triplet"] = recognize.compute_video_AP("ivt")

            for task in ["verb", "instrument", "target", "triplet"]:
                self.logger.info(" {task} results:")
                self.logger.info(f"{task}: mAP={results[task]['mAP']:.4f}, ")
            return results

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

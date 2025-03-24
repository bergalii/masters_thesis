import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn
import logging
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from recognition import Recognition
import copy
from modules import AttentionModule, MultiTaskHead


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
        attention_module_common_dim: int,
        hidden_layer_dim: int,
        gradient_clipping: float,
        consistency_loss_weight: float,
        guidance_scale: float,
        warmup_epochs: int = 5,
        temperature: float = 2.0,
    ):
        self.num_epochs = num_epochs
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs
        self.temperature = temperature
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.label_mappings = label_mappings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_layer_dim = hidden_layer_dim
        self.attention_module_common_dim = attention_module_common_dim
        self.gradient_clipping = gradient_clipping
        self.guidance_scale = guidance_scale
        self.consistency_loss_weight = consistency_loss_weight
        self.task_weights = {
            "verb": 0.5,
            "instrument": 0.5,
            "target": 1.0,
            "triplet": 1.5,
        }
        self.device = device
        self.logger = logger
        self.dir_name = dir_name

        self._initialize_guidance_matrices(triplet_to_ivt)

        # Mapping to go from triplet id to individual task ids
        self.triplet_to_ivt = torch.tensor(
            [triplet_to_ivt[idx] for idx in range(len(triplet_to_ivt))],
            device=device,
        )
        self.feature_dims = {k: v for k, v in num_classes.items() if k != "triplet"}
        self._configure_models()

    def _initialize_guidance_matrices(self, triplet_to_ivt):
        """Initialize the matrices for guidance"""
        self.MI = torch.zeros(
            (self.num_classes["instrument"], self.num_classes["triplet"])
        ).to(self.device)
        self.MV = torch.zeros(
            (self.num_classes["verb"], self.num_classes["triplet"])
        ).to(self.device)
        self.MT = torch.zeros(
            (self.num_classes["target"], self.num_classes["triplet"])
        ).to(self.device)

        for t, (inst, verb, target) in triplet_to_ivt.items():
            self.MI[inst, t] = 1
            self.MV[verb, t] = 1
            self.MT[target, t] = 1

    def _configure_models(self):
        """Initialize and configure teacher and student models"""
        # Initialize the teacher model
        self.teacher_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        # We need to remove the original classification head
        self.teacher_model.head = nn.Identity()
        in_features = self.teacher_model.num_features

        # Teacher heads
        self.teacher_model.verb_head = MultiTaskHead(
            in_features, self.num_classes["verb"], self.hidden_layer_dim
        ).to(self.device)
        self.teacher_model.instrument_head = MultiTaskHead(
            in_features, self.num_classes["instrument"], self.hidden_layer_dim
        ).to(self.device)
        self.teacher_model.target_head = MultiTaskHead(
            in_features, self.num_classes["target"], self.hidden_layer_dim
        ).to(self.device)

        self.teacher_model.attention_module = AttentionModule(
            self.feature_dims,
            self.hidden_layer_dim,
            self.attention_module_common_dim,
            num_heads=4,
            dropout=0.3,
        ).to(self.device)

        total_input_size = (
            in_features  # Backbone features
            + self.attention_module_common_dim  # Attention module output
            + sum(self.feature_dims.values())  # Probability outputs from each tas
        )

        self.teacher_model.triplet_head = MultiTaskHead(
            total_input_size, self.num_classes["triplet"], self.hidden_layer_dim
        ).to(self.device)

        # Deep copy the teacher model to create identical student model
        self.student_model = copy.deepcopy(self.teacher_model)

        # Create optimizers and schedulers
        self.teacher_optimizer, self.teacher_scheduler, self.initial_lrs = (
            self._create_optimizer_and_scheduler(self.teacher_model)
        )
        self.student_optimizer, self.student_scheduler, _ = (
            self._create_optimizer_and_scheduler(self.student_model)
        )

    def _create_optimizer_and_scheduler(self, model):
        """Create optimizer and scheduler with parameter groups"""

        # Separate parameters into four groups
        backbone_params_decay = []
        backbone_params_no_decay = []
        head_params_decay = []
        head_params_no_decay = []

        for name, param in model.named_parameters():

            # Check if the parameter should have weight decay
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay = True
            else:
                no_decay = False

            # Determine if it's a backbone or head parameter
            if "head" in name or "attention_module" in name:
                if no_decay:
                    head_params_no_decay.append(param)
                else:
                    head_params_decay.append(param)
            else:
                if no_decay:
                    backbone_params_no_decay.append(param)
                else:
                    backbone_params_decay.append(param)

        optimizer = AdamW(
            [
                {
                    "params": backbone_params_decay,
                    "lr": self.learning_rate / 10,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": backbone_params_no_decay,
                    "lr": self.learning_rate / 10,
                    "weight_decay": 0.0,
                },
                {
                    "params": head_params_decay,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": head_params_no_decay,
                    "lr": self.learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        )

        # Store the initial learning rates for later use
        initial_lrs = [group["lr"] for group in optimizer.param_groups]

        # scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
        # OneCycleLR provides good performance with a warm-up phase followed by
        # a gradual cosine annealing decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                self.learning_rate / 10,
                self.learning_rate / 10,
                self.learning_rate,
                self.learning_rate,
            ],
            total_steps=len(self.train_loader) * int(self.num_epochs * 0.7),
            pct_start=0.1,  # 10% of training time is warmup
            anneal_strategy="cos",
            div_factor=5.0,  # initial_lr = max_lr / div_factor
            final_div_factor=10.0,  # final_lr = initial_lr / final_div_factor
        )
        return optimizer, scheduler, initial_lrs

    def _forward_pass(self, model, inputs):
        """Perform forward pass through the model based on the model type [teacher, student]"""
        backbone_features = model(inputs)
        # Get individual component predictions
        verb_logits, verb_hidden = model.verb_head(backbone_features)
        instrument_logits, inst_hidden = model.instrument_head(backbone_features)
        target_logits, target_hidden = model.target_head(backbone_features)

        attention_output = model.attention_module(
            verb_hidden,
            verb_logits,
            inst_hidden,
            instrument_logits,
            target_hidden,
            target_logits,
        )

        # Combine all features for triplet prediction
        combined_features = torch.cat(
            [
                backbone_features,  # original features from backbone
                attention_output,  # attention-fused features
                torch.sigmoid(verb_logits),  # Probability predictions
                torch.sigmoid(instrument_logits),
                torch.sigmoid(target_logits),
            ],
            dim=1,
        )

        triplet_logits, _ = model.triplet_head(combined_features)

        return {
            "verb": verb_logits,
            "instrument": instrument_logits,
            "target": target_logits,
            "triplet": triplet_logits,
        }

    def _calculate_consistency_loss(self, triplet_logits, component_logits):
        """
        Enforce consistency between triplet predictions and component predictions

        Args:
            triplet_logits: Logits from the triplet head
            component_logits: Dictionary of logits from component heads
            alpha: Weighting factor for the consistency loss
        """
        # Get component probabilities
        inst_probs = torch.sigmoid(component_logits["instrument"])
        verb_probs = torch.sigmoid(component_logits["verb"])
        target_probs = torch.sigmoid(component_logits["target"])

        # Get triplet probabilities
        triplet_probs = torch.sigmoid(triplet_logits)

        # Calculate expected triplet probabilities from individual components
        expected_triplets = torch.zeros_like(triplet_probs)
        for t_idx in range(self.num_classes["triplet"]):
            i_idx, v_idx, tg_idx = self.triplet_to_ivt[t_idx]

            # Use geometric mean to combine probabilities to avoid extremely small values
            combined_prob = torch.pow(
                torch.clamp(
                    inst_probs[:, i_idx]
                    * verb_probs[:, v_idx]
                    * target_probs[:, tg_idx],
                    min=1e-6,
                ),
                1 / 3,
            )
            expected_triplets[:, t_idx] = combined_prob

        # Calculate binary cross entropy between predicted and expected
        consistency_loss = F.binary_cross_entropy(
            triplet_probs,
            expected_triplets.detach(),  # Detach to avoid backprop through components
            reduction="mean",
        )

        return self.consistency_loss_weight * consistency_loss

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

        # Get teacher's predictions for student mode
        if mode == "student":
            with torch.no_grad():
                teacher_outputs = self._forward_pass(self.teacher_model, inputs)

        for task in outputs:
            # Standard BCE Loss with ground truth labels
            gt_loss = F.binary_cross_entropy_with_logits(outputs[task], labels[task])

            # Combine with distillation loss for the student model
            if mode == "student":
                # Apply temperature scaling to teacher and student outputs
                teacher_logits = teacher_outputs[task] / self.temperature
                student_logits = outputs[task] / self.temperature

                # Get soft targets from teacher
                soft_targets = torch.sigmoid(teacher_logits)

                # Calculate distillation loss
                distillation_loss = F.binary_cross_entropy_with_logits(
                    student_logits, soft_targets
                ) * (
                    self.temperature**2
                )  # Scale the loss back

                # Gradually balance between ground truth and teacher guidance
                task_loss = (1 - self.alpha) * gt_loss + self.alpha * distillation_loss
            else:
                # For teacher model, just use ground truth loss
                task_loss = gt_loss

            # Apply task-specific weight
            total_loss += self.task_weights[task] * task_loss
            losses[task] = task_loss.item()

        consistency_loss = self._calculate_consistency_loss(
            outputs["triplet"],
            {k: outputs[k] for k in ["verb", "instrument", "target"]},
        )
        total_loss += consistency_loss

        return total_loss, losses

    def _update_task_weights(self, validation_metrics):
        """Update task weights based on validation performance"""
        # Get a list of tasks excluding 'triplet'
        component_tasks = [task for task in self.task_weights if task != "triplet"]

        # Calculate total inverse performance for component tasks only
        total_inverse = sum(
            1.0 / (validation_metrics[task] + 0.1) for task in component_tasks
        )

        # Update weights for component tasks only
        for task in component_tasks:
            # Normalize weights relative to each other, while keeping their average at 0.5
            self.task_weights[task] = (
                (1.0 / (validation_metrics[task] + 0.1)) / total_inverse
            ) * 1.0

        # Keep triplet weight fixed at 1.5
        self.task_weights["triplet"] = 1.5

        self.logger.info(f"Task weights: {self.task_weights}")

    def _train_model(self, model, optimizer, lr_scheduler, mode):
        """Train either teacher or student model"""

        # Track best triplet mAP
        best_map = 0.0

        for epoch in range(int(self.num_epochs * 0.7)):
            model.train()
            epoch_losses = {}

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                # Only the triplet labels for the student model
                batch_labels = {
                    task: labels[task].to(self.device)
                    for task in ["verb", "instrument", "target", "triplet"]
                }

                outputs = self._forward_pass(model, inputs)

                total_loss, task_losses = self._compute_loss(
                    outputs, batch_labels, inputs, mode
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.gradient_clipping
                )
                optimizer.step()
                lr_scheduler.step()

                # Accumulate losses
                for task, loss in task_losses.items():
                    epoch_losses[task] = epoch_losses.get(task, 0) + loss

            ## Validation ##
            self.logger.info(
                f"Validation Results - Epoch {epoch+1}/{int(self.num_epochs * 0.7)}:"
            )
            val_metrics = self._validate_model(model, mode)

            triplet_map = val_metrics.get("triplet")
            # Log the new learning rate
            current_lrs = [group["lr"] for group in optimizer.param_groups]
            if len(current_lrs) == 1:
                self.logger.info(f"Learning rate: {current_lrs[0]:.6f}")
            else:
                self.logger.info(
                    f"Learning rates: {[f'{lr:.6f}' for lr in current_lrs]}"
                )

            # Update dynamic weights
            self._update_task_weights(val_metrics)

            if mode == "student":
                self.alpha = min(1.0, (epoch + 1) / self.warmup_epochs)

            # Log metrics
            self.logger.info("Training Results:")

            for task in epoch_losses:
                avg_loss = epoch_losses[task] / len(self.train_loader)
                self.logger.info(
                    f"{task.capitalize()} - Loss: {avg_loss:.4f}, mAP: {val_metrics.get(task, 0):.4f}"
                )

            self.logger.info("-" * 50)
            # Save best models for each task
            if triplet_map > best_map:
                best_map = triplet_map
                torch.save(model.state_dict(), f"{self.dir_name}/best_model_{mode}.pth")
                self.logger.info(f"New best triplet mAP: {triplet_map:.4f}")
                self.logger.info("-" * 50)

    def _train_model_components(self, model, optimizer):
        """Train only component tasks (verb, instrument, target)"""

        # Fixed rates for component training)
        for i, param_group in enumerate(optimizer.param_groups):
            if i < 2:  # Backbone parameter groups (first two groups)
                param_group["lr"] = self.learning_rate / 10
            else:  # Head parameter groups (last two groups)
                param_group["lr"] = self.learning_rate

        for epoch in range(int(self.num_epochs * 0.3)):
            model.train()
            epoch_losses = {}

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                batch_labels = {
                    task: labels[task].to(self.device)
                    for task in ["verb", "instrument", "target"]
                }

                # Forward pass without triplet head
                backbone_features = model(inputs)
                verb_logits, _ = model.verb_head(backbone_features)
                instrument_logits, _ = model.instrument_head(backbone_features)
                target_logits, _ = model.target_head(backbone_features)

                outputs = {
                    "verb": verb_logits,
                    "instrument": instrument_logits,
                    "target": target_logits,
                }

                # Skip triplet loss
                total_loss = 0
                task_losses = {}

                for task in outputs:
                    task_loss = F.binary_cross_entropy_with_logits(
                        outputs[task], batch_labels[task]
                    )
                    total_loss += self.task_weights[task] * task_loss
                    task_losses[task] = task_loss.item()

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.gradient_clipping
                )
                optimizer.step()

                # Accumulate losses
                for task, loss in task_losses.items():
                    epoch_losses[task] = epoch_losses.get(task, 0) + loss

            # Log metrics
            self.logger.info(f"Phase 1 - Epoch {epoch+1}/{int(self.num_epochs * 0.3)}:")
            for task in epoch_losses:
                avg_loss = epoch_losses[task] / len(self.train_loader)
                self.logger.info(f"{task.capitalize()} - Loss: {avg_loss:.4f}")

    def _validate_model(self, model, mode):
        """Validate model and compute metrics for all tasks"""

        model.eval()

        recognize = Recognition(num_class=self.num_classes["triplet"])
        # Reset for the current validation run
        recognize.reset()

        with torch.no_grad():
            for inputs_batch, batch_labels in self.val_loader:
                # inputs have dimension [B, N, C, T, H, W]
                batch_size = inputs_batch.shape[0]
                num_clips = inputs_batch.shape[1]

                # Process each video in the batch
                for b in range(batch_size):
                    video_outputs = {
                        task: [] for task in ["verb", "instrument", "target", "triplet"]
                    }
                    # Process each clip individually
                    for c in range(num_clips):
                        # Extract single clip: [C, T, H, W]
                        clip = (
                            inputs_batch[b, c].unsqueeze(0).to(self.device)
                        )  # Add batch dimension

                        # Get the predictions for this clip
                        clip_outputs = self._forward_pass(model, clip)

                        # Store predictions for each task
                        for task, outputs in clip_outputs.items():
                            video_outputs[task].append(outputs)

                    # Average predictions across clips for each task
                    task_logits = {}
                    for task, outputs_list in video_outputs.items():
                        # Concatenate along batch dimension then average
                        outputs_tensor = torch.cat([o for o in outputs_list], dim=0)
                        task_logits[task] = torch.mean(
                            outputs_tensor, dim=0, keepdim=True
                        )

                    # Convert all task logits to probabilities
                    task_probabilities = {
                        task: torch.sigmoid(logits)
                        for task, logits in task_logits.items()
                    }

                    # Get guidance from individual tasks
                    guidance_inst = torch.matmul(
                        task_probabilities["instrument"], self.MI
                    )
                    guidance_verb = torch.matmul(task_probabilities["verb"], self.MV)
                    guidance_target = torch.matmul(
                        task_probabilities["target"], self.MT
                    )

                    # Combine guidance outputs
                    guidance = guidance_inst * guidance_verb * guidance_target

                    # Apply guidance with a scale factor
                    guided_triplet_probs = (
                        1 - self.guidance_scale
                    ) * task_probabilities["triplet"] + self.guidance_scale * (
                        guidance * task_probabilities["triplet"]
                    )

                    # Update with triplet predictions and the label for the video
                    predictions = guided_triplet_probs.cpu().numpy()
                    labels = batch_labels["triplet"][b].unsqueeze(0).cpu().numpy()

                    # Update the recognizer with the current video
                    recognize.update(labels, predictions)

        # Compute and log metrics
        task_metrics = {}
        component_map = {
            "triplet": "ivt",
            "instrument": "i",
            "verb": "v",
            "target": "t",
        }

        for task, component in component_map.items():
            # Calculate metrics for this component
            results = recognize.compute_AP(component=component)

            # Store mean AP and class APs
            mean_ap = results["mAP"]
            class_aps = results["AP"]

            # Initialize task metrics
            task_metrics[task] = {"mAP": mean_ap, "per_class": {}}

            # Log the results
            self.logger.info(f"{task.upper()} METRICS:")
            self.logger.info(f"  Overall mAP: {mean_ap:.4f}")

            # Store and log per-class metrics
            for i in range(len(class_aps)):

                # Get the class name based on the component
                if task == "triplet":
                    original_id = self.val_loader.dataset.index_to_triplet[i]
                    label_name = self.label_mappings[task].get(
                        original_id, f"Class_{original_id}"
                    )
                else:
                    label_name = self.label_mappings[task].get(i, f"Class_{i}")

                # Store AP for this class
                task_metrics[task]["per_class"][label_name] = {"AP": class_aps[i]}
                # Log per-class AP
                self.logger.info(f"  {label_name}:")
                self.logger.info(f"    AP: {class_aps[i]:.4f}")

        return {task: metrics["mAP"] for task, metrics in task_metrics.items()}

    def train(self):
        """Execute training with curriculum learning approach"""

        self.logger.info("Training teacher model...")
        total_trainable_params = sum(
            p.numel() for p in self.teacher_model.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable parameters: {total_trainable_params:,}")
        self.logger.info("-" * 50)

        # Phase 1: Train component tasks only
        self.logger.info("Phase 1: Training component tasks...")

        # Freeze the triplet head parameters to prevent updates
        for param in self.teacher_model.triplet_head.parameters():
            param.requires_grad = False

        for param in self.teacher_model.attention_module.parameters():
            param.requires_grad = False

        # self._train_model_components(self.teacher_model, self.teacher_optimizer)

        # Phase 2: Freeze component heads and train only triplet task
        self.logger.info("Phase 2: Training the whole teacher model...")

        # Unfreeze triplet head
        for param in self.teacher_model.triplet_head.parameters():
            param.requires_grad = True

        # Unfreeze attention module
        for param in self.teacher_model.attention_module.parameters():
            param.requires_grad = True

        self._train_model(
            self.teacher_model,
            self.teacher_optimizer,
            self.teacher_scheduler,
            "teacher",
        )

        # After teacher training completes, load the best teacher model for distillation
        best_teacher_path = f"{self.dir_name}/best_model_teacher.pth"
        self.logger.info(
            f"Loading best teacher model from {best_teacher_path} for distillation..."
        )
        # Load the best teacher model weights
        teacher_state_dict = torch.load(best_teacher_path)
        self.teacher_model.load_state_dict(teacher_state_dict)
        self.teacher_model.eval()

        # Train the student model
        self.logger.info("Training the student model...")
        trainable_params_student = sum(
            p.numel() for p in self.student_model.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable parameters: {trainable_params_student:,}")
        # Freeze the triplet head parameters to prevent updates
        for param in self.student_model.triplet_head.parameters():
            param.requires_grad = False

        for param in self.student_model.attention_module.parameters():
            param.requires_grad = False

        self._train_model_components(self.student_model, self.student_optimizer)

        # Unfreeze triplet head
        for param in self.student_model.triplet_head.parameters():
            param.requires_grad = True

        # Unfreeze attention module
        for param in self.student_model.attention_module.parameters():
            param.requires_grad = True

        self._train_model(
            self.student_model,
            self.student_optimizer,
            self.student_scheduler,
            "student",
        )

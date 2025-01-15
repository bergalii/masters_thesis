from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


class SelfDistillationTrainer:
    def __init__(
        self,
        num_epochs: int,
        batch_size: int,
        train_loader,
        val_loader,
        transition_matrix,
        action_durations,
        warmup_epochs: int = 1,
        temperature: float = 2.0,
    ):
        self.num_epochs = num_epochs
        self.alpha = min(
            1.0, num_epochs / warmup_epochs
        )  # Gradually increase from 0 to 1 the impact of distill loss
        self.temperature = temperature
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.transition_matrix = torch.FloatTensor(transition_matrix).to(DEVICE)
        self.previous_prediction = None
        self.action_durations = action_durations
        # self.temperature = max(1.0, initial_temperature * (1 - epoch/num_epochs)) gradually increase temp
        self._configure_models()

    def _configure_models(self):
        # Initialize models, optimizer, scheduler
        self.teacher_model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT).to(
            DEVICE
        )  # Your model architecture
        self.student_model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT).to(
            DEVICE
        )  # Same architecture for self-distillation
        # 2. Modify their heads
        for model in [self.teacher_model, self.student_model]:
            model.head = nn.Sequential(
                nn.LayerNorm(model.num_features),
                nn.Dropout(p=0.5),
                nn.Linear(model.num_features, 512),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, NUM_CLASSES),
            ).to(DEVICE)

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
        print("Training teacher model...")
        self._train_model(
            self.teacher_model,
            self.teacher_optimizer,
            self.teacher_scheduler,
            is_teacher=True,
        )

        # Generate soft labels using teacher
        print("Generating soft labels...")
        soft_labels = self._generate_soft_labels()

        # Train student model using soft labels
        print("Training student model...")
        self._train_model(
            self.student_model,
            self.student_optimizer,
            self.student_lr_scheduler,
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

            for batch_idx, (inputs, labels, clip_lengths) in enumerate(
                self.train_loader
            ):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                if is_teacher:
                    # Default Loss
                    classification_loss = F.binary_cross_entropy_with_logits(
                        outputs, labels
                    )
                    transition_loss = self._compute_transition_loss(outputs)
                    duration_loss = self._compute_duration_loss(outputs, clip_lengths)
                    total_loss = (
                        classification_loss
                        + 0.1 * transition_loss
                        + 0.1 * duration_loss
                    )
                else:
                    # Distillation loss
                    soft_targets = soft_labels[
                        batch_idx
                        * self.train_loader.batch_size : (batch_idx + 1)
                        * self.train_loader.batch_size
                    ]
                    soft_targets = soft_targets.to(DEVICE)

                    distill_loss = F.binary_cross_entropy_with_logits(
                        outputs / self.temperature, soft_targets / self.temperature
                    ) * (self.temperature * self.temperature)

                    hard_loss = F.binary_cross_entropy_with_logits(outputs, labels)
                    total_loss = (
                        1 - self.alpha
                    ) * hard_loss + self.alpha * distill_loss

                optimizer.zero_grad()
                total_loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += total_loss.item()

            # Validation
            val_map = self._validate_model(model)
            lr_scheduler.step(val_map)
            print(
                f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss/len(self.train_loader):.4f} - mAP: {val_map:.4f}"
            )

            if val_map > best_map:
                best_map = val_map
                self._save_checkpoint(model, "best_model.pth", val_map)

    def _compute_transition_loss(self, outputs: torch.Tensor):
        """Compute loss based on transition probabilities"""

        # Convert outputs to probabilities
        current_probs = torch.sigmoid(outputs[0])
        if self.previous_prediction is None:
            self.previous_prediction = current_probs.detach()
            return torch.tensor(0.0, device=outputs.device)

        # Calculate expected probabilities for current prediction based on previous
        expected_current = torch.matmul(
            self.previous_prediction, self.transition_matrix
        )

        # Compute loss between expected and actual current probabilities
        transition_loss = F.mse_loss(current_probs, expected_current)

        # Update previous prediction for next time
        self.previous_prediction = current_probs.detach()

        return transition_loss

    def _compute_duration_loss(
        self, outputs: torch.Tensor, clip_lengths: List[int]
    ) -> torch.Tensor:
        """
        Compute duration loss
        """
        probs = torch.sigmoid(outputs)
        pred_labels = (probs > 0.7).cpu().numpy().astype(int)

        batch_size = outputs.shape[0]
        total_loss = torch.tensor(0.0, device=outputs.device)

        for i in range(batch_size):
            pred_tuple = tuple(pred_labels[i])
            actual_length = clip_lengths[i]

            if pred_tuple in self.action_durations:
                expected_length = self.action_durations[pred_tuple]
                # Compute relative error and clip to [0, 1]
                duration_diff = min(
                    abs(actual_length - expected_length) / expected_length, 1.0
                )
                total_loss = total_loss + duration_diff
            else:
                # total_loss = total_loss + 1.0  # Maximum penalty
                # or skip
                continue

        # Average over batch size to keep scale similar to BCE
        return total_loss / batch_size

    def _generate_soft_labels(self, teacher_model) -> torch.Tensor:
        """Generate soft labels using trained teacher model"""
        teacher_model.eval()
        all_soft_labels = []

        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(DEVICE)
                outputs = teacher_model(inputs)
                soft_labels = torch.sigmoid(outputs)
                all_soft_labels.append(soft_labels.cpu())

        return torch.cat(all_soft_labels, dim=0)

    def _validate_model(self, model) -> dict:
        """
        Validate model and return detailed performance metrics
        Returns:
            dict: Dictionary containing overall and per-class metrics
        """
        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, clip_lengths in self.val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        outputs = torch.cat(all_outputs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        predictions = torch.sigmoid(outputs).numpy()
        true_labels = labels.numpy()

        # Overall metrics
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

        # Calculate additional per-class metrics
        for i in range(len(class_aps)):
            class_preds = predictions[:, i]
            class_labels = true_labels[:, i]

            # Calculate optimal threshold using F1 score
            thresholds = np.arange(0, 1, 0.01)
            f1_scores = [
                f1_score(class_labels, class_preds > threshold)
                for threshold in thresholds
            ]
            optimal_threshold = thresholds[np.argmax(f1_scores)]

            # Calculate metrics at optimal threshold
            binary_preds = class_preds > optimal_threshold
            metrics["per_class"][f"class_{i}"] = {
                "AP": class_aps[i],
                "precision": precision_score(class_labels, binary_preds),
                "recall": recall_score(class_labels, binary_preds),
                "f1": f1_score(class_labels, binary_preds),
                "optimal_threshold": optimal_threshold,
            }

        # Print detailed report
        print("\nValidation Results:")
        print(f"Overall mAP: {metrics['overall']['mAP']:.4f}")
        print(f"Macro AP: {metrics['overall']['macro_AP']:.4f}")
        print(f"Weighted AP: {metrics['overall']['weighted_AP']:.4f}")

        print("\nPer-class Performance:")
        for class_id, class_metrics in metrics["per_class"].items():
            print(f"\n{class_id}:")
            print(f"  AP: {class_metrics['AP']:.4f}")
            print(f"  F1: {class_metrics['f1']:.4f}")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  Optimal threshold: {class_metrics['optimal_threshold']:.2f}")

        return metrics["overall"]["mAP"]

    def _save_checkpoint(self, model, filename: str, val_map: float):
        """Save model checkpoint"""
        torch.save(
            {"model_state_dict": model.state_dict(), "val_map": val_map}, filename
        )

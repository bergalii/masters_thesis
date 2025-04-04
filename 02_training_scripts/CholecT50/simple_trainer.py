import torch
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
import logging
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from recognition import Recognition


class TripletHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_layer_dim: int):
        super().__init__()
        # Extract hidden features for attention
        self.hidden = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, hidden_layer_dim),
            nn.GELU(),
        )
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_layer_dim, num_classes),
        )

    def forward(self, x):
        hidden_features = self.hidden(x)
        logits = self.classifier(hidden_features)
        return logits, hidden_features


class SimplifiedTrainer:
    def __init__(
        self,
        num_epochs: int,
        train_loader,
        val_loader,
        num_classes: int,
        label_mappings: dict,
        device,
        logger: logging.Logger,
        dir_name: str,
        learning_rate: float,
        weight_decay: float,
        hidden_layer_dim: int,
        gradient_clipping: float,
    ):
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.label_mappings = label_mappings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_layer_dim = hidden_layer_dim
        self.gradient_clipping = gradient_clipping
        self.device = device
        self.logger = logger
        self.dir_name = dir_name

        self._configure_model()

    def _configure_model(self):
        """Initialize the model with backbone and triplet head only"""
        # Initialize model with pretrained weights
        self.model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        # Remove the original classification head
        self.model.head = nn.Identity()
        in_features = self.model.num_features

        # Add triplet head only
        self.model.triplet_head = TripletHead(
            in_features, self.num_classes["triplet"], self.hidden_layer_dim
        ).to(self.device)

        # Create optimizer and scheduler
        self.optimizer, self.scheduler = self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):
        """Create optimizer and scheduler with parameter groups"""
        # Separate parameters into backbone and head groups
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "triplet_head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": self.learning_rate / 10,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": head_params,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
            ]
        )

        # Use OneCycleLR scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.learning_rate / 10, self.learning_rate],
            total_steps=len(self.train_loader) * self.num_epochs,
            pct_start=0.1,  # 10% of training time is warmup
            anneal_strategy="cos",
            div_factor=5.0,
            final_div_factor=10.0,
        )

        return optimizer, scheduler

    def _forward_pass(self, inputs):
        """Simple forward pass through backbone and triplet head"""
        backbone_features = self.model(inputs)
        triplet_logits, _ = self.model.triplet_head(backbone_features)
        return triplet_logits

    def train(self):
        """Execute model training"""
        self.logger.info("Training simplified model...")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable parameters: {total_trainable_params:,}")
        self.logger.info("-" * 50)

        # Track best performance
        best_map = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                triplet_labels = labels["triplet"].to(self.device)

                # Forward pass
                triplet_logits = self._forward_pass(inputs)

                # Compute loss
                loss = F.binary_cross_entropy_with_logits(
                    triplet_logits, triplet_labels
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.gradient_clipping
                )
                self.optimizer.step()
                self.scheduler.step()

                # Accumulate loss
                epoch_loss += loss.item()

            # Calculate average loss
            avg_loss = epoch_loss / len(self.train_loader)

            # Validation
            self.logger.info(f"Validation Results - Epoch {epoch+1}/{self.num_epochs}:")
            triplet_map = self._validate_model()

            # Log metrics
            current_lrs = [group["lr"] for group in self.optimizer.param_groups]
            self.logger.info(f"Learning rates: {[f'{lr:.6f}' for lr in current_lrs]}")
            self.logger.info(f"Training Loss: {avg_loss:.4f}")
            self.logger.info(f"Validation mAP: {triplet_map:.4f}")
            self.logger.info("-" * 50)

            # Save best model
            if triplet_map > best_map:
                best_map = triplet_map
                torch.save(self.model.state_dict(), f"{self.dir_name}/best_model.pth")
                self.logger.info(f"New best triplet mAP: {triplet_map:.4f}")
                self.logger.info(f"Model saved to {self.dir_name}/best_model.pth")
                self.logger.info("-" * 50)

    def _validate_model(self):
        """Validate model and compute metrics"""
        self.model.eval()

        # Initialize recognition module for metric calculation
        recognize = Recognition(num_class=self.num_classes["triplet"])
        recognize.reset()

        with torch.no_grad():
            for inputs_batch, batch_labels in self.val_loader:
                # Process each video in the batch\
                batch_size = inputs_batch.shape[0]
                num_clips = inputs_batch.shape[1]

                for b in range(batch_size):
                    video_outputs = []

                    # Process each clip in the video
                    for c in range(num_clips):
                        # Extract single clip and add batch dimension
                        clip = inputs_batch[b, c].unsqueeze(0).to(self.device)

                        # Get predictions for this clip
                        clip_logits = self._forward_pass(clip)
                        video_outputs.append(clip_logits)

                    # Average predictions across clips
                    outputs_tensor = torch.cat(video_outputs, dim=0)
                    avg_logits = torch.mean(outputs_tensor, dim=0, keepdim=True)

                    # Convert to probabilities
                    predictions = torch.sigmoid(avg_logits).cpu().numpy()
                    labels = batch_labels["triplet"][b].unsqueeze(0).cpu().numpy()

                    # Update the recognizer with current video
                    recognize.update(labels, predictions)

        # Compute and log metrics
        results = recognize.compute_AP(component="ivt")
        mean_ap = results["mAP"]
        class_aps = results["AP"]

        # Log results
        self.logger.info(f"Overall mAP: {mean_ap:.4f}")

        # Log per-class metrics
        for i in range(len(class_aps)):
            original_id = (
                self.val_loader.dataset.index_to_triplet[i]
                if hasattr(self.val_loader.dataset, "index_to_triplet")
                else i
            )
            label_name = self.label_mappings.get(original_id, f"Class_{original_id}")
            self.logger.info(f"  {label_name}: AP = {class_aps[i]:.4f}")

        return mean_ap

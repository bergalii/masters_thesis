import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
import logging
from torchvision.models import swin_v2_s, Swin_V2_S_Weights
from recognition import Recognition
import logging
from utils import (
    setup_logging,
    set_seeds,
    load_configs,
)
from pathlib import Path


class TripletHeadWithLSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_layer_dim: int,
        lstm_hidden_dim: int,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        # Feature extraction layer
        self.hidden = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, hidden_layer_dim),
            nn.GELU(),
        )

        # LSTM layer for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_layer_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim * 2),  # * 2 for bidirectional
            nn.Dropout(p=0.3),
            nn.Linear(lstm_hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, features]
        batch_size, num_frames, features = x.shape

        # Apply feature extraction to each frame
        x = x.view(-1, features)  # [batch_size * num_frames, features]
        hidden_features = self.hidden(x)  # [batch_size * num_frames, hidden_dim]

        # Reshape for LSTM
        hidden_features = hidden_features.view(
            batch_size, num_frames, -1
        )  # [batch_size, num_frames, hidden_dim]

        # Apply LSTM
        lstm_out, _ = self.lstm(
            hidden_features
        )  # [batch_size, num_frames, lstm_hidden_dim * 2]

        # Take the final frame's output
        final_hidden = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim * 2]

        # Classification
        logits = self.classifier(final_hidden)  # [batch_size, num_classes]

        return logits, final_hidden


class LSTMTrainer:
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
        learning_rate: float,
        weight_decay: float,
        hidden_layer_dim: int,
        lstm_hidden_dim: int,
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
        self.lstm_hidden_dim = lstm_hidden_dim
        self.gradient_clipping = gradient_clipping
        self.device = device
        self.logger = logger
        self.dir_name = dir_name

        self._configure_model()

    def _configure_model(self):
        """Initialize the model with image backbone and LSTM head"""
        # Initialize model with pretrained weights
        self.model = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT).to(self.device)
        # Remove the original classification head
        in_features = self.model.head.in_features
        self.model.head = nn.Identity()

        # Add triplet head with LSTM for temporal modeling
        self.model.triplet_head = TripletHeadWithLSTM(
            in_features,
            self.num_classes["triplet"],
            self.hidden_layer_dim,
            self.lstm_hidden_dim,
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

        # Use OneCycleLR scheduler similar to video model
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.learning_rate / 10, self.learning_rate],
            total_steps=len(self.train_loader) * int(self.num_epochs * 0.7),
            pct_start=0.1,  # 10% of training time is warmup
            anneal_strategy="cos",
            div_factor=5.0,
            final_div_factor=10.0,
        )

        return optimizer, scheduler

    def _forward_pass(self, inputs):
        """Forward pass with temporal modeling with LSTM"""
        # Inputs shape is always [batch_size, C, T, H, W]
        batch_size, channels, frames, height, width = inputs.shape

        # Process each frame through backbone
        all_features = []
        for f in range(frames):
            # Extract frame f
            frame_batch = inputs[:, :, f, :, :]  # [batch_size, C, H, W]

            # Pass through backbone
            frame_features = self.model(frame_batch)  # [batch_size, feature_dim]

            # Add to list
            all_features.append(frame_features)

        # Stack frame features to create temporal sequence
        temporal_features = torch.stack(
            all_features, dim=1
        )  # [batch_size, frames, feature_dim]

        # Process through LSTM head
        logits, _ = self.model.triplet_head(temporal_features)

        return logits

    def train(self):
        """Execute model training"""
        self.logger.info("Training LSTM model...")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable parameters: {total_trainable_params:,}")
        self.logger.info("-" * 50)

        # Track best performance
        best_map = 0.0

        for epoch in range(int(self.num_epochs * 0.7)):
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
            validation_metrics = self._validate_model()
            triplet_map = validation_metrics["triplet"]

            # Log metrics
            current_lrs = [group["lr"] for group in self.optimizer.param_groups]
            self.logger.info(f"Learning rates: {[f'{lr:.6f}' for lr in current_lrs]}")
            self.logger.info(f"Training Loss: {avg_loss:.4f}")
            self.logger.info(f"TRIPLET: mAP = {validation_metrics['triplet']:.4f}")
            self.logger.info(
                f"INSTRUMENT: mAP = {validation_metrics['instrument']:.4f}"
            )
            self.logger.info(f"VERB: mAP = {validation_metrics['verb']:.4f}")
            self.logger.info(f"TARGET: mAP = {validation_metrics['target']:.4f}")
            self.logger.info("-" * 50)

            # Save best model
            if triplet_map > best_map:
                best_map = triplet_map
                torch.save(self.model.state_dict(), f"{self.dir_name}/best_model.pth")
                self.logger.info(f"New best triplet mAP: {triplet_map:.4f}")
                self.logger.info(f"Model saved to {self.dir_name}/best_model.pth")
                self.logger.info("-" * 50)

    def _validate_model(self):
        """Validate model and compute metrics for all components"""
        self.model.eval()

        # Initialize recognition module for metric calculation
        recognize = Recognition(num_class=self.num_classes["triplet"])
        recognize.reset()

        with torch.no_grad():
            for inputs_batch, batch_labels in self.val_loader:
                # inputs_batch shape: [batch_size, num_clips, C, T, H, W]
                batch_size = inputs_batch.shape[0]
                num_clips = inputs_batch.shape[1]

                # Process each video in the batch
                for b in range(batch_size):
                    # Process each clip and collect predictions
                    all_clip_logits = []

                    for c in range(num_clips):
                        # Extract single clip and move to device
                        clip = (
                            inputs_batch[b, c].unsqueeze(0).to(self.device)
                        )  # [1, C, T, H, W]

                        # Get predictions for this clip
                        clip_logits = self._forward_pass(clip)
                        all_clip_logits.append(clip_logits)

                    # Stack and average predictions across all clips
                    stacked_logits = torch.cat(
                        all_clip_logits, dim=0
                    )  # [num_clips, num_classes]
                    avg_logits = torch.mean(
                        stacked_logits, dim=0, keepdim=True
                    )  # [1, num_classes]

                    # Convert to probabilities
                    avg_probs = torch.sigmoid(avg_logits)

                    # Update recognizer with averaged predictions
                    predictions = avg_probs.cpu().numpy()
                    labels = batch_labels["triplet"][b].unsqueeze(0).cpu().numpy()
                    recognize.update(labels, predictions)

        # Compute metrics for all components
        component_results = {
            "triplet": recognize.compute_AP(component="ivt")["mAP"],
            "instrument": recognize.compute_AP(component="i")["mAP"],
            "verb": recognize.compute_AP(component="v")["mAP"],
            "target": recognize.compute_AP(component="t")["mAP"],
        }

        return component_results


def main():
    CLIPS_DIR = r"05_datasets_dir/CholecT50/videos"
    ANNOTATIONS_PATH = r"05_datasets_dir/CholecT50/annotations.csv"
    CONFIGS_PATH = r"02_training_scripts/CholecT50/configs.yaml"

    dir_name, logger = setup_logging("training")
    model_dir = Path(f"04_models_dir/{dir_name}")
    model_dir.mkdir(exist_ok=True)

    torch.cuda.set_device(1)
    DEVICE = torch.device("cuda:1")
    logger.info(f"Cuda is active and using GPU: {torch.cuda.current_device()}")
    set_seeds()

    configs = load_configs(CONFIGS_PATH, logger)

    # Create datasets using the same MultiTaskVideoDataset class used by the video model
    train_dataset = MultiTaskVideoDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        clip_length=configs["clip_length"],
        split="train",
        train_ratio=configs["train_ratio"],
        train=True,
        frame_width=configs["frame_width"],
        frame_height=configs["frame_height"],
        min_occurrences=configs["min_occurrences"],
        cross_val_fold=configs["val_split"],
    )

    val_dataset = MultiTaskVideoDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        clip_length=configs["clip_length"],
        split="val",
        train_ratio=configs["train_ratio"],
        train=False,
        frame_width=configs["frame_width"],
        frame_height=configs["frame_height"],
        min_occurrences=configs["min_occurrences"],
        cross_val_fold=configs["val_split"],
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=configs["batch_size"],
        shuffle=False,
    )

    logger.info(f"Created datasets:")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    logger.info(f"  Number of classes:")
    for task, num_class in train_dataset.num_classes.items():
        logger.info(f"    {task}: {num_class}")

    # Create LSTM trainer
    trainer = LSTMTrainer(
        num_epochs=configs["num_epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=train_dataset.num_classes,
        label_mappings=train_dataset.label_mappings["triplet"],
        device=DEVICE,
        logger=logger,
        dir_name=model_dir,
        learning_rate=configs["learning_rate"],
        weight_decay=configs["weight_decay"],
        hidden_layer_dim=configs["hidden_layer_dim"],
        lstm_hidden_dim=256,
        gradient_clipping=configs["gradient_clipping"],
    )

    # Train the model
    logger.info("Starting LSTM training process...")
    trainer.train()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

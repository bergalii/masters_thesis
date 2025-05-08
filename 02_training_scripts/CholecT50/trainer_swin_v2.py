import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os
import ast
import logging
from pathlib import Path
from utils import (
    setup_logging,
    set_seeds,
    load_configs,
)
from torchvision.models import swin_v2_s, Swin_V2_S_Weights
from torchvision.transforms import Compose, ToTensor, Normalize
import albumentations as A
from decord import VideoReader, cpu
from recognition import Recognition
from disentangle import Disentangle


class MultiTaskImageDataset(Dataset):
    def __init__(
        self,
        clips_dir: str,
        annotations_path: str,
        split: str,
        train: bool,
        frame_width: int,
        frame_height: int,
        cross_val_fold: int,
        fps: int = 1,
    ):
        self.clips_dir = clips_dir
        self.train = train
        self.split = split
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.annotations = pd.read_csv(annotations_path)
        self.cross_val_fold = cross_val_fold

        # Extract video IDs from file names
        self.annotations["video_id"] = self.annotations["file_name"].apply(
            lambda x: int(x.split("_")[0].replace("video", ""))
        )

        # Initialize global mappings before splitting the dataset
        self._initialize_global_mappings()

        split_indices = self._create_cross_val_split(cross_val_fold)

        self.annotations = self.annotations.iloc[split_indices].reset_index(drop=True)

        # Initialize transforms
        self.preprocess = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.3656, 0.3660, 0.3670], std=[0.2025, 0.2021, 0.2027]),
            ]
        )

        self.transform = A.Compose(
            [
                # Color adjustments
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.25, 0.25),
                            contrast_limit=(-0.25, 0.25),
                            p=0.4,
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=25,
                            val_shift_limit=20,
                            p=0.3,
                        ),
                    ],
                    p=0.7,
                ),
                # Flips
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]),
                # Detail enhancement
                A.CLAHE(clip_limit=(1, 1.5), tile_grid_size=(6, 6), p=0.3),
                # Blurs to simulate focus variations
                A.OneOf(
                    [
                        A.AdvancedBlur(blur_limit=(3, 5), p=0.3),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                    ],
                    p=0.2,
                ),
                # Reduced noise
                A.GaussNoise(std_range=(0.05, 0.15), p=0.2),
            ]
        )

        # Prepare the image samples list
        self.image_samples = self._prepare_image_samples()

    def _initialize_global_mappings(self):
        """Initialize global mappings for all possible triplets"""
        # Create mappings dict
        self.label_mappings = {
            "instrument": {},
            "verb": {},
            "target": {},
            "triplet": {},
        }

        # Create mappings from original IDs to continuous indices
        self.triplet_to_index = {}  # Original ID -> continuous index
        self.index_to_triplet = {}  # Continuous index -> original ID

        # Extract all unique triplet IDs from annotations
        all_triplet_ids = set()
        for _, row in self.annotations.iterrows():
            triplet_labels = ast.literal_eval(row["triplet_label"])
            for triplet_id in triplet_labels:
                all_triplet_ids.add(triplet_id)

        # Create continuous mapping for all triplet IDs
        for continuous_idx, original_id in enumerate(sorted(all_triplet_ids)):
            self.triplet_to_index[original_id] = continuous_idx
            self.index_to_triplet[continuous_idx] = original_id

        # Initialize triplet to IVT mapping
        self.triplet_to_ivt = {}

        # Process annotations to extract mappings
        for _, row in self.annotations.iterrows():
            # Convert string lists to actual lists if needed
            for col in [
                "instrument_label",
                "verb_label",
                "target_label",
                "triplet_label",
            ]:
                if isinstance(row[col], str):
                    row[col] = ast.literal_eval(row[col])

            # If triplet names are available, extract mappings
            if "triplet_label_names" in row and row["triplet_label_names"]:
                triplet_names = ast.literal_eval(row["triplet_label_names"])

                for triplet_id, (triplet, inst_id, verb_id, target_id) in enumerate(
                    zip(
                        triplet_names,
                        row["instrument_label"],
                        row["verb_label"],
                        row["target_label"],
                    )
                ):
                    # Each triplet is in format 'instrument,verb,target'
                    inst_name, verb_name, target_name = triplet.split(",")

                    # Update mappings
                    self.label_mappings["instrument"][inst_id] = inst_name
                    self.label_mappings["verb"][verb_id] = verb_name
                    self.label_mappings["target"][target_id] = target_name

                    # Use the actual triplet ID from the row data
                    actual_triplet_id = row["triplet_label"][triplet_id]

                    # Store triplet name
                    self.label_mappings["triplet"][actual_triplet_id] = triplet

                    # Store the mapping to component IDs
                    if continuous_idx := self.triplet_to_index.get(actual_triplet_id):
                        self.triplet_to_ivt[continuous_idx] = [
                            inst_id,
                            verb_id,
                            target_id,
                        ]

        # Determine the number of classes for each category
        # Get unique IDs for each category
        inst_ids = set()
        verb_ids = set()
        target_ids = set()

        for _, row in self.annotations.iterrows():
            for inst_id in ast.literal_eval(str(row["instrument_label"])):
                inst_ids.add(inst_id)
            for verb_id in ast.literal_eval(str(row["verb_label"])):
                verb_ids.add(verb_id)
            for target_id in ast.literal_eval(str(row["target_label"])):
                target_ids.add(target_id)

        self.num_classes = {
            "instrument": max(inst_ids) + 1 if inst_ids else 0,
            "verb": max(verb_ids) + 1 if verb_ids else 0,
            "target": max(target_ids) + 1 if target_ids else 0,
            "triplet": len(self.triplet_to_index),
        }

    def _create_cross_val_split(self, fold: int) -> np.ndarray:
        """
        Create a split based on the cross-validation fold.

        Args:
            fold: Which cross-validation fold to use (1-5)

        Returns:
            np.ndarray: Indices for the current split (train or validation)
        """

        # Get the list of validation videos for this fold
        val_videos = Disentangle().cross_val_splits[fold]

        # Find all unique video IDs in the dataset
        all_video_ids = set(self.annotations["video_id"].unique())

        # For training, use all videos that are not in the validation set
        train_videos = [v for v in all_video_ids if v not in val_videos]

        # Get indices for the selected split
        if self.split == "train":
            indices = self.annotations[
                self.annotations["video_id"].isin(train_videos)
            ].index.values
        else:
            indices = self.annotations[
                self.annotations["video_id"].isin(val_videos)
            ].index.values

        return indices

    def _prepare_image_samples(self):
        """Extract frames from videos at the specified fps rate and prepare samples."""
        image_samples = []

        for idx, row in self.annotations.iterrows():
            video_path = os.path.join(self.clips_dir, row["file_name"])

            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue

            # Open the video with decord
            video = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(video)

            # Calculate frame indices based on fps
            frame_interval = max(1, int(25 / self.fps))
            frame_indices = list(range(0, total_frames, frame_interval))

            # Add each frame as a sample with the same labels
            for frame_idx in frame_indices:
                image_samples.append(
                    {
                        "video_idx": idx,
                        "frame_idx": frame_idx,
                        "labels": {
                            "instrument": ast.literal_eval(
                                str(row["instrument_label"])
                            ),
                            "verb": ast.literal_eval(str(row["verb_label"])),
                            "target": ast.literal_eval(str(row["target_label"])),
                            "triplet": ast.literal_eval(str(row["triplet_label"])),
                        },
                    }
                )

        return image_samples

    def _create_multi_hot(self, label_ids, category: str):
        """Create a multi-hot encoded tensor for a specific category."""
        multi_hot = torch.zeros(self.num_classes[category])

        if category == "triplet":
            # Map discontinuous triplet IDs to continuous indices
            continuous_indices = [
                self.triplet_to_index.get(lid, 0)
                for lid in label_ids
                if lid in self.triplet_to_index
            ]
            multi_hot[continuous_indices] = 1
        else:
            # For other categories, just set the specified indices to 1
            valid_ids = [
                lid for lid in label_ids if 0 <= lid < self.num_classes[category]
            ]
            if valid_ids:
                multi_hot[valid_ids] = 1

        return multi_hot

    def __len__(self):
        return len(self.image_samples)

    def __getitem__(self, idx):
        sample = self.image_samples[idx]
        video_idx = sample["video_idx"]
        frame_idx = sample["frame_idx"]

        # Get the video path
        video_path = os.path.join(
            self.clips_dir, self.annotations.iloc[video_idx]["file_name"]
        )

        # Extract the frame
        video = VideoReader(video_path, ctx=cpu(0))
        # Ensure frame index is within bounds
        frame_idx = min(frame_idx, len(video) - 1)
        frame = video[frame_idx].asnumpy()  # Convert to numpy array

        # Process the frame - handle black borders
        # Detect borders
        threshold = 20  # Threshold for black border detection
        margin = 5  # Margin around detected content

        col_sums = np.sum(frame, axis=(0, 2))
        col_mask = col_sums > threshold * frame.shape[0]  # Scale threshold by height

        # Find where content begins (left border) and ends (right border)
        non_zero_indices = np.where(col_mask)[0]

        if len(non_zero_indices) > 0:
            left_border = non_zero_indices[0]
            right_border = non_zero_indices[-1]
        else:
            # Fallback if detection fails
            left_border = 0
            right_border = frame.shape[1] - 1

        # Add a margin to ensure not to crop too aggressively
        left_border = max(0, left_border - margin)
        right_border = min(frame.shape[1] - 1, right_border + margin)

        # Crop the frame to the detected content area
        cropped_frame = frame[:, left_border : right_border + 1, :]

        # Resize to the target dimensions
        resized_frame = cv2.resize(
            cropped_frame,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Apply augmentations for training
        if self.train:
            augmented = self.transform(image=resized_frame)
            resized_frame = augmented["image"]

        # Convert to tensor and normalize
        image_tensor = self.preprocess(resized_frame)

        # Create multi-hot labels
        labels = {
            "instrument": self._create_multi_hot(
                sample["labels"]["instrument"], "instrument"
            ),
            "verb": self._create_multi_hot(sample["labels"]["verb"], "verb"),
            "target": self._create_multi_hot(sample["labels"]["target"], "target"),
            "triplet": self._create_multi_hot(sample["labels"]["triplet"], "triplet"),
        }

        return image_tensor, labels


class TripletHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_layer_dim: int):
        super().__init__()
        # Extract hidden features for attention
        self.hidden = nn.Sequential(
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


class Trainer:
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
        self.model = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT).to(self.device)
        in_features = self.model.head.in_features
        # Remove the original classification head
        self.model.head = nn.Identity()

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
            pct_start=0.4,  # 10% of training time is warmup
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=20.0,
        )

        return optimizer, scheduler

    def _forward_pass(self, inputs):
        """Simple forward pass through backbone and triplet head"""
        backbone_features = self.model(inputs)
        triplet_logits, _ = self.model.triplet_head(backbone_features)
        return triplet_logits

    def train(self):
        """Execute model training"""
        self.logger.info("Training simplified image model...")
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
            for inputs, batch_labels in self.val_loader:
                inputs = inputs.to(self.device)

                # Forward pass
                triplet_logits = self._forward_pass(inputs)

                # Convert to probabilities
                triplet_probs = torch.sigmoid(triplet_logits)

                # Get batch predictions and labels
                predictions = triplet_probs.cpu().numpy()
                labels = batch_labels["triplet"].cpu().numpy()

                # Update the recognizer with the current batch
                recognize.update(labels, predictions)

        # Compute and log metrics for all components
        component_results = {
            "triplet": recognize.compute_AP(component="ivt"),
            "instrument": recognize.compute_AP(component="i"),
            "verb": recognize.compute_AP(component="v"),
            "target": recognize.compute_AP(component="t"),
        }

        # Log overall results for each component
        self.logger.info("VALIDATION METRICS:")
        self.logger.info(f"TRIPLET: mAP = {component_results['triplet']['mAP']:.4f}")
        self.logger.info(
            f"INSTRUMENT: mAP = {component_results['instrument']['mAP']:.4f}"
        )
        self.logger.info(f"VERB: mAP = {component_results['verb']['mAP']:.4f}")
        self.logger.info(f"TARGET: mAP = {component_results['target']['mAP']:.4f}")
        self.logger.info("-" * 30)

        # Log per-class metrics for triplet
        triplet_aps = component_results["triplet"]["AP"]
        self.logger.info("PER-CLASS TRIPLET METRICS:")
        for i in range(len(triplet_aps)):
            original_id = (
                self.val_loader.dataset.index_to_triplet[i]
                if hasattr(self.val_loader.dataset, "index_to_triplet")
                else i
            )
            label_name = self.label_mappings.get(original_id, f"Class_{original_id}")
            self.logger.info(f"  {label_name}: AP = {triplet_aps[i]:.4f}")

        # Return the triplet mAP for model selection
        return component_results["triplet"]["mAP"]


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

    # Create datasets
    train_dataset = MultiTaskImageDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        split="train",
        train=True,
        frame_width=configs["frame_width"],
        frame_height=configs["frame_height"],
        cross_val_fold=configs["val_split"],
        fps=1,  # Sample at 1 frame per second
    )

    val_dataset = MultiTaskImageDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        split="val",
        train=False,
        frame_width=configs["frame_width"],
        frame_height=configs["frame_height"],
        cross_val_fold=configs["val_split"],
        fps=1,  # Sample at 1 frame per second
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], shuffle=False
    )

    logger.info(f"Created datasets:")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    logger.info(f"  Number of classes:")
    for task, num_class in train_dataset.num_classes.items():
        logger.info(f"    {task}: {num_class}")

    # Create trainer
    trainer = Trainer(
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
        gradient_clipping=configs["gradient_clipping"],
    )

    # Train the model
    logger.info("Starting training process...")
    trainer.train()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

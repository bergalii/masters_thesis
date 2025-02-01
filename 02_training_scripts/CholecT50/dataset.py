from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from decord import VideoReader, cpu
import ast
from typing import List, Dict, Tuple
import albumentations as A
from torchvision.transforms import Compose, ToTensor, Normalize
from collections import defaultdict
import random


class MultiTaskVideoDataset(Dataset):
    def __init__(
        self,
        clips_dir: str,
        annotations_path: str,
        clip_length: int,
        split: str,
        train_ratio: float,
        train: bool,
        frame_width: int,
        frame_height: int,
        min_occurrences: int = 100,
    ):
        self.clips_dir = clips_dir
        self.clip_length = clip_length
        self.train = train
        self.split = split
        self.min_occurrences = min_occurrences
        self.frame_width = frame_width
        self.frame_height = frame_height
        # Read the annotations
        self.annotations = pd.read_csv(annotations_path)

        # Debugging
        # self.annotations = pd.read_csv(annotations_path).head(100)

        split_indices = self._create_stratified_split(train_ratio)
        self.annotations = self.annotations.iloc[split_indices].reset_index(drop=True)

        # Balance training set
        if split == "train":
            self._balance_dataset()

        # Initialize label name mappings for each category
        self.label_mappings = {
            "instrument": {},
            "verb": {},
            "target": {},
            "triplet": {},
        }

        # Create continuous index mappings for triplets
        self.triplet_to_index = {}
        self.index_to_triplet = {}

        # First, collect all unique triplet IDs
        all_triplet_ids = set()
        for _, row in self.annotations.iterrows():
            triplet_labels = ast.literal_eval(row["triplet_label"])
            all_triplet_ids.update(triplet_labels)

        # Create continuous mapping for triplet IDs
        for new_idx, original_id in enumerate(sorted(all_triplet_ids)):
            self.triplet_to_index[original_id] = new_idx
            self.index_to_triplet[new_idx] = original_id

        # Process string representations of lists and build label mappings
        for _, row in self.annotations.iterrows():
            # Convert string lists to actual lists
            for col in [
                "instrument_label",
                "verb_label",
                "target_label",
                "triplet_label",
            ]:
                row[col] = ast.literal_eval(row[col])

            # Process triplet_label_names to extract mappings
            action_names = ast.literal_eval(row["triplet_label_names"])
            for triplet_id, (triplet, inst_id, verb_id, target_id) in enumerate(
                zip(
                    action_names,
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
                triplet_id = row["triplet_label"][triplet_id]
                self.label_mappings["triplet"][triplet_id] = triplet

        # Calculate num_classes c
        self.num_classes = {
            "instrument": max(self.label_mappings["instrument"].keys()) + 1,
            "verb": max(self.label_mappings["verb"].keys()) + 1,
            "target": max(self.label_mappings["target"].keys()) + 1,
            "triplet": len(self.triplet_to_index),
        }

        # Initialize transforms (same as before)
        self.preprocess = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.3656, 0.3660, 0.3670], std=[0.2025, 0.2021, 0.2027]),
            ]
        )

        self.transform = A.ReplayCompose(
            [
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(90, 110), p=0.5),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.05, 0.05),
                            contrast_limit=(-0.05, 0.05),
                            p=0.5,
                        ),
                    ],
                    p=0.3,
                ),
                A.CLAHE(clip_limit=(1, 1.1), tile_grid_size=(6, 6), p=0.3),
                A.AdvancedBlur(blur_limit=(3, 7), p=0.3),
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.6,
                ),
            ]
        )

    def _create_stratified_split(self, train_ratio: float) -> np.ndarray:
        """
        Create a stratified split ensuring each triplet combination appears in both train and val sets.

        Args:
            train_ratio: Fraction of data to use for training

        Returns:
            np.ndarray: Indices for the current split (train or validation)
        """
        # Create a mapping of triplet combinations to video indices
        triplet_to_indices = defaultdict(list)

        for idx, row in self.annotations.iterrows():
            # Convert string representation to actual list
            triplet_combo = tuple(sorted(ast.literal_eval(row["triplet_label"])))
            triplet_to_indices[triplet_combo].append(idx)

        train_indices = []
        val_indices = []

        # For each triplet combination
        for triplet_combo, indices in triplet_to_indices.items():
            combo_indices = np.array(indices)
            np.random.shuffle(combo_indices)

            # Ensure at least one sample in each split
            if len(combo_indices) == 1:
                # If only one sample, add to training set
                train_indices.extend(combo_indices)
            else:
                # Calculate training size, ensuring at least one sample remains for validation
                combo_train_size = min(
                    max(1, int(len(combo_indices) * train_ratio)),
                    len(combo_indices) - 1,  # Ensure at least one sample for validation
                )

                train_indices.extend(combo_indices[:combo_train_size])
                val_indices.extend(combo_indices[combo_train_size:])

        # Convert to numpy arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        # Return appropriate indices based on split
        if self.split == "train":
            return train_indices
        else:
            return val_indices

    def _balance_dataset(self):
        """Oversample clips containing underrepresented triplets"""
        triplet_counts = defaultdict(int)
        # Count current triplet occurrences
        for _, row in self.annotations.iterrows():
            for triplet in ast.literal_eval(row["triplet_label"]):
                triplet_counts[triplet] += 1

        # Identify triplets needing more samples
        needed_triplets = {
            triplet: max(self.min_occurrences - count, 0)
            for triplet, count in triplet_counts.items()
        }

        # Collect indices of clips containing each triplet
        triplet_clips = defaultdict(list)
        for idx, row in self.annotations.iterrows():
            for triplet in ast.literal_eval(row["triplet_label"]):
                if needed_triplets.get(triplet, 0) > 0:
                    triplet_clips[triplet].append(idx)

        # Oversample clips
        new_samples = []
        for triplet, needed in needed_triplets.items():
            if needed == 0 or triplet not in triplet_clips:
                continue

            clips = triplet_clips[triplet]

            # Add extra samples
            new_samples.extend(random.choices(clips, k=needed))

        # Add new samples to annotations
        if new_samples:
            new_df = self.annotations.iloc[new_samples]
            self.annotations = pd.concat([self.annotations, new_df], ignore_index=True)
            self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)

    def _create_multi_hot(self, label_ids: List[int], category: str) -> torch.Tensor:
        """
        Create a multi-hot encoded tensor for a specific category.

        Args:
            label_ids: List of label IDs that are active
            category: Category name ('instrument', 'verb', 'target')

        Returns:
            Multi-hot encoded tensor
        """
        multi_hot = torch.zeros(self.num_classes[category])
        if label_ids:
            if category == "triplet":
                # Map discontinuous triplet IDs to continuous indices
                continuous_indices = [self.triplet_to_index[lid] for lid in label_ids]
                multi_hot[continuous_indices] = 1
            else:
                multi_hot[label_ids] = 1
        return multi_hot

    def get_label_names(self) -> Dict[str, Dict[int, str]]:
        """
        Get the mapping of label IDs to their names for each category.
        """
        return self.label_mappings

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Get annotation row
        row = self.annotations.iloc[idx]

        # Load video (same as before)
        video_path = f"{self.clips_dir}/{row['file_name']}"
        video = VideoReader(
            video_path, width=self.frame_width, height=self.frame_height, ctx=cpu(0)
        )

        total_frames = len(video)

        if self.train:
            # Random temporal sampling
            start_idx = random.randint(0, total_frames - self.clip_length)
            indices = np.linspace(
                start_idx, start_idx + self.clip_length - 1, self.clip_length, dtype=int
            )
        else:
            # Evenly spaced sampling for validation clips
            indices = np.linspace(0, total_frames - 1, self.clip_length, dtype=int)

        frames = video.get_batch(indices).asnumpy()
        # Apply augmentations and preprocessing (same as before)
        if self.train:
            data = self.transform(image=frames[0])
            augmented_frames = []
            for frame in frames:
                augmented = A.ReplayCompose.replay(data["replay"], image=frame)
                augmented_frames.append(augmented["image"])
            frames = np.stack(augmented_frames)

        # Preprocess frames
        frames = torch.stack([self.preprocess(frame) for frame in frames])
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Create multi-hot encoded labels for each category
        labels = {
            "instrument": self._create_multi_hot(
                ast.literal_eval(str(row["instrument_label"])), "instrument"
            ),
            "verb": self._create_multi_hot(
                ast.literal_eval(str(row["verb_label"])), "verb"
            ),
            "target": self._create_multi_hot(
                ast.literal_eval(str(row["target_label"])), "target"
            ),
            "triplet": self._create_multi_hot(
                ast.literal_eval(str(row["triplet_label"])), "triplet"
            ),
        }

        return frames, labels

    def get_continuous_triplet_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Get the mapping between original triplet IDs and continuous indices.

        Returns:
            Tuple containing:
            - Dict mapping original triplet IDs to continuous indices
            - Dict mapping continuous indices back to original triplet IDs
        """
        return self.triplet_to_index, self.index_to_triplet

    @staticmethod
    def calculate_video_mean_std(
        loader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mean and standard deviation across all frames in the video dataset.

        Args:
            dataset: MultiTaskVideoDataset instance
            batch_size: Number of videos to process in each batch
            num_workers: Number of worker processes for data loading

        Returns:
            tuple: (mean, std) tensors of shape (3,) for RGB channels
        """

        # Initialize accumulators for each channel
        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)
        num_frames = 0

        # Process each batch
        for frames, _ in loader:
            b, c, t, h, w = frames.shape
            frames = frames.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

            # Calculate mean and squared mean for the batch
            batch_samples = frames.size(0)
            channels_sum += frames.mean(dim=(0, 2, 3)) * batch_samples
            channels_squared_sum += (frames**2).mean(dim=(0, 2, 3)) * batch_samples
            num_frames += batch_samples

        # Calculate final statistics
        mean = channels_sum / num_frames
        std = torch.sqrt(channels_squared_sum / num_frames - mean**2)

        print(f"Dataset mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"Dataset std: [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import cv2
from decord import VideoReader, cpu
import ast
from typing import List, Dict, Tuple
import albumentations as A
from torchvision.transforms import Compose, ToTensor, Normalize
from collections import defaultdict
import random
from disentangle import Disentangle


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
        min_occurrences: int,
        cross_val_fold: int = None,
    ):
        self.clips_dir = clips_dir
        self.clip_length = clip_length
        self.train = train
        self.split = split
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.annotations = pd.read_csv(annotations_path)
        self.cross_val_fold = cross_val_fold

        # Extract video IDs from file names
        self.annotations["video_id"] = self.annotations["file_name"].apply(
            lambda x: int(x.split("_")[0].replace("video", ""))
        )

        # First initialize global mappings before splitting the dataset
        self._initialize_global_mappings()

        # If using cross-validation, use the specified fold
        if cross_val_fold is not None:
            split_indices = self._create_cross_val_split(cross_val_fold)
        else:
            # Otherwise use the stratified splits
            split_indices = self._create_stratified_split(train_ratio)

        self.annotations = self.annotations.iloc[split_indices].reset_index(drop=True)

        # Balance the training set based on minimum occurences for each triplet
        if split == "train":
            self._balance_dataset(min_occurrences)

        # Initialize transforms
        self.preprocess = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.3656, 0.3660, 0.3670], std=[0.2025, 0.2021, 0.2027]),
            ]
        )

        self.transform = A.ReplayCompose(
            [
                # Color adjustments
                A.OneOf(
                    [
                        A.RandomGamma(
                            gamma_limit=(70, 130), p=0.4
                        ),  # Wider gamma range
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

    def _initialize_global_mappings(self):
        """Initialize global mappings for all possible triplets using the Disentangle class"""
        all_triplet_data = Disentangle().bank

        # Initialize mapping dictionaries
        self.label_mappings = {
            "instrument": {},
            "verb": {},
            "target": {},
            "triplet": {},
        }

        # Create mappings from original (non-continuous) IDs to continuous indices
        self.triplet_to_index = {}  # Original ID -> continuous index
        self.index_to_triplet = {}  # Continuous index -> original ID

        # Extract all unique triplet IDs
        all_original_triplet_ids = sorted([int(row[0]) for row in all_triplet_data])

        # Create continuous mapping for ALL possible triplet IDs
        for continuous_idx, original_id in enumerate(all_original_triplet_ids):
            self.triplet_to_index[original_id] = continuous_idx
            self.index_to_triplet[continuous_idx] = original_id

        # Initialize triplet to IVT mapping
        self.triplet_to_ivt = {}

        # Create mapping from continuous indices to IVT
        for row in all_triplet_data:
            original_id = int(row[0])
            if original_id in self.triplet_to_index:
                continuous_idx = self.triplet_to_index[original_id]
                # Store [instrument, verb, target] mapping
                self.triplet_to_ivt[continuous_idx] = [
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                ]
                # Also store the triplet name for reference
                self.label_mappings["triplet"][
                    original_id
                ] = f"{int(row[1])},{int(row[2])},{int(row[3])}"

        # Process annotations to extract instrument, verb, target mappings
        for _, row in self.annotations.iterrows():
            # Convert string lists to actual lists
            for col in [
                "instrument_label",
                "verb_label",
                "target_label",
                "triplet_label",
            ]:
                if isinstance(row[col], str):
                    row[col] = ast.literal_eval(row[col])

            # Move this inside the loop if you want to process all rows
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
                    # Store the triplet name
                    self.label_mappings["triplet"][actual_triplet_id] = triplet

        # Determine the number of classes for each category
        # For triplets, use the continuous indices
        # For other categories, get the max ID
        inst_ids = set()
        verb_ids = set()
        target_ids = set()

        for row in all_triplet_data:
            inst_ids.add(int(row[1]))
            verb_ids.add(int(row[2]))
            target_ids.add(int(row[3]))

        self.num_classes = {
            "instrument": max(inst_ids) + 1,
            "verb": max(verb_ids) + 1,
            "target": max(target_ids) + 1,
            "triplet": len(self.triplet_to_index),
        }

        # Create reverse mapping from continuous index to original triplet ID
        self.triplet_continuous_to_original = {
            v: k for k, v in self.triplet_to_index.items()
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

    def _balance_dataset(self, min_occurrences):
        """Oversample clips containing underrepresented triplets"""

        triplet_counts = defaultdict(int)
        # Count current triplet occurrences
        for _, row in self.annotations.iterrows():
            for triplet in ast.literal_eval(row["triplet_label"]):
                triplet_counts[triplet] += 1

        # Identify triplets needing more samples
        needed_triplets = {
            triplet: max(min_occurrences - count, 0)
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

        if category == "triplet":
            # Map discontinuous triplet IDs to continuous indices
            continuous_indices = [self.triplet_to_index[lid] for lid in label_ids]
            multi_hot[continuous_indices] = 1
        else:
            multi_hot[label_ids] = 1

        return multi_hot

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Get annotation row
        row = self.annotations.iloc[idx]

        # Load video
        video_path = f"{self.clips_dir}/{row['file_name']}"
        original_video = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(original_video)

        if self.train:
            # Calculate the ranges for start and end indices
            start_range = int(total_frames * 0.2)  # First 20% of frames
            end_range = int(total_frames * 0.8)  # Last 20% of frames
            # Ensure we have enough frames between start and end for the clip
            if end_range - start_range < self.clip_length:
                # Random temporal sampling
                start_idx = random.randint(0, total_frames - self.clip_length)
                indices = np.linspace(
                    start_idx,
                    start_idx + self.clip_length - 1,
                    self.clip_length,
                    dtype=int,
                )
            else:
                # Sample start frame from first 20%
                start_idx = random.randint(0, start_range)
                # Sample end frame from last 20%
                end_idx = random.randint(end_range, total_frames - 1)
                # Create evenly spaced indices between start and end
                indices = np.linspace(start_idx, end_idx, self.clip_length, dtype=int)

            # Get the sampled frames
            original_frames = original_video.get_batch(indices).asnumpy()

            # Process each frame individually to handle camera movement
            processed_frames = []
            threshold = 20  # Threshold for black border detection
            margin = 5  # Margin around detected content

            for frame in original_frames:
                # Detect borders for this specific frame
                col_sums = np.sum(frame, axis=(0, 2))
                col_mask = (
                    col_sums > threshold * frame.shape[0]
                )  # Scale threshold by height

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

                processed_frames.append(resized_frame)

            # Stack frames back into a video
            frames = np.stack(processed_frames)

            # Apply augmentations
            data = self.transform(image=frames[0])
            augmented_frames = []
            for frame in frames:
                augmented = A.ReplayCompose.replay(data["replay"], image=frame)
                augmented_frames.append(augmented["image"])
            frames = np.stack(augmented_frames)

            # Preprocess frames
            frames = torch.stack([self.preprocess(frame) for frame in frames])
            frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

            # Return a single clip for training
            frames_tensor = frames
        else:
            # For validation, sample 4 clips from different parts of the video
            all_clips = []
            # Define sampling strategies for 4 clips
            sampling_strategies = [
                # Clip 1: Beginning section (00-40%)
                (0.0, 0.4),
                # Clip 2: Middle section (30-70%)
                (0.3, 0.7),
                # Clip 3: End section (60-100%)
                (0.6, 1.0),
                # Clip 4: Full video overview (evenly spaced)
                (0.0, 1.0),
            ]

            for i, (start_percent, end_percent) in enumerate(sampling_strategies):
                if i == 3:  # Last clip evenly spaced
                    indices = np.linspace(
                        0, total_frames - 1, self.clip_length, dtype=int
                    )
                else:
                    # Calculate frame indices based on percentage
                    start_idx = int(total_frames * start_percent)
                    end_idx = int(total_frames * end_percent)
                    # Clamp indices to valid range
                    end_idx = min(end_idx, total_frames - 1)
                    start_idx = max(start_idx, 0)

                    # Ensure we have enough frames
                    if end_idx - start_idx < self.clip_length:
                        # If not, fall back to centered sampling in this segment
                        mid_point = (start_idx + end_idx) // 2
                        half_length = self.clip_length // 2
                        start_idx = max(0, mid_point - half_length)
                        end_idx = min(
                            total_frames - 1, start_idx + self.clip_length - 1
                        )
                        indices = np.linspace(
                            start_idx, end_idx, self.clip_length, dtype=int
                        )
                    else:
                        indices = np.linspace(
                            start_idx, end_idx, self.clip_length, dtype=int
                        )

                # Get the sampled frames for this clip
                original_frames = original_video.get_batch(indices).asnumpy()

                # Process each frame (black border detection)
                processed_frames = []
                threshold = 20  # Threshold for black border detection
                margin = 5  # Margin around detected content

                for frame in original_frames:
                    # Detect borders for this specific frame
                    col_sums = np.sum(frame, axis=(0, 2))
                    col_mask = (
                        col_sums > threshold * frame.shape[0]
                    )  # Scale threshold by height

                    # Find where content begins (left border) and ends (right border)
                    non_zero_indices = np.where(col_mask)[0]

                    if len(non_zero_indices) > 0:
                        left_border = non_zero_indices[0]
                        right_border = non_zero_indices[-1]
                    else:
                        # Fallback if detection fails
                        left_border = 0
                        right_border = frame.shape[1] - 1

                    # Add a margin to ensure we don't crop too aggressively
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

                    processed_frames.append(resized_frame)

                # Stack frames back into a video
                frames = np.stack(processed_frames)

                # No augmentation for validation, just preprocess
                frames = torch.stack([self.preprocess(frame) for frame in frames])
                frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

                all_clips.append(frames)

            # Stack all clips into a tensor of shape [num_clips, C, T, H, W]
            frames_tensor = torch.stack(all_clips)

        # Create multi-hot encoded labels for each task
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
            "video_id": row["video_id"],
        }

        return frames_tensor, labels

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

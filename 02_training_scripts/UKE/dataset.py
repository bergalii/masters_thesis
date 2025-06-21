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
import os


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
    ):
        self.clips_dir = clips_dir
        self.clip_length = clip_length
        self.train = train
        self.split = split
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.annotations = pd.read_csv(annotations_path)

        # Extract video IDs from file names
        self.annotations["video_id"] = self.annotations["file_name"].apply(
            lambda x: int(x.split("_")[0].replace("video", ""))
        )

        # First initialize global mappings before splitting the dataset
        self._initialize_global_mappings()

        # split_indices = self._create_stratified_split(train_ratio)
        split_indices = self._create_video_based_split(train_ratio)
        self.annotations = self.annotations.iloc[split_indices].reset_index(drop=True)

        # Balance the training set based on minimum occurences for each triplet
        if split == "train":
            self._balance_dataset(min_occurrences)

        # Initialize transforms
        self.preprocess = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.3729, 0.2067, 0.2102], std=[0.3171, 0.2348, 0.2401]),
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
        """Initialize global mappings for all possible triplets from the CSV data"""

        # Initialize mapping dictionaries
        self.label_mappings = {
            "instrument": {},
            "verb": {},
            "target": {},
            "triplet": {},
        }

        # Initialize triplet to IVT mapping
        self.triplet_to_ivt = {}

        # Process annotations to extract mappings from CSV data
        for _, row in self.annotations.iterrows():
            # Convert string lists to actual lists
            action_labels = ast.literal_eval(row["action_label"])
            action_names = ast.literal_eval(row["action_label_names"])
            instrument_labels = ast.literal_eval(row["instrument_label"])
            instrument_names = ast.literal_eval(row["instrument_label_names"])
            verb_labels = ast.literal_eval(row["verb_label"])
            verb_names = ast.literal_eval(row["verb_label_names"])
            target_labels = ast.literal_eval(row["target_label"])
            target_names = ast.literal_eval(row["target_label_names"])

            # Parse each action name to extract instrument, verb, target
            for action_id, action_name in zip(action_labels, action_names):
                # Store the action name
                self.label_mappings["triplet"][action_id] = action_name

                # Parse the action name to get instrument, verb, target
                parts = action_name.split(",")
                instrument_name = parts[0]
                verb_name = parts[1]
                target_name = parts[2]

                # Find the corresponding IDs
                instrument_id = instrument_names.index(instrument_name)
                verb_id = verb_names.index(verb_name)
                target_id = target_names.index(target_name)

                # Store [instrument, verb, target] mapping
                self.triplet_to_ivt[action_id] = [
                    instrument_labels[instrument_id],
                    verb_labels[verb_id],
                    target_labels[target_id],
                ]

            # Create mappings for individual components
            for inst_id, inst_name in zip(instrument_labels, instrument_names):
                self.label_mappings["instrument"][inst_id] = inst_name

            for verb_id, verb_name in zip(verb_labels, verb_names):
                self.label_mappings["verb"][verb_id] = verb_name

            for target_id, target_name in zip(target_labels, target_names):
                self.label_mappings["target"][target_id] = target_name

        # Determine the number of classes for each category
        all_instrument_ids = []
        all_verb_ids = []
        all_target_ids = []
        all_action_ids = []

        for _, row in self.annotations.iterrows():
            all_instrument_ids.extend(ast.literal_eval(row["instrument_label"]))
            all_verb_ids.extend(ast.literal_eval(row["verb_label"]))
            all_target_ids.extend(ast.literal_eval(row["target_label"]))
            all_action_ids.extend(ast.literal_eval(row["action_label"]))

        self.num_classes = {
            "instrument": max(all_instrument_ids) + 1,
            "verb": max(all_verb_ids) + 1,
            "target": max(all_target_ids) + 1,
            "triplet": max(all_action_ids) + 1,
        }

    def _create_stratified_split(self, train_ratio: float) -> np.ndarray:
        """
        Create a stratified split ensuring proper 80-20 distribution for each triplet combination.
        """
        # Create a mapping of triplet combinations to video indices
        triplet_to_indices = defaultdict(list)

        for idx, row in self.annotations.iterrows():
            triplet_combo = tuple(sorted(ast.literal_eval(row["action_label"])))
            triplet_to_indices[triplet_combo].append(idx)

        train_indices = []
        val_indices = []

        # For each triplet combination
        for triplet_combo, indices in triplet_to_indices.items():
            combo_indices = np.array(indices)
            np.random.shuffle(combo_indices)

            n_samples = len(combo_indices)

            if n_samples == 1:
                # Single sample: put ONLY in validation
                # val_indices.extend(combo_indices)
                continue
            else:
                # Multiple samples: use exact 80-20 split
                # Always ensure at least 1 sample in validation
                n_train = int(np.ceil(n_samples * train_ratio))
                # Ensure we leave at least 1 for validation
                n_train = min(n_train, n_samples - 1)

                train_indices.extend(combo_indices[:n_train])
                val_indices.extend(combo_indices[n_train:])

        # Convert to numpy arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        train_set = set(train_indices)
        val_set = set(val_indices)
        overlap = train_set.intersection(val_set)
        if overlap:
            print(f"Data leakage detected! {len(overlap)} samples in both sets")

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        if self.split == "train":
            return train_indices
        else:
            return val_indices

    def _create_video_based_split(self, train_ratio: float) -> np.ndarray:
        """
        Create a video-based split where video02 is used for training and video01 for validation.
        The train_ratio parameter is ignored in this implementation.
        """
        train_indices = []
        val_indices = []

        for idx, row in self.annotations.iterrows():
            video_id = row["video_id"]

            # Check if this sample contains the rare "Clip" target
            action_labels = ast.literal_eval(row["action_label"])
            has_clip_target = 64 in action_labels  # Triplet 64 is the Clip one

            if has_clip_target:
                # Force all Clip samples into training set
                train_indices.append(idx)
            elif video_id == 2:  # Normal video02 for training
                train_indices.append(idx)
            elif video_id == 1:  # Normal video01 for validation
                val_indices.append(idx)

        # Convert to numpy arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        if self.split == "train":
            return train_indices
        else:
            return val_indices

    def _balance_dataset(self, min_occurrences):
        """Oversample clips containing underrepresented triplets"""

        triplet_counts = defaultdict(int)
        # Count current triplet occurrences
        for _, row in self.annotations.iterrows():
            for action in ast.literal_eval(row["action_label"]):
                triplet_counts[action] += 1

        # Identify triplets needing more samples
        needed_triplets = {
            action: max(min_occurrences - count, 0)
            for action, count in triplet_counts.items()
        }

        # Collect indices of clips containing each triplet
        triplet_clips = defaultdict(list)
        for idx, row in self.annotations.iterrows():
            for action in ast.literal_eval(row["action_label"]):
                if needed_triplets.get(action, 0) > 0:
                    triplet_clips[action].append(idx)

        # Oversample clips
        new_samples = []
        for action, needed in needed_triplets.items():
            if needed == 0 or action not in triplet_clips:
                continue

            clips = triplet_clips[action]

            # Add extra samples
            new_samples.extend(random.choices(clips, k=needed))

        # Add new samples to annotations
        if new_samples:
            new_df = self.annotations.iloc[new_samples]
            self.annotations = pd.concat([self.annotations, new_df], ignore_index=True)
            self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)

    # def _create_multi_hot(self, label_ids: List[int], category: str) -> torch.Tensor:
    #     """
    #     Create a multi-hot encoded tensor for a specific category.

    #     Args:
    #         label_ids: List of label IDs that are active
    #         category: Category name ('instrument', 'verb', 'target', 'triplet')

    #     Returns:
    #         Multi-hot encoded tensor
    #     """
    #     multi_hot = torch.zeros(self.num_classes[category])
    #     multi_hot[label_ids] = 1
    #     return multi_hot

    def _create_multi_hot(self, label_ids: List[int], category: str) -> torch.Tensor:
        if category == "triplet":
            # Quick hardcoded mapping for testing
            # ACTIVE_TRIPLETS = [
            #     2,
            #     3,
            #     5,
            #     7,
            #     8,
            #     9,
            #     11,
            #     12,
            #     13,
            #     14,
            #     15,
            #     16,
            #     18,
            #     19,
            #     20,
            #     22,
            #     23,
            #     24,
            #     26,
            #     27,
            #     28,
            #     29,
            #     31,
            #     32,
            #     35,
            #     39,
            #     40,
            #     41,
            #     42,
            #     43,
            #     44,
            #     45,
            #     46,
            #     47,
            #     48,
            #     49,
            #     50,
            #     51,
            #     54,
            #     55,
            #     56,
            #     57,
            #     59,
            #     60,
            #     61,
            #     62,
            #     63,
            #     65,
            #     66,
            # ]

            ACTIVE_TRIPLETS = [
                2,
                5,
                7,
                10,
                11,
                13,
                14,
                15,
                16,
                20,
                22,
                23,
                31,
                32,
                35,
                38,
                39,
                40,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                51,
                52,
                54,
                57,
                61,
                63,
                65,
            ]

            multi_hot = torch.zeros(len(ACTIVE_TRIPLETS))
            for label_id in label_ids:
                if label_id in ACTIVE_TRIPLETS:
                    compact_idx = ACTIVE_TRIPLETS.index(label_id)
                    multi_hot[compact_idx] = 1
            return multi_hot
        else:
            # Original logic for other categories
            multi_hot = torch.zeros(self.num_classes[category])
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
                ast.literal_eval(str(row["action_label"])), "triplet"
            ),
            "video_id": row["video_id"],
        }

        return frames_tensor, labels

    def get_compact_triplet_to_ivt(self):
        """Return a compact triplet_to_ivt mapping"""
        # ACTIVE_TRIPLETS = [
        #     2,
        #     3,
        #     5,
        #     7,
        #     8,
        #     9,
        #     11,
        #     12,
        #     13,
        #     14,
        #     15,
        #     16,
        #     18,
        #     19,
        #     20,
        #     22,
        #     23,
        #     24,
        #     26,
        #     27,
        #     28,
        #     29,
        #     31,
        #     32,
        #     35,
        #     39,
        #     40,
        #     41,
        #     42,
        #     43,
        #     44,
        #     45,
        #     46,
        #     47,
        #     48,
        #     49,
        #     50,
        #     51,
        #     54,
        #     55,
        #     56,
        #     57,
        #     59,
        #     60,
        #     61,
        #     62,
        #     63,
        #     65,
        #     66,
        # ]

        ACTIVE_TRIPLETS = [
            2,
            5,
            7,
            10,
            11,
            13,
            14,
            15,
            16,
            20,
            22,
            23,
            31,
            32,
            35,
            38,
            39,
            40,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            51,
            52,
            54,
            57,
            61,
            63,
            65,
        ]

        compact_mapping = {}
        for compact_idx, original_triplet_id in enumerate(ACTIVE_TRIPLETS):
            if original_triplet_id in self.triplet_to_ivt:
                compact_mapping[compact_idx] = self.triplet_to_ivt[original_triplet_id]

        return compact_mapping

    @staticmethod
    def calculate_video_mean_std(
        clips_dir: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mean and std across all frames in all videos.

        Args:
            clips_dir: Directory containing mp4 files

        Returns:
            tuple: (mean, std) tensors of shape (3,)
        """

        # Get all mp4 files
        video_files = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]

        # Initialize accumulators
        pixel_sum = torch.zeros(3, dtype=torch.float64)
        pixel_sq_sum = torch.zeros(3, dtype=torch.float64)
        num_pixels = 0

        print(f"Processing {len(video_files)} videos...")

        for i, video_file in enumerate(video_files):
            video_path = os.path.join(clips_dir, video_file)
            print(f"Processing {i+1}/{len(video_files)}: {video_file}")

            try:
                video = VideoReader(video_path, ctx=cpu(0))

                # Process all frames
                for frame_idx in range(len(video)):
                    frame = video[frame_idx].asnumpy()

                    # Convert to tensor and normalize to [0, 1]
                    frame_tensor = torch.from_numpy(frame).float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW

                    # Accumulate statistics
                    pixel_sum += frame_tensor.sum(dim=[1, 2]).double()
                    pixel_sq_sum += (frame_tensor**2).sum(dim=[1, 2]).double()
                    num_pixels += frame_tensor.shape[1] * frame_tensor.shape[2]

            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

        # Calculate final statistics
        mean = pixel_sum / num_pixels
        variance = (pixel_sq_sum / num_pixels) - (mean**2)
        std = torch.sqrt(variance)

        # Convert to float32
        mean = mean.float()
        std = std.float()

        print(f"\nProcessed {num_pixels:,} total pixels")
        print(f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

        return mean, std

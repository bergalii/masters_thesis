from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from decord import VideoReader
from decord import cpu
import ast
from typing import List, Dict, Tuple, Optional
import albumentations as A
from torchvision.transforms import Compose, ToTensor, Normalize
from collections import defaultdict


class MultiTaskVideoDataset(Dataset):
    def __init__(
        self,
        clips_dir: str,
        annotations_path: str,
        clip_length: int,
        train: bool = True,
    ):
        self.clips_dir = clips_dir
        self.clip_length = clip_length
        self.train = train

        # Read and process annotations
        self.annotations = pd.read_csv(annotations_path)

        # Initialize label name mappings for each category
        self.label_mappings = {
            "instrument": {},  # Will store instrument_id -> instrument_name
            "verb": {},  # Will store verb_id -> verb_name
            "target": {},  # Will store target_id -> target_name
        }

        # Process string representations of lists and build label mappings
        for idx, row in self.annotations.iterrows():
            # Convert string lists to actual lists
            for col in [
                "instrument_label",
                "verb_label",
                "target_label",
            ]:
                row[col] = ast.literal_eval(row[col])

            # Process action_label_names to extract mappings
            action_names = ast.literal_eval(row["action_label_names"])
            for triplet, inst_id, verb_id, target_id in zip(
                action_names,
                row["instrument_label"],
                row["verb_label"],
                row["target_label"],
            ):
                # Each triplet is in format 'instrument,verb,target'
                inst_name, verb_name, target_name = triplet.split(",")

                # Update mappings
                self.label_mappings["instrument"][inst_id] = inst_name
                self.label_mappings["verb"][verb_id] = verb_name
                self.label_mappings["target"][target_id] = target_name

        # Get number of classes for each category
        self.num_classes = {
            category: max(mapping.keys()) + 1
            for category, mapping in self.label_mappings.items()
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
        if label_ids:  # Check if list is not empty
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
        video = VideoReader(video_path, ctx=cpu(0))

        # Sample frames (implementation remains the same as your previous version)
        indices = np.linspace(0, len(video) - 1, self.clip_length, dtype=int)
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
        }

        return frames, labels


# Function to print label mappings in a readable format
def print_label_mappings(dataset):
    print("\nLabel Mappings for Each Category:")
    mappings = dataset.get_label_names()

    for category, mapping in mappings.items():
        print(f"\n{category.upper()} LABELS:")
        for label_id, label_name in sorted(mapping.items()):
            print(f"  {label_id}: {label_name}")

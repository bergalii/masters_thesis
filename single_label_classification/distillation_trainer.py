import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision.models.video.swin_transformer import swin3d_t, Swin3D_T_Weights
from decord import VideoReader, cpu
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms._transforms_video import ToTensorVideo, NormalizeVideo
import torch.optim as optim
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision.transforms import Compose
import random

train_dir = "/content/drive/MyDrive/SurgicalActions160 Dataset/train"
val_dir = "/content/drive/MyDrive/SurgicalActions160 Dataset/valid"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
num_classes = len(os.listdir(train_dir))  # Number of class folders
SEED_VALUE = 42


class VideoDataset(Dataset):
    def __init__(self, root_dir, train, clip_length):
        self.root_dir = root_dir
        self.train = train
        self.clip_length = clip_length

        self.preprocess = Compose(
            [
                ToTensorVideo(),  # (T, H, W, C) -> (C, T, H, W)
                NormalizeVideo(
                    mean=[0.3656, 0.3660, 0.3670], std=[0.2025, 0.2021, 0.2027]
                ),
            ]
        )

        self.transform = transform = A.ReplayCompose(
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
                A.AdvancedBlur(
                    blur_limit=(3, 7),
                    sigma_x_limit=(0.2, 1.0),
                    sigma_y_limit=(0.2, 1.0),
                    rotate_limit=(-90, 90),
                    beta_limit=(0.5, 8.0),
                    noise_limit=(0.9, 1.1),
                    p=0.3,
                ),
                A.Defocus(radius=(1.5, 2.5), alias_blur=(0.1, 0.2), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.15), p=0.3),
                A.Downscale(
                    scale_range=(0.8, 0.9),
                    interpolation_pair={
                        "downscale": cv2.INTER_LANCZOS4,
                        "upscale": cv2.INTER_LANCZOS4,
                    },
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.6,
                ),
            ]
        )

        self.video_paths = []
        self.labels = []

        # Iterate through class folders
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for video_file in os.listdir(class_path):
                    if video_file.endswith(".mp4"):
                        self.video_paths.append(os.path.join(class_path, video_file))
                        self.labels.append(label)

    def _apply_augmentations(self, frames):
        """Apply the same augmentation transform to all frames in the clip."""

        # Get initial transform params from first frame
        data = self.transform(image=frames[0])

        # Apply same transform to all frames
        augmented_frames = []
        for frame in frames:
            augmented = A.ReplayCompose.replay(data["replay"], image=frame)
            augmented_frames.append(augmented["image"])

        return np.stack(augmented_frames)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Decode video using decord
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        # Uniformly sample clip_length frames
        indices = torch.linspace(0, total_frames - 1, self.clip_length).long()

        frames = vr.get_batch(indices.tolist()).asnumpy()  # (T, H, W, C)

        # Apply augmentations across frames (only to train set)
        if self.train:
            frames = self._apply_augmentations(frames)

        frames = torch.from_numpy(frames)
        # Apply preprocessing & transforms (both train & val set)
        frames = self.preprocess(frames)
        return frames, label

    # Initialize dataset and dataloader
    @staticmethod
    def create_dataloader(root_dir, batch_size, train, clip_length, shuffle):
        dataset = VideoDataset(root_dir=root_dir, train=train, clip_length=clip_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return dataloader

    @staticmethod
    def get_mean_std():
        loader = VideoDataset.create_dataloader(
            root_dir=train_dir, batch_size=8, train=False, clip_length=15
        )
        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)
        num_pixels = 0

        for videos, label in loader:
            batch_size, channels, frames, width, height = videos.shape
            videos = videos.reshape(
                -1, channels, width, height
            )  # Flatten batch and frames

            # Accumulate mean and squared mean
            channels_sum += videos.mean(dim=(0, 2, 3)) * videos.size(0)
            channels_squared_sum += (videos**2).mean(dim=(0, 2, 3)) * videos.size(0)
            num_pixels += videos.size(0)

        # Compute mean and std
        mean = channels_sum / num_pixels
        std = torch.sqrt(channels_squared_sum / num_pixels - mean**2)

        return mean, std


class SelfDistillationTrainer:
    def __init__(
        self,
        num_epochs: int,
        batch_size: int,
        clip_length: int,
        warmup_epochs: int = 1,
        temperature: float = 1.0,
    ):
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.temperature = temperature

        # Create dataloaders
        self.train_loader = VideoDataset.create_dataloader(
            root_dir=train_dir,
            batch_size=batch_size,
            train=True,
            clip_length=clip_length,
            shuffle=True,
        )
        self.val_loader = VideoDataset.create_dataloader(
            root_dir=val_dir,
            batch_size=batch_size,
            train=False,
            clip_length=clip_length,
            shuffle=False,
        )

    def configure_training(self, teacher_model, student_model, optimizer, lr_scheduler):
        self.teacher_model = teacher_model.to(device) if teacher_model else None
        self.student_model = student_model.to(device) if student_model else None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_teacher(self):
        """Train teacher model with hard labels"""
        print("Training teacher model...")
        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self._train_teacher_epoch()

            # Evaluate
            val_loss, val_acc = self._evaluate_teacher()

            self.lr_scheduler.step(val_acc)

            # Print metrics
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.teacher_model.state_dict(), "best_teacher.pth")
                print(
                    f"New best teacher model saved with validation accuracy: {val_acc:.4f}"
                )

    def generate_soft_labels(self):
        """Generate soft labels using trained teacher model"""
        print("Generating soft labels...")
        self.teacher_model.eval()
        soft_labels = []

        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(device)
                outputs = self.teacher_model(inputs)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                soft_labels.append(probs.cpu())

        return torch.cat(soft_labels, dim=0)

    def train_student(self, soft_labels):
        """Train student model with soft labels"""
        print("Training student model...")
        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self._train_student_epoch(soft_labels)

            # Evaluate
            val_loss, val_acc = self._evaluate_student()

            self.lr_scheduler.step(val_acc)

            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.student_model.state_dict(), "best_student.pth")
                print(
                    f"New best student model saved with validation accuracy: {val_acc:.4f}"
                )

    def _train_teacher_epoch(self):
        self.teacher_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.teacher_model(inputs)
            # Use BCE loss for teacher
            loss = F.binary_cross_entropy_with_logits(
                outputs, F.one_hot(labels, num_classes=outputs.size(1)).float()
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.teacher_model.parameters(), max_norm=5.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.train_loader), correct / total

    def _train_student_epoch(self, soft_labels):
        self.student_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(device)
            batch_soft_labels = soft_labels[
                batch_idx
                * self.train_loader.batch_size : (batch_idx + 1)
                * self.train_loader.batch_size
            ].to(device)

            outputs = self.student_model(inputs)
            # Use BCE loss between student outputs and teacher's soft labels
            loss = F.binary_cross_entropy_with_logits(
                outputs / self.temperature, batch_soft_labels / self.temperature
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(), max_norm=5.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)

        return total_loss / len(self.train_loader), correct / total

    def _evaluate_teacher(self):
        return self._evaluate_model(self.teacher_model)

    def _evaluate_student(self):
        return self._evaluate_model(self.student_model)

    def _evaluate_model(self, model):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = F.binary_cross_entropy_with_logits(
                    outputs, F.one_hot(labels, num_classes=outputs.size(1)).float()
                )

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total


# 1. Initialize teacher & student models
teacher_model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
student_model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)

# 2. Modify their heads
for model in [teacher_model, student_model]:
    model.head = nn.Sequential(
        nn.LayerNorm(model.num_features),
        nn.Dropout(p=0.5),
        nn.Linear(model.num_features, 512),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

#############################################
# TEACHER
#############################################

# 3. Define the teacher optimizer + scheduler
teacher_decay = []
teacher_no_decay = []
for name, param in teacher_model.named_parameters():
    if not param.requires_grad:
        continue
    if "bias" in name:
        teacher_no_decay.append(param)
    else:
        teacher_decay.append(param)

teacher_optimizer = torch.optim.SGD(
    [
        {"params": teacher_decay, "weight_decay": 0.0001},
        {"params": teacher_no_decay, "weight_decay": 0.0},
    ],
    lr=0.001,
    momentum=0.9,
    nesterov=True,
)

teacher_scheduler = ReduceLROnPlateau(
    teacher_optimizer, mode="max", factor=0.1, patience=4
)

# 4. Initialize trainer for teacher
trainer = SelfDistillationTrainer(
    num_epochs=20, batch_size=4, clip_length=10, temperature=1.0
)

# 5. Train teacher
trainer.configure_training(
    teacher_model=teacher_model,
    student_model=None,
    optimizer=teacher_optimizer,
    lr_scheduler=teacher_scheduler,
)
trainer.train_teacher()

#############################################
# STUDENT
#############################################

# 6. Generate soft labels from the trained teacher
soft_labels = trainer.generate_soft_labels()

# 7. Define the student optimizer + scheduler
student_decay = []
student_no_decay = []
for name, param in student_model.named_parameters():
    if not param.requires_grad:
        continue
    if "bias" in name:
        student_no_decay.append(param)
    else:
        student_decay.append(param)

student_optimizer = torch.optim.SGD(
    [
        {"params": student_decay, "weight_decay": 0.0001},
        {"params": student_no_decay, "weight_decay": 0.0},
    ],
    lr=0.001,
    momentum=0.9,
    nesterov=True,
)

student_scheduler = ReduceLROnPlateau(
    student_optimizer, mode="max", factor=0.1, patience=4
)

trainer.configure_training(
    teacher_model=None,
    student_model=student_model,
    optimizer=student_optimizer,
    lr_scheduler=student_scheduler,
)
trainer.train_student(soft_labels)

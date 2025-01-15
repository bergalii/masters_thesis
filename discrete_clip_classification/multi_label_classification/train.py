import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision.transforms import Compose
import cv2
from typing import List, Dict, Tuple
import torch.nn.functional as F

import torch.nn as nn

from torchvision.models.video.swin_transformer import swin3d_t, Swin3D_T_Weights
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from collections import defaultdict

TRAIN_FRAMES_DIR = r"/content/drive/MyDrive/TUHH Master's Studies/Master's Thesis/dataset/dataset/images"
TRAIN_ANNOTATIONS_DIR = r"/content/drive/MyDrive/TUHH Master's Studies/Master's Thesis/dataset/dataset/annotations"
VAL_FRAMES_DIR = r"/content/drive/MyDrive/TUHH Master's Studies/Master's Thesis/dataset/dataset/images"
VAL_ANNOTATIONS_DIR = r"/content/drive/MyDrive/TUHH Master's Studies/Master's Thesis/dataset/dataset/annotations"
CLASSES = [
    "CuttingMesocolon",
    "PullingVasDeferens",
    "ClippingVasDeferens",
    "CuttingVasDeferens",
    "ClippingTissue",
    "PullingSeminalVesicle",
    "ClippingSeminalVesicle",
    "CuttingSeminalVesicle",
    "SuckingBlood",
    "SuckingSmoke",
    "PullingTissue",
    "CuttingTissue",
    "BaggingProstate",
    "BladderNeckDissection",
    "BladderAnastomosis",
    "PullingProstate",
    "ClippingBladderNeck",
    "CuttingThread",
    "UrethraDissection",
    "CuttingProstate",
    "PullingBladderNeck",
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize trainer
trainer = SelfDistillationTrainer(
    num_epochs=50,
    batch_size=1,
    train_loader=train_loader,
    val_loader=val_loader,
    transition_matrix=train_dataset.transition_matrix,
    action_durations=train_dataset.action_durations,
    warmup_epochs=1,
)

# Train with self-distillation
trainer.train()

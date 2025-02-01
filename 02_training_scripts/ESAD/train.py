import os
import torch
from torch.utils.data import DataLoader
from dataset import MultiLabelFrameDataset
from trainer import SelfDistillationTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

num_epochs = 50
batch_size = 1
max_clip_length = 15
warmup_epochs = 1


train_dataset = MultiLabelFrameDataset(
    frames_dir=TRAIN_FRAMES_DIR,
    annotations_dir=TRAIN_ANNOTATIONS_DIR,
    max_clip_length=max_clip_length,
    train=True,
    num_classes=NUM_CLASSES,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Initialize trainer
trainer = SelfDistillationTrainer(
    num_epochs=50,
    batch_size=1,
    train_loader=train_loader,
    val_loader=val_loader,
    transition_matrix=train_dataset.transition_matrix,
    action_durations=train_dataset.action_durations,
    warmup_epochs=1,
    device=DEVICE,
)

# Train with self-distillation
trainer.train()

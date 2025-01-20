from dataset import MultiTaskVideoDataset, print_label_mappings
from torch.utils.data import DataLoader
from trainer import SelfDistillationTrainer
import torch


CLIPS_DIR = r"videos"
ANNOTATIONS_PATH = r"annotations.csv"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

num_epochs = 50
batch_size = 2
warmup_epochs = 1


train_dataset = MultiTaskVideoDataset(
    clips_dir=CLIPS_DIR, annotations_path=ANNOTATIONS_PATH, clip_length=8, train=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Initialize trainer
trainer = SelfDistillationTrainer(
    num_epochs=50,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=train_dataset.num_classes["verb"],
    warmup_epochs=1,
    device=DEVICE,
)

# Train with self-distillation
trainer.train()

from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from trainer_multi import MultiTaskSelfDistillationTrainer

# from trainer_multi_ivt import MultiTaskSelfDistillationTrainer
import torch
from utils import setup_logging, set_seeds
import ast
from pathlib import Path

CLIPS_DIR = r"videos"
ANNOTATIONS_PATH = r"annotations.csv"

dir_name, logger = setup_logging("training")
model_dir = Path(f"models/{dir_name}")
model_dir.mkdir(exist_ok=True)

torch.cuda.set_device(1)
DEVICE = torch.device("cuda:1")
logger.info(f"Cuda is active and using GPU: {torch.cuda.current_device()}")
set_seeds()


def get_label_counts(dataset, category):
    counts = {label_id: 0 for label_id in dataset.label_mappings[category].keys()}
    for idx in range(len(dataset.annotations)):
        row = dataset.annotations.iloc[idx]
        labels = ast.literal_eval(str(row[f"{category}_label"]))
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def print_and_get_mappings(dataset):
    mappings = dataset.get_label_names()
    all_mappings = {}

    for category in ["instrument", "verb", "target", "triplet"]:
        counts = get_label_counts(dataset, category)
        logger.info(f"{category.upper()} LABELS:")
        for label_id, label_name in sorted(mappings[category].items()):
            count = counts.get(label_id, 0)
            logger.info(f"  {label_id}: {label_name} - {count}")
        all_mappings[category] = mappings[category]

    return all_mappings


num_epochs = 50
batch_size = 8
warmup_epochs = 5
clip_length = 10

train_dataset = MultiTaskVideoDataset(
    clips_dir=CLIPS_DIR,
    annotations_path=ANNOTATIONS_PATH,
    clip_length=10,
    split="train",
    train=True,
)
val_dataset = MultiTaskVideoDataset(
    clips_dir=CLIPS_DIR,
    annotations_path=ANNOTATIONS_PATH,
    clip_length=10,
    split="val",
    train=False,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# MultiTaskVideoDataset.calculate_video_mean_std(train_loader)

logger.info("-- Training Dataset Classes--")
mappings = print_and_get_mappings(train_dataset)
logger.info("-- Validation Dataset Classes -- ")
print_and_get_mappings(val_dataset)
logger.info("-" * 50)

trainer = MultiTaskSelfDistillationTrainer(
    num_epochs=num_epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    label_mappings=mappings,
    num_classes={
        "verb": train_dataset.num_classes["verb"],
        "instrument": train_dataset.num_classes["instrument"],
        "target": train_dataset.num_classes["target"],
        "triplet": train_dataset.num_classes["triplet"],
    },
    warmup_epochs=1,
    device=DEVICE,
    logger=logger,
    dir_name=model_dir,
)

trainer.train()

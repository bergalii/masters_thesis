from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from trainer_multi import MultiTaskSelfDistillationTrainer
import torch
from utils import setup_logging, set_seeds, print_and_get_mappings, load_configs
from pathlib import Path


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

    train_dataset = MultiTaskVideoDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        clip_length=configs["clip_length"],
        split="train",
        train_ratio=configs["train_ratio"],
        train=True,
        frame_width=configs["frame_width"],
        frame_height=configs["frame_height"],
        min_occurrences=configs["min_occurrences"],
        cross_val_fold=configs["val_split"],
    )
    val_dataset = MultiTaskVideoDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        clip_length=configs["clip_length"],
        split="val",
        train_ratio=configs["train_ratio"],
        train=False,
        frame_width=configs["frame_width"],
        frame_height=configs["frame_height"],
        min_occurrences=configs["min_occurrences"],
        cross_val_fold=configs["val_split"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], shuffle=False
    )

    logger.info("-- Training Dataset Classes--")
    mappings = print_and_get_mappings(train_dataset, logger)
    logger.info("-- Validation Dataset Classes -- ")
    print_and_get_mappings(val_dataset, logger)
    logger.info("-" * 50)

    trainer = MultiTaskSelfDistillationTrainer(
        num_epochs=configs["num_epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        label_mappings=mappings,
        num_classes=train_dataset.num_classes,
        triplet_to_ivt=train_dataset.triplet_to_ivt,
        warmup_epochs=configs["warmup_epochs"],
        learning_rate=configs["learning_rate"],
        weight_decay=configs["weight_decay"],
        hidden_layer_dim=configs["hidden_layer_dim"],
        gradient_clipping=configs["gradient_clipping"],
        guidance_scale=configs["guidance_scale"],
        device=DEVICE,
        logger=logger,
        dir_name=model_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()

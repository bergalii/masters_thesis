from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from trainer import MultiTaskSelfDistillationTrainer
import torch
from utils import (
    setup_logging,
    set_seeds,
    load_configs,
    print_combined_mappings,
)
from pathlib import Path


def main():
    CLIPS_DIR = r"05_datasets_dir/UKE/clips"
    ANNOTATIONS_PATH = r"/data/Berk/masters_thesis/annotations_combined.csv"
    # ANNOTATIONS_PATH = r"05_datasets_dir/UKE/gt.csv"
    CONFIGS_PATH = r"02_training_scripts/CholecT50/configs.yaml"

    dir_name, logger = setup_logging("training")
    model_dir = Path(f"04_models_dir/{dir_name}_UKE")
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
    )
    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], shuffle=False
    )

    logger.info("--Training and Validation Dataset Classes--")
    mappings = print_combined_mappings(train_dataset, val_dataset, logger)
    logger.info("-" * 50)

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

    compact_triplet_to_ivt = train_dataset.get_compact_triplet_to_ivt()
    trainer = MultiTaskSelfDistillationTrainer(
        num_epochs=configs["num_epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        label_mappings=mappings,
        # num_classes=train_dataset.num_classes,
        num_classes={
            "instrument": train_dataset.num_classes["instrument"],
            "verb": train_dataset.num_classes["verb"],
            "target": train_dataset.num_classes["target"],
            "triplet": len(ACTIVE_TRIPLETS),
        },
        # triplet_to_ivt=train_dataset.triplet_to_ivt,
        triplet_to_ivt=compact_triplet_to_ivt,
        warmup_epochs=configs["warmup_epochs"],
        learning_rate=configs["learning_rate"],
        weight_decay=configs["weight_decay"],
        hidden_layer_dim=configs["hidden_layer_dim"],
        attention_module_common_dim=configs["attention_module_common_dim"],
        gradient_clipping=configs["gradient_clipping"],
        guidance_scale=configs["guidance_scale"],
        consistency_loss_weight=configs["consistency_loss_weight"],
        device=DEVICE,
        logger=logger,
        dir_name=model_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()

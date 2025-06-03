import logging
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
import os
import ast
import yaml
from typing import Dict


def setup_logging(experiment_name: str) -> logging.Logger:
    # Create logs directory
    log_dir = Path("03_logs_dir")
    log_dir.mkdir(exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        filename=str(log_file), level=logging.INFO, format="%(message)s"
    )

    return f"{experiment_name}_{timestamp}", logging.getLogger(experiment_name)


def set_seeds(seed_value: int = 42) -> None:
    # Python's random module
    random.seed(seed_value)
    # Numpy
    np.random.seed(seed_value)
    # PyTorch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    # Set CUDA deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed_value)


def load_configs(config_path: str, logger: logging.Logger = None) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        if logger:
            logger.info("HYPERPARAMS:")
            for key, value in config.items():
                logger.info(f"  {key}: {value}")
    return config


def get_label_counts(dataset, category) -> Dict:
    counts = {label_id: 0 for label_id in dataset.label_mappings[category].keys()}
    for idx in range(len(dataset.annotations)):
        row = dataset.annotations.iloc[idx]
        # labels = ast.literal_eval(str(row[f"{category}_label"]))
        if category == "triplet":
            column_name = "action_label"
        else:
            column_name = f"{category}_label"
        labels = ast.literal_eval(str(row[column_name]))
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def print_and_get_mappings(dataset, logger: logging.Logger) -> Dict:
    mappings = dataset.label_mappings
    all_mappings = {}

    for category in ["instrument", "verb", "target", "triplet"]:
        counts = get_label_counts(dataset, category)
        logger.info(f"{category.upper()} LABELS:")
        for label_id, label_name in sorted(mappings[category].items()):
            count = counts.get(label_id, 0)
            logger.info(f"  {label_id}: {label_name} - {count}")
        all_mappings[category] = mappings[category]

    return all_mappings


def print_combined_mappings(train_dataset, val_dataset, logger: logging.Logger) -> Dict:
    train_mappings = train_dataset.label_mappings
    all_mappings = {}

    for category in ["instrument", "verb", "target", "triplet"]:
        train_counts = get_label_counts(train_dataset, category)
        val_counts = get_label_counts(val_dataset, category)

        logger.info(f"{category.upper()} LABELS:")
        for label_id, label_name in sorted(train_mappings[category].items()):
            train_count = train_counts.get(label_id, 0)
            val_count = val_counts.get(label_id, 0)
            logger.info(f"  {label_id}: {label_name} - {train_count}, {val_count}")

        all_mappings[category] = train_mappings[category]

    return all_mappings


def resolve_nan(class_aps):
    equiv_nan = ["-0", "-0.", "-0.0", "-.0"]
    class_aps = list(map(str, class_aps))
    class_aps = [np.nan if x in equiv_nan else x for x in class_aps]
    class_aps = np.array(list(map(float, class_aps)))
    return class_aps

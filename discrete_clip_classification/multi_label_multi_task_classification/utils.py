import logging
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
import os


def setup_logging(experiment_name: str) -> logging.Logger:
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%m-%d %H:%M",
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

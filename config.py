import os
import torch
from pathlib import Path

# Data directories
# Get base directory relative to current file location
BASE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "code_data" / "nn_img_recog"
DATASET_DIR = BASE_DATA_DIR / "dataset"
CHECKPOINT_DIR = BASE_DATA_DIR / "checkpoints"

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 90
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Model parameters
NUM_CLASSES = 10
DROPOUT_RATE = 0.3

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data parameters
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

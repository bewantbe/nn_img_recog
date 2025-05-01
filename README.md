# AlexNet Implementation for CIFAR-10

This project implements AlexNet for the CIFAR-10 dataset using PyTorch. The implementation is adapted to handle the smaller input size of CIFAR-10 (32x32) compared to the original AlexNet architecture (224x224).

## Project Structure

```
.
├── config.py           # Configuration parameters
├── model.py           # AlexNet model implementation
├── data_loader.py     # Dataset and data loader utilities
├── utils.py           # Helper functions
├── train.py           # Training script
└── requirements.txt   # Python dependencies
```

## Setup

1. Set up virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Storage

All large files (dataset, checkpoints) are stored in `E:/code_data/nn_img_recog/`:
- Dataset: `E:/code_data/nn_img_recog/dataset/`
- Checkpoints: `E:/code_data/nn_img_recog/checkpoints/`

## Training

To train the model:
```bash
python train.py
```

The training script:
- Uses CIFAR-10 dataset (automatically downloaded)
- Implements data augmentation
- Uses SGD optimizer with momentum
- Includes learning rate scheduling
- Saves checkpoints during training
- Tracks and saves the best model

Training progress is displayed with:
- Batch progress
- Loss values
- Accuracy metrics
- Learning rate

## Model Architecture

The AlexNet architecture has been adapted for CIFAR-10:
- Modified first layer for 32x32 input
- Adjusted kernel sizes and strides
- Preserved key architectural features:
  * ReLU activations
  * Local response normalization
  * Dropout regularization
  * Dense layers configuration

## Checkpointing

The training script automatically saves:
- Regular checkpoints (`checkpoint.pth`)
- Best model (`model_best.pth`)

Checkpoints include:
- Model state
- Optimizer state
- Current epoch
- Best accuracy

# AlexNet Architectural Variations Study

## Introduction
This report summarizes an experimental study of different architectural variations of the AlexNet model trained on CIFAR-10 dataset. The study explores modifications to both convolutional layers and fully connected layers to understand their impact on model performance.

## Architectural Variations

### Convolutional Layer Variations

1. **Original AlexNet-like (conv1)**
   - Configuration: `[(3, 64, 3), (64, 192, 3), (192, 384, 3), (384, 256, 3), (256, 256, 3)]`
   - Standard 3×3 kernels following original AlexNet pattern

2. **Wider Network (conv2)**
   - Configuration: `[(3, 128, 3), (128, 256, 3), (256, 512, 3), (512, 384, 3), (384, 384, 3)]`
   - Increased channel dimensions for better feature representation

3. **Narrower Network (conv3)**
   - Configuration: `[(3, 32, 3), (32, 96, 3), (96, 192, 3), (192, 128, 3), (128, 128, 3)]`
   - Reduced channel dimensions for lighter model

4. **Larger Kernels (conv4)**
   - Configuration: `[(3, 64, 5), (64, 192, 5), (192, 384, 5), (384, 256, 5), (256, 256, 5)]`
   - Using 5×5 kernels for larger receptive field

5. **Mixed Kernel Sizes (conv5)**
   - Configuration: `[(3, 64, 3), (64, 192, 5), (192, 384, 7), (384, 256, 5), (256, 256, 3)]`
   - Progressive kernel size changes (3→5→7→5→3)

### Fully Connected Layer Variations

1. **Original (fc1)**
   - Sizes: `[4096, 4096]`
   - Standard AlexNet FC configuration

2. **Smaller (fc2)**
   - Sizes: `[2048, 2048]`
   - Reduced parameter count

3. **Larger (fc3)**
   - Sizes: `[8192, 8192]`
   - Increased parameter count

4. **Decreasing (fc4)**
   - Sizes: `[4096, 2048]`
   - Progressive dimension reduction

5. **Increasing (fc5)**
   - Sizes: `[2048, 4096]`
   - Progressive dimension increase

## Results

To view the results, run `test_architecture.py` script. The script will:
1. Train each architectural variation for 50 epochs
2. Save training and validation metrics
3. Generate a CSV file with final accuracies
4. The results will be automatically updated in this report

## Analysis

Once results are available, this section will include:
- Comparison of different architectural variations
- Impact of network width vs depth
- Effect of kernel sizes
- Influence of FC layer configurations
- Best performing combinations

## Training Setup
- Learning Rate: See config.py
- Momentum: See config.py
- Weight Decay: See config.py
- Number of Classes: 10 (CIFAR-10)
- Dropout Rate: See config.py
- Epochs: 50

## Conclusions

This section will be updated with conclusions after running the experiments, highlighting:
- Most effective architectural choices
- Performance-complexity trade-offs
- Recommendations for future improvements

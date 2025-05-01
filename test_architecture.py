# This script tests different architectural variations of the AlexNet model using PyTorch.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import csv
import time
from pathlib import Path
import numpy as np

from config import (
    DEVICE, LEARNING_RATE, MOMENTUM, 
    WEIGHT_DECAY, NUM_CLASSES, DROPOUT_RATE
)
from data_loader import get_data_loaders
from utils import AverageMeter, accuracy

class ModifiedAlexNet(nn.Module):
    def __init__(self, conv_config, fc_sizes, num_classes=NUM_CLASSES):
        super(ModifiedAlexNet, self).__init__()
        
        # Feature extractors (convolutional layers)
        layers = []
        for i, (in_channels, out_channels, kernel_size) in enumerate(conv_config):
            # Add convolutional layer
            padding = kernel_size // 2  # Maintain spatial dimensions
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            
            # Add MaxPool after every two conv layers (similar to original AlexNet)
            if i % 2 == 0:
                layers.append(nn.MaxPool2d(kernel_size=2))
        
        self.features = nn.Sequential(*layers)
        
        # Calculate the size of flattened features
        # Assuming input size is 32x32 (CIFAR-10)
        # After 3 MaxPool layers (stride=2), spatial dimensions are reduced by factor of 8
        feature_size = conv_config[-1][1] * 4 * 4  # 256 * 4 * 4 in original AlexNet
        
        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(feature_size, fc_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_sizes[1], num_classes),
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def train_model(model, train_loader, val_loader, architecture_name, num_epochs=50):
    writer = SummaryWriter(f'runs/alex/{architecture_name}/constant_dropout{DROPOUT_RATE}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs} (AlexNet architecture variation: {architecture_name})')
        epoch_start_time = time.time()
        
        # Train
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        for i, (images, target) in enumerate(train_loader):
            images, target = images.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            
            acc1 = accuracy(output, target)[0]
            train_loss.update(loss.item(), images.size(0))
            train_acc.update(acc1.item(), images.size(0))
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Train: [{i}/{len(train_loader)}]\t'
                      f'Loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'
                      f'Acc@1 {train_acc.val:.3f} ({train_acc.avg:.3f})')
        
        # Validate
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images, target = images.to(DEVICE), target.to(DEVICE)
                
                output = model(images)
                loss = criterion(output, target)
                
                acc1 = accuracy(output, target)[0]
                val_loss.update(loss.item(), images.size(0))
                val_acc.update(acc1.item(), images.size(0))
        
        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/valid', val_loss.avg, epoch)
        writer.add_scalar('Accuracy/train', train_acc.avg, epoch)
        writer.add_scalar('Accuracy/valid', val_acc.avg, epoch)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f'AlexNet architecture variation: {architecture_name} Epoch: {epoch+1}')
        print(f'Training Loss: {train_loss.avg:.4f}, Training Acc: {train_acc.avg:.2f}%')
        print(f'Validation Loss: {val_loss.avg:.4f}, Validation Acc: {val_acc.avg:.2f}%')
        print(f'Epoch Duration: {epoch_duration:.2f} seconds')
        
        train_losses.append(train_loss.avg)
        train_accs.append(train_acc.avg)
        val_losses.append(val_loss.avg)
        val_accs.append(val_acc.avg)
    
    writer.close()
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1]
    }

def main():
    start_time = time.time()
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    # Define architectural variations
    conv_variations = [
        # Format: [(in_channels, out_channels, kernel_size), ...]
        # Original AlexNet-like
        [(3, 64, 3), (64, 192, 3), (192, 384, 3), (384, 256, 3), (256, 256, 3)],
        # Wider networks (more channels)
        [(3, 128, 3), (128, 256, 3), (256, 512, 3), (512, 384, 3), (384, 384, 3)],
        # Narrower networks (fewer channels)
        [(3, 32, 3), (32, 96, 3), (96, 192, 3), (192, 128, 3), (128, 128, 3)],
        # Larger kernels
        [(3, 64, 5), (64, 192, 5), (192, 384, 5), (384, 256, 5), (256, 256, 5)],
        # Mixed kernel sizes
        [(3, 64, 3), (64, 192, 5), (192, 384, 7), (384, 256, 5), (256, 256, 3)]
    ]
    
    fc_variations = [
        # Format: [hidden1_size, hidden2_size]
        [4096, 4096],  # Original
        [2048, 2048],  # Smaller
        [8192, 8192],  # Larger
        [4096, 2048],  # Decreasing
        [2048, 4096]   # Increasing
    ]
    
    results = []
    
    for i, conv_config in enumerate(conv_variations):
        for j, fc_sizes in enumerate(fc_variations):
            architecture_name = f"conv{i+1}_fc{j+1}"
            print(f"\nTesting AlexNet architecture variation: {architecture_name}")
            print("Conv config:", conv_config)
            print("FC sizes:", fc_sizes)
            
            # Create model with current architecture
            model = ModifiedAlexNet(conv_config, fc_sizes).to(DEVICE)
            
            # Train and evaluate
            result = train_model(model, train_loader, val_loader, architecture_name)
            
            # Store results
            results.append({
                'architecture': architecture_name,
                'conv_config': str(conv_config),
                'fc_sizes': str(fc_sizes),
                'final_train_acc': result['final_train_acc'],
                'final_val_acc': result['final_val_acc']
            })
            
            # Save current results to CSV
            with open(results_dir / 'architecture_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['architecture', 'conv_config', 'fc_sizes', 
                                                     'final_train_acc', 'final_val_acc'])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\nResults for AlexNet architecture variation {architecture_name}:")
            print(f"Final Training Accuracy: {result['final_train_acc']:.2f}%")
            print(f"Final Validation Accuracy: {result['final_val_acc']:.2f}%")
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\nTotal Runtime: {total_runtime:.2f} seconds ({total_runtime/3600:.2f} hours)")

if __name__ == '__main__':
    main()

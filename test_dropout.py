# This script tests the effect of different dropout rates on the performance of a neural network model (AlexNet) using PyTorch.

# run tensorboard by command: tensorboard --logdir=runs

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
    WEIGHT_DECAY, CHECKPOINT_DIR
)
from model import AlexNet
from data_loader import get_data_loaders
from utils import AverageMeter, accuracy

def train_model(model, train_loader, val_loader, dropout_ratio, num_epochs=50):
    writer = SummaryWriter(f'runs/dropout_{dropout_ratio:.1f}')
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
        print(f'\nEpoch: {epoch+1}/{num_epochs} (Dropout: {dropout_ratio})')
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
        
        print(f'Dropout: {dropout_ratio} Epoch: {epoch+1}')
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
    
    # Test different dropout ratios
    dropout_ratios = np.arange(0, 1.0, 0.1)
    results = []
    
    for dropout_ratio in dropout_ratios:
        print(f"\nTesting dropout ratio: {dropout_ratio:.1f}")
        
        # Create model with current dropout ratio
        model = AlexNet().to(DEVICE)
        # Update dropout rates in the classifier
        model.classifier[0].p = dropout_ratio
        model.classifier[3].p = dropout_ratio
        
        # Train and evaluate
        result = train_model(model, train_loader, val_loader, dropout_ratio)
        
        # Store results
        results.append({
            'dropout_ratio': dropout_ratio,
            'final_train_acc': result['final_train_acc'],
            'final_val_acc': result['final_val_acc']
        })
        
        # Save current results to CSV
        with open(results_dir / 'dropout_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['dropout_ratio', 'final_train_acc', 'final_val_acc'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults for dropout_ratio = {dropout_ratio:.1f}:")
        print(f"Final Training Accuracy: {result['final_train_acc']:.2f}%")
        print(f"Final Validation Accuracy: {result['final_val_acc']:.2f}%")

    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\nTotal Runtime: {total_runtime:.2f} seconds ({total_runtime/3600:.2f} hours)")

if __name__ == '__main__':
    main()

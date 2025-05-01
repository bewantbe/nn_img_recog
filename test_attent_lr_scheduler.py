# This script tests different learning rate scheduling strategies on the performance of an attention-based neural network model using PyTorch.

# run tensorboard by command: tensorboard --logdir=runs_attent

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import csv
import time
from pathlib import Path
import numpy as np
from torch.optim.lr_scheduler import (
    StepLR, 
    ReduceLROnPlateau, 
    CosineAnnealingLR, 
    OneCycleLR,
    ExponentialLR
)

from config import (
    DEVICE, LEARNING_RATE, MOMENTUM, 
    WEIGHT_DECAY, CHECKPOINT_DIR, NUM_EPOCHS
)
from model_attent import AttentionNet
from data_loader import get_data_loaders
from utils import AverageMeter, accuracy

def get_scheduler(scheduler_name, optimizer, train_loader_len):
    """
    Create and return the specified learning rate scheduler
    """
    if scheduler_name == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif scheduler_name == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE * 10,  # peak learning rate
            epochs=NUM_EPOCHS,
            steps_per_epoch=train_loader_len,
            pct_start=0.3,  # spend 30% of iterations in increasing LR
            anneal_strategy='cos'
        )
    elif scheduler_name == 'exp':
        return ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def train_model(model, train_loader, val_loader, scheduler_name, num_epochs=NUM_EPOCHS):
    writer = SummaryWriter(f'runs_attent/scheduler_{scheduler_name}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = get_scheduler(scheduler_name, optimizer, len(train_loader))
    is_plateau_scheduler = isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs} (Attention Model, Scheduler: {scheduler_name})')
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
            
            # Step OneCycleLR scheduler after each batch
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            if i % 100 == 0:
                print(f'Train: [{i}/{len(train_loader)}]\t'
                      f'Loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'
                      f'Acc@1 {train_acc.val:.3f} ({train_acc.avg:.3f})\t'
                      f'LR {optimizer.param_groups[0]["lr"]:.6f}')
        
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
        
        # Step learning rate scheduler (except OneCycleLR which is stepped per batch)
        if is_plateau_scheduler:
            scheduler.step(val_acc.avg)
        elif not isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/valid', val_loss.avg, epoch)
        writer.add_scalar('Accuracy/train', train_acc.avg, epoch)
        writer.add_scalar('Accuracy/valid', val_acc.avg, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f'Attention Model - Scheduler: {scheduler_name} Epoch: {epoch+1}')
        print(f'Training Loss: {train_loss.avg:.4f}, Training Acc: {train_acc.avg:.2f}%')
        print(f'Validation Loss: {val_loss.avg:.4f}, Validation Acc: {val_acc.avg:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
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
        'learning_rates': learning_rates,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'best_val_acc': max(val_accs)
    }

def main():
    start_time = time.time()
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    # Test different schedulers
    schedulers = ['step', 'plateau', 'cosine', 'onecycle', 'exp']
    results = []
    
    for scheduler_name in schedulers:
        print(f"\nTesting attention model with scheduler: {scheduler_name}")
        
        # Create model
        model = AttentionNet().to(DEVICE)
        
        # Train and evaluate
        result = train_model(model, train_loader, val_loader, scheduler_name)
        
        # Store results
        results.append({
            'scheduler': scheduler_name,
            'final_train_acc': result['final_train_acc'],
            'final_val_acc': result['final_val_acc'],
            'best_val_acc': result['best_val_acc']
        })
        
        # Save current results to CSV
        with open(results_dir / 'attent_scheduler_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['scheduler', 'final_train_acc', 
                                                 'final_val_acc', 'best_val_acc'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults for attention model with scheduler = {scheduler_name}:")
        print(f"Final Training Accuracy: {result['final_train_acc']:.2f}%")
        print(f"Final Validation Accuracy: {result['final_val_acc']:.2f}%")
        print(f"Best Validation Accuracy: {result['best_val_acc']:.2f}%")
        
        # Save learning rate curve
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(result['learning_rates'])
        plt.title(f'Learning Rate Schedule - Attention Model with {scheduler_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(results_dir / f'lr_curve_attent_{scheduler_name}.png')
        plt.close()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\nTotal Runtime: {total_runtime:.2f} seconds ({total_runtime/3600:.2f} hours)")

if __name__ == '__main__':
    main()

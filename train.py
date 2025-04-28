import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from pathlib import Path

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, MOMENTUM, 
    WEIGHT_DECAY, CHECKPOINT_DIR
)
from model import AlexNet
from data_loader import get_data_loaders
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint, get_lr

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to train mode
    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        images, target = images.to(DEVICE), target.to(DEVICE)

        # Forward pass
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        acc1 = accuracy(output, target)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # Compute gradients and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'LR {get_lr(optimizer):.6f}')

    return losses.avg, top1.avg

def validate(model, val_loader, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(DEVICE), target.to(DEVICE)

            # Forward pass
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(f' * Acc@1 {top1.avg:.3f}')
    return losses.avg, top1.avg

def main():
    # Create model
    model = AlexNet().to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Get data loaders
    train_loader, val_loader = get_data_loaders()

    # Optionally resume from a checkpoint
    start_epoch, best_acc1 = load_checkpoint(model, optimizer)
    
    # Train model
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'\nEpoch: {epoch+1}/{NUM_EPOCHS}')
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch)

        # Evaluate on validation set
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step()

        # Remember best acc@1 and save checkpoint
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)

        save_checkpoint(
            model, 
            optimizer,
            epoch + 1,
            best_acc1,
            filename='checkpoint.pth'
        )
        
        if is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_acc1,
                filename='model_best.pth'
            )

        print(f'Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
        print(f'Best Validation Acc: {best_acc1:.2f}%')

if __name__ == '__main__':
    main()

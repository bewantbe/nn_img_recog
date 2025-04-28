import torch
import os
from pathlib import Path
from config import CHECKPOINT_DIR

def save_checkpoint(model, optimizer, epoch, best_acc, filename='checkpoint.pth'):
    """Save model checkpoint to disk"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    path = CHECKPOINT_DIR / filename
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """Load model checkpoint from disk"""
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        return 0, 0  # epoch, best_acc

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_acc']

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_lr(optimizer):
    """Get the current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

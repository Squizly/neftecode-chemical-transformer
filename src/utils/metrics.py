import torch
import torch.nn.functional as F

def custom_mae_loss(pred, target):
    loss = F.huber_loss(pred, target, reduction='none', delta=1.0)
    return (loss * torch.tensor([0.6, 0.4], device=pred.device)).mean()

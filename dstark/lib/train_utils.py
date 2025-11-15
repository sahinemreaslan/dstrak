"""
Training utilities
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from typing import Dict, Any


def build_optimizer(model, config: Dict[str, Any]):
    """Build optimizer"""

    # Separate backbone and head parameters
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Different learning rates for backbone and head
    param_groups = [
        {'params': backbone_params, 'lr': config.get('backbone_lr', 1e-5)},
        {'params': head_params, 'lr': config.get('head_lr', 1e-4)},
    ]

    optimizer = AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 1e-4)
    )

    return optimizer


def build_lr_scheduler(optimizer, config: Dict[str, Any]):
    """Build learning rate scheduler"""

    scheduler_type = config.get('scheduler_type', 'cosine')

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('epochs', 300),
            eta_min=config.get('min_lr', 1e-7)
        )
    elif scheduler_type == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [150, 250]),
            gamma=config.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class AverageMeter:
    """Compute and store the average and current value"""

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

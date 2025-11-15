"""
DSTARK Training Script

Train DSTARK tracker with DINOv3 backbone
Supports flexible template and search sizes
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Add dstark to path
sys.path.insert(0, str(Path(__file__).parent))

from dstark.models import DSTARKTracker
from dstark.data import TrackingDataset
from dstark.lib import TrackingLoss, build_optimizer, build_lr_scheduler, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Train DSTARK tracker')
    parser.add_argument('--config', type=str, default='configs/dstark_train.yaml',
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default='dinov3_vits16_pretrain.pth',
                        help='Pretrained DINOv3 weights')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use')

    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    loss_meter = AverageMeter()
    bbox_loss_meter = AverageMeter()
    giou_loss_meter = AverageMeter()
    conf_loss_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        template = batch['template'].to(device)
        search = batch['search'].to(device)
        search_bbox = batch['search_bbox'].to(device)

        # Forward pass
        predictions = model(template, search)

        # Compute loss
        targets = {'search_bbox': search_bbox}
        losses = criterion(predictions, targets)

        total_loss = losses['total_loss']

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update meters
        loss_meter.update(total_loss.item(), template.size(0))
        bbox_loss_meter.update(losses['bbox_loss'].item(), template.size(0))
        giou_loss_meter.update(losses['giou_loss'].item(), template.size(0))
        conf_loss_meter.update(losses['conf_loss'].item(), template.size(0))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'bbox': f'{bbox_loss_meter.avg:.4f}',
            'giou': f'{giou_loss_meter.avg:.4f}',
            'conf': f'{conf_loss_meter.avg:.4f}'
        })

    return {
        'total_loss': loss_meter.avg,
        'bbox_loss': bbox_loss_meter.avg,
        'giou_loss': giou_loss_meter.avg,
        'conf_loss': conf_loss_meter.avg,
    }


def save_checkpoint(state, output_dir, filename='checkpoint.pth'):
    """Save checkpoint"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    torch.save(state, filepath)
    print(f'Checkpoint saved to {filepath}')


def main():
    args = parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config if exists
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Build model
    print('Building DSTARK model...')
    model_config = {
        'backbone_config': {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'pretrained_path': args.pretrained if Path(args.pretrained).exists() else None
        },
        'hidden_dim': config.get('hidden_dim', 256),
    }

    model = DSTARKTracker(**model_config)
    model = model.to(device)

    # Build dataset
    print('Loading dataset...')
    train_dataset = TrackingDataset(
        root_dir=args.data_root,
        dataset_name='GOT10k',
        split='train',
        template_size=config.get('template_size', 128),
        search_size=config.get('search_size', 256),
        max_template_size=config.get('max_template_size', 256),
        max_search_size=config.get('max_search_size', 512),
        augment=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f'Training dataset: {len(train_dataset)} samples')

    # Build loss
    criterion = TrackingLoss(
        bbox_loss_weight=config.get('bbox_loss_weight', 5.0),
        giou_loss_weight=config.get('giou_loss_weight', 2.0),
        conf_loss_weight=config.get('conf_loss_weight', 1.0),
    )

    # Build optimizer
    optimizer_config = {
        'backbone_lr': config.get('backbone_lr', 1e-5),
        'head_lr': config.get('head_lr', 1e-4),
        'weight_decay': config.get('weight_decay', 1e-4),
    }
    optimizer = build_optimizer(model, optimizer_config)

    # Build scheduler
    scheduler_config = {
        'scheduler_type': config.get('scheduler_type', 'cosine'),
        'epochs': args.epochs,
        'min_lr': config.get('min_lr', 1e-7),
    }
    scheduler = build_lr_scheduler(optimizer, scheduler_config)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...')
    print('=' * 80)

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_stats = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Update scheduler
        scheduler.step()

        # Print stats
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Total Loss: {train_stats["total_loss"]:.4f}')
        print(f'  BBox Loss:  {train_stats["bbox_loss"]:.4f}')
        print(f'  GIoU Loss:  {train_stats["giou_loss"]:.4f}')
        print(f'  Conf Loss:  {train_stats["conf_loss"]:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        print('=' * 80)

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_stats': train_stats,
            }
            save_checkpoint(checkpoint, output_dir, f'checkpoint_epoch_{epoch + 1}.pth')

        # Save best model
        if train_stats['total_loss'] < best_loss:
            best_loss = train_stats['total_loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_stats': train_stats,
            }
            save_checkpoint(checkpoint, output_dir, 'best_model.pth')
            print(f'New best model saved! Loss: {best_loss:.4f}')

    print('\nTraining completed!')


if __name__ == '__main__':
    main()

"""
Configuration Management for DSTARK

This module provides type-safe configuration classes using dataclasses.
Benefits:
- Type safety at design time
- Default values
- Easy serialization/deserialization
- Self-documenting code
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class BackboneConfig:
    """
    Base configuration for backbone networks.

    This is an abstract configuration that can be extended for specific backbones.
    """
    type: str  # 'dinov3', 'resnet', 'swin', etc.
    pretrained_path: Optional[str] = None
    freeze_backbone: bool = False
    freeze_except_last_n: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type,
            'pretrained_path': self.pretrained_path,
            'freeze_backbone': self.freeze_backbone,
            'freeze_except_last_n': self.freeze_except_last_n
        }


@dataclass
class DINOv3Config(BackboneConfig):
    """
    Configuration for DINOv3 backbone.

    Provides sensible defaults for DINOv3-Small architecture.
    """
    type: str = 'dinov3'
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 384  # DINOv3 Small
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'in_chans': self.in_chans,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
        })
        return base_dict

    @staticmethod
    def small(pretrained_path: Optional[str] = None) -> 'DINOv3Config':
        """DINOv3 Small configuration."""
        return DINOv3Config(
            embed_dim=384,
            depth=12,
            num_heads=6,
            pretrained_path=pretrained_path
        )

    @staticmethod
    def base(pretrained_path: Optional[str] = None) -> 'DINOv3Config':
        """DINOv3 Base configuration."""
        return DINOv3Config(
            embed_dim=768,
            depth=12,
            num_heads=12,
            pretrained_path=pretrained_path
        )

    @staticmethod
    def large(pretrained_path: Optional[str] = None) -> 'DINOv3Config':
        """DINOv3 Large configuration."""
        return DINOv3Config(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            pretrained_path=pretrained_path
        )


@dataclass
class HeadConfig:
    """
    Configuration for tracking heads.
    """
    type: str  # 'correlation', 'transformer', etc.
    hidden_dim: int = 256

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type,
            'hidden_dim': self.hidden_dim
        }


@dataclass
class CorrelationHeadConfig(HeadConfig):
    """
    Configuration for correlation-based tracking head.
    """
    type: str = 'correlation'
    hidden_dim: int = 256
    use_batch_norm: bool = True
    correlation_type: str = 'dot_product'  # 'dot_product', 'cosine', 'learned'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'use_batch_norm': self.use_batch_norm,
            'correlation_type': self.correlation_type,
        })
        return base_dict


@dataclass
class DSTARKConfig:
    """
    Complete DSTARK model configuration.

    This class aggregates all component configurations and provides
    a single source of truth for model architecture.

    Design Pattern: Builder Pattern (via dataclass)
    """
    backbone: BackboneConfig
    head: HeadConfig

    # Model metadata
    model_name: str = 'DSTARK'
    version: str = '1.0.0'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'backbone': self.backbone.to_dict(),
            'head': self.head.to_dict(),
        }

    @staticmethod
    def default() -> 'DSTARKConfig':
        """Default DSTARK configuration with DINOv3-Small."""
        return DSTARKConfig(
            backbone=DINOv3Config.small(),
            head=CorrelationHeadConfig()
        )

    @staticmethod
    def from_pretrained(
        pretrained_path: str,
        backbone_size: str = 'small'
    ) -> 'DSTARKConfig':
        """
        Create configuration with pretrained weights.

        Args:
            pretrained_path: Path to pretrained backbone weights
            backbone_size: 'small', 'base', or 'large'

        Returns:
            DSTARKConfig with pretrained path set
        """
        if backbone_size == 'small':
            backbone = DINOv3Config.small(pretrained_path)
        elif backbone_size == 'base':
            backbone = DINOv3Config.base(pretrained_path)
        elif backbone_size == 'large':
            backbone = DINOv3Config.large(pretrained_path)
        else:
            raise ValueError(f"Unknown backbone size: {backbone_size}")

        return DSTARKConfig(
            backbone=backbone,
            head=CorrelationHeadConfig()
        )

    def validate(self) -> bool:
        """
        Validate configuration consistency.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check that head hidden_dim is compatible with backbone embed_dim
        if isinstance(self.backbone, DINOv3Config):
            if self.head.hidden_dim > self.backbone.embed_dim:
                raise ValueError(
                    f"Head hidden_dim ({self.head.hidden_dim}) should not exceed "
                    f"backbone embed_dim ({self.backbone.embed_dim})"
                )

        # Check pretrained path exists if provided
        if self.backbone.pretrained_path is not None:
            path = Path(self.backbone.pretrained_path)
            if not path.exists():
                raise ValueError(
                    f"Pretrained path does not exist: {self.backbone.pretrained_path}"
                )

        return True


@dataclass
class TrainingConfig:
    """
    Training configuration for DSTARK.

    Separates training hyperparameters from model architecture.
    Follows Single Responsibility Principle.
    """
    # Optimization
    epochs: int = 300
    batch_size: int = 16
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'exponential'
    warmup_epochs: int = 10
    min_lr: float = 1e-7

    # Loss weights
    bbox_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    conf_loss_weight: float = 1.0

    # Data
    template_size: int = 128
    search_size: int = 256
    max_template_size: int = 256
    max_search_size: int = 512

    # Logging & Checkpointing
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'backbone_lr': self.backbone_lr,
            'head_lr': self.head_lr,
            'weight_decay': self.weight_decay,
            'grad_clip_norm': self.grad_clip_norm,
            'scheduler_type': self.scheduler_type,
            'warmup_epochs': self.warmup_epochs,
            'min_lr': self.min_lr,
            'bbox_loss_weight': self.bbox_loss_weight,
            'giou_loss_weight': self.giou_loss_weight,
            'conf_loss_weight': self.conf_loss_weight,
            'template_size': self.template_size,
            'search_size': self.search_size,
            'max_template_size': self.max_template_size,
            'max_search_size': self.max_search_size,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'eval_interval': self.eval_interval,
        }

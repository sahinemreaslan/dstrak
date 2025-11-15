"""
Model Factory for DSTARK

Implements Factory Pattern for creating tracker components.
Provides a centralized, type-safe way to build models from configurations.

Design Patterns:
- Factory Pattern: Encapsulates object creation logic
- Builder Pattern: Constructs complex objects step by step
"""

from typing import Optional, Type
import torch.nn as nn

from .base_backbone import BaseBackbone, FlexibleBackbone
from .base_head import BaseTrackingHead, CorrelationBasedHead
from .dinov3_backbone import DINOv3Backbone
from .dstark_tracker import DSTARKTracker, CorrelationHead
from .config import (
    DSTARKConfig,
    BackboneConfig,
    DINOv3Config,
    HeadConfig,
    CorrelationHeadConfig
)


class BackboneFactory:
    """
    Factory for creating backbone networks.

    Centralizes backbone creation logic and makes it easy to add
    new backbone types without modifying existing code (Open/Closed Principle).
    """

    # Registry of available backbones
    _registry = {
        'dinov3': DINOv3Backbone,
    }

    @classmethod
    def register_backbone(
        cls,
        name: str,
        backbone_class: Type[BaseBackbone]
    ) -> None:
        """
        Register a new backbone type.

        Allows users to extend DSTARK with custom backbones.

        Args:
            name: Identifier for the backbone type
            backbone_class: Backbone class (must inherit from BaseBackbone)
        """
        if not issubclass(backbone_class, BaseBackbone):
            raise ValueError(
                f"Backbone class must inherit from BaseBackbone, "
                f"got {backbone_class}"
            )
        cls._registry[name] = backbone_class

    @classmethod
    def create_backbone(
        cls,
        config: BackboneConfig
    ) -> BaseBackbone:
        """
        Create a backbone from configuration.

        Args:
            config: Backbone configuration

        Returns:
            Initialized backbone

        Raises:
            ValueError: If backbone type is not recognized
        """
        backbone_type = config.type

        if backbone_type not in cls._registry:
            raise ValueError(
                f"Unknown backbone type: {backbone_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        backbone_class = cls._registry[backbone_type]

        # Build backbone based on config type
        if isinstance(config, DINOv3Config):
            backbone = backbone_class(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_chans=config.in_chans,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop_rate=config.drop_rate,
                attn_drop_rate=config.attn_drop_rate,
                pretrained_path=config.pretrained_path
            )
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        # Apply freezing if specified
        if config.freeze_backbone:
            backbone.freeze_backbone()
        elif config.freeze_except_last_n is not None:
            backbone.freeze_except_last_n_layers(config.freeze_except_last_n)

        return backbone


class HeadFactory:
    """
    Factory for creating tracking heads.

    Centralizes head creation logic for different tracking strategies.
    """

    # Registry of available heads
    _registry = {
        'correlation': CorrelationHead,
    }

    @classmethod
    def register_head(
        cls,
        name: str,
        head_class: Type[BaseTrackingHead]
    ) -> None:
        """
        Register a new head type.

        Args:
            name: Identifier for the head type
            head_class: Head class (must inherit from BaseTrackingHead)
        """
        if not issubclass(head_class, BaseTrackingHead):
            raise ValueError(
                f"Head class must inherit from BaseTrackingHead, "
                f"got {head_class}"
            )
        cls._registry[name] = head_class

    @classmethod
    def create_head(
        cls,
        config: HeadConfig,
        feat_dim: int
    ) -> BaseTrackingHead:
        """
        Create a tracking head from configuration.

        Args:
            config: Head configuration
            feat_dim: Feature dimension from backbone

        Returns:
            Initialized tracking head

        Raises:
            ValueError: If head type is not recognized
        """
        head_type = config.type

        if head_type not in cls._registry:
            raise ValueError(
                f"Unknown head type: {head_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        head_class = cls._registry[head_type]

        # Build head based on config type
        if isinstance(config, CorrelationHeadConfig):
            head = head_class(
                feat_dim=feat_dim,
                hidden_dim=config.hidden_dim
            )
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        return head


class ModelFactory:
    """
    Main factory for creating complete DSTARK trackers.

    This is the primary entry point for model creation.

    Example usage:
        >>> config = DSTARKConfig.from_pretrained('path/to/weights.pth')
        >>> tracker = ModelFactory.create_tracker(config)
    """

    @staticmethod
    def create_tracker(config: DSTARKConfig) -> DSTARKTracker:
        """
        Create a complete DSTARK tracker from configuration.

        This method orchestrates the creation of all components
        and assembles them into a working tracker.

        Args:
            config: Complete DSTARK configuration

        Returns:
            Initialized and configured tracker

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        config.validate()

        # Create backbone
        backbone = BackboneFactory.create_backbone(config.backbone)

        # Create head (needs to know backbone's feature dimension)
        head = HeadFactory.create_head(config.head, feat_dim=backbone.embed_dim)

        # Assemble tracker
        tracker = DSTARKTracker(backbone=backbone, head=head)

        return tracker

    @staticmethod
    def create_default_tracker(
        pretrained_path: Optional[str] = None,
        backbone_size: str = 'small',
        hidden_dim: int = 256
    ) -> DSTARKTracker:
        """
        Create a default DSTARK tracker with sensible defaults.

        Args:
            pretrained_path: Path to pretrained backbone weights
            backbone_size: 'small', 'base', or 'large'
            hidden_dim: Hidden dimension for tracking head

        Returns:
            Initialized tracker with default configuration
        """
        # Create default config
        if pretrained_path:
            config = DSTARKConfig.from_pretrained(pretrained_path, backbone_size)
        else:
            config = DSTARKConfig.default()

        # Update hidden_dim if specified
        config.head.hidden_dim = hidden_dim

        # Create tracker
        return ModelFactory.create_tracker(config)


# Convenience function for backward compatibility
def build_dstark_from_config(config: DSTARKConfig) -> DSTARKTracker:
    """
    Build DSTARK tracker from typed configuration.

    This is the recommended way to create models.

    Args:
        config: Type-safe DSTARK configuration

    Returns:
        Initialized tracker
    """
    return ModelFactory.create_tracker(config)

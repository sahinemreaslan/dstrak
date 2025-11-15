"""
DSTARK Models Module

This module provides a clean, well-architected visual tracking system
following SOLID principles and design patterns.

Main Components:
- Base classes for extensibility (base_backbone, base_head)
- Concrete implementations (DINOv3Backbone, CorrelationHead)
- Type-safe configuration (config)
- Factory pattern for model creation (factory)
- Main tracker (DSTARKTracker)

Recommended Usage:
    >>> from dstark.models import ModelFactory, DSTARKConfig
    >>> config = DSTARKConfig.from_pretrained('weights.pth', backbone_size='small')
    >>> tracker = ModelFactory.create_tracker(config)

Legacy Usage (still supported):
    >>> from dstark.models import DSTARKTracker, build_dstark
    >>> tracker = build_dstark({'hidden_dim': 256})
"""

# Base classes
from .base_backbone import BaseBackbone, FlexibleBackbone
from .base_head import BaseTrackingHead, CorrelationBasedHead, TransformerBasedHead

# Concrete implementations
from .dinov3_backbone import DINOv3Backbone
from .dstark_tracker import DSTARKTracker, CorrelationHead, build_dstark

# Configuration
from .config import (
    DSTARKConfig,
    DINOv3Config,
    BackboneConfig,
    HeadConfig,
    CorrelationHeadConfig,
    TrainingConfig,
)

# Factory
from .factory import (
    ModelFactory,
    BackboneFactory,
    HeadFactory,
    build_dstark_from_config,
)

__all__ = [
    # Base classes
    'BaseBackbone',
    'FlexibleBackbone',
    'BaseTrackingHead',
    'CorrelationBasedHead',
    'TransformerBasedHead',

    # Concrete implementations
    'DINOv3Backbone',
    'DSTARKTracker',
    'CorrelationHead',

    # Configuration
    'DSTARKConfig',
    'DINOv3Config',
    'BackboneConfig',
    'HeadConfig',
    'CorrelationHeadConfig',
    'TrainingConfig',

    # Factory
    'ModelFactory',
    'BackboneFactory',
    'HeadFactory',
    'build_dstark_from_config',

    # Legacy support
    'build_dstark',
]

# Version
__version__ = '1.0.0'

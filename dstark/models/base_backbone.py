"""
Abstract Base Classes for Backbone Networks

This module defines the interface for backbone networks used in DSTARK.
Following SOLID principles, particularly:
- Interface Segregation Principle (ISP)
- Dependency Inversion Principle (DIP)
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional


class BaseBackbone(ABC, nn.Module):
    """
    Abstract base class for all backbone networks.

    This interface ensures that all backbones provide consistent APIs,
    allowing for easy swapping of different architectures (DINOv3, ResNet, etc.)

    Design Pattern: Strategy Pattern
    - Different backbones can be plugged in without changing the tracker code
    """

    def __init__(self):
        super().__init__()
        self._embed_dim: Optional[int] = None
        self._patch_size: Optional[int] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            features: Extracted features [B, N, embed_dim]
                     where N depends on input size and patch_size
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Return the embedding dimension of the backbone."""
        pass

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Return the patch size used by the backbone."""
        pass

    @abstractmethod
    def load_pretrained(self, pretrained_path: str) -> None:
        """
        Load pretrained weights.

        Args:
            pretrained_path: Path to pretrained weights file
        """
        pass

    def get_feature_map_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate output feature map size for given input size.

        Args:
            input_size: (height, width) of input image

        Returns:
            (num_patches_h, num_patches_w): Feature map dimensions
        """
        h, w = input_size
        return h // self.patch_size, w // self.patch_size

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (useful for fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_except_last_n_layers(self, n: int) -> None:
        """
        Freeze all layers except the last n layers.

        Args:
            n: Number of last layers to keep trainable
        """
        raise NotImplementedError("Subclasses should implement this if needed")


class FlexibleBackbone(BaseBackbone):
    """
    Extended interface for backbones that support flexible input sizes.

    This is particularly useful for tracking where template and search regions
    can have different sizes.

    Examples: DINOv3 (with RoPE), Vision Transformers with interpolated pos encoding
    """

    @abstractmethod
    def interpolate_pos_encoding(
        self,
        x: torch.Tensor,
        w: int,
        h: int
    ) -> torch.Tensor:
        """
        Interpolate positional encodings for arbitrary input sizes.

        Args:
            x: Feature tensor
            w: Width of input image
            h: Height of input image

        Returns:
            Interpolated positional encodings
        """
        pass

    def supports_flexible_size(self) -> bool:
        """Check if this backbone supports flexible input sizes."""
        return True

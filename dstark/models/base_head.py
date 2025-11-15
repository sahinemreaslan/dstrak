"""
Abstract Base Classes for Tracking Heads

This module defines the interface for tracking heads used in DSTARK.
Following SOLID principles for extensibility and maintainability.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Dict


class BaseTrackingHead(ABC, nn.Module):
    """
    Abstract base class for tracking heads.

    This interface ensures that all tracking heads provide consistent APIs,
    allowing for easy experimentation with different tracking strategies.

    Design Pattern: Strategy Pattern
    - Different tracking heads (correlation-based, transformer-based, etc.)
      can be used interchangeably
    """

    def __init__(self, feat_dim: int, hidden_dim: int):
        """
        Initialize tracking head.

        Args:
            feat_dim: Dimension of input features from backbone
            hidden_dim: Hidden dimension for internal processing
        """
        super().__init__()
        self._feat_dim = feat_dim
        self._hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        template_feat: torch.Tensor,
        search_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bounding box and confidence from template and search features.

        Args:
            template_feat: Template features [B, N_template, feat_dim]
            search_feat: Search features [B, N_search, feat_dim]

        Returns:
            pred_boxes: Predicted bounding boxes [B, 4, H, W]
            pred_conf: Confidence scores [B, 1, H, W]
        """
        pass

    @property
    def feat_dim(self) -> int:
        """Return the input feature dimension."""
        return self._feat_dim

    @property
    def hidden_dim(self) -> int:
        """Return the hidden dimension."""
        return self._hidden_dim

    def get_output_size(self, search_feat: torch.Tensor) -> Tuple[int, int]:
        """
        Calculate output spatial dimensions based on search features.

        Args:
            search_feat: Search features [B, N_search, feat_dim]

        Returns:
            (height, width): Output spatial dimensions
        """
        N_search = search_feat.shape[1] - 1  # Exclude CLS token
        h = w = int(N_search ** 0.5)
        return h, w


class CorrelationBasedHead(BaseTrackingHead):
    """
    Base class for correlation-based tracking heads.

    These heads compute similarity between template and search features
    to localize the target object.
    """

    @abstractmethod
    def compute_correlation(
        self,
        template_feat: torch.Tensor,
        search_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation map between template and search features.

        Args:
            template_feat: Template features [B, N_template, feat_dim]
            search_feat: Search features [B, N_search, feat_dim]

        Returns:
            correlation_map: Correlation map [B, 1, H, W]
        """
        pass


class TransformerBasedHead(BaseTrackingHead):
    """
    Base class for transformer-based tracking heads.

    These heads use cross-attention mechanisms to fuse template and search features.
    """

    @abstractmethod
    def cross_attention(
        self,
        template_feat: torch.Tensor,
        search_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention between template and search features.

        Args:
            template_feat: Template features [B, N_template, feat_dim]
            search_feat: Search features [B, N_search, feat_dim]

        Returns:
            fused_features: Fused features [B, N_search, feat_dim]
        """
        pass

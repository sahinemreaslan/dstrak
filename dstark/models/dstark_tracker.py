"""
DSTARK Tracker Head

Main tracking model that combines DINOv3 backbone with correlation-based tracking.
Supports flexible template and search sizes without hardcoded constraints.

Design Patterns:
- Strategy Pattern: Interchangeable backbone and head components
- Dependency Injection: Components injected rather than created internally
- Facade Pattern: Simple interface hiding complex tracking logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base_head import CorrelationBasedHead
from .base_backbone import BaseBackbone


class CorrelationHead(CorrelationBasedHead):
    """
    Correlation-based tracking head implementation.

    Implements the CorrelationBasedHead interface with specific
    dot-product based correlation computation.

    Architecture:
    1. Project template & search features to hidden_dim
    2. Compute dot-product correlation
    3. Weight search features by correlation
    4. Predict bbox and confidence via CNN heads
    """

    def __init__(self, feat_dim: int = 384, hidden_dim: int = 256):
        """
        Initialize correlation head.

        Args:
            feat_dim: Input feature dimension from backbone
            hidden_dim: Hidden dimension for processing
        """
        super().__init__(feat_dim, hidden_dim)

        # Feature adjustment layers
        self.template_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.search_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Correlation processing
        self.correlation_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Box regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1)  # x, y, w, h
        )

        # Confidence head
        self.conf_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # confidence score
        )

    def compute_correlation(
        self,
        template_feat: torch.Tensor,
        search_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation map between template and search features.

        Uses dot-product similarity with L2 normalization.

        Args:
            template_feat: Template features [B, N_template, feat_dim]
            search_feat: Search features [B, N_search, feat_dim]

        Returns:
            correlation_map: Correlation map [B, 1, H, W]
        """
        # Remove CLS tokens
        template_tokens = template_feat[:, 1:, :]
        search_tokens = search_feat[:, 1:, :]

        # Normalize for stable correlation
        template_tokens = F.normalize(template_tokens, dim=-1)
        search_tokens = F.normalize(search_tokens, dim=-1)

        # Average template tokens to create a prototype
        template_proto = template_tokens.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Compute similarity
        correlation = torch.matmul(search_tokens, template_proto.transpose(1, 2))  # [B, N_s, 1]
        correlation = correlation.squeeze(-1)  # [B, N_s]

        # Reshape to spatial
        N_s = search_tokens.shape[1]
        h_s = w_s = int(N_s ** 0.5)
        correlation_map = correlation.view(-1, 1, h_s, w_s)

        return correlation_map

    def forward(
        self,
        template_feat: torch.Tensor,
        search_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bounding box and confidence from features.

        Args:
            template_feat: Template features [B, N_template, feat_dim]
            search_feat: Search features [B, N_search, feat_dim]

        Returns:
            pred_boxes: Predicted bounding boxes [B, 4, H, W]
            pred_conf: Confidence scores [B, 1, H, W]
        """
        # Project features to hidden dimension
        template_feat_proj = self.template_proj(template_feat)  # [B, N_t, hidden_dim]
        search_feat_proj = self.search_proj(search_feat)  # [B, N_s, hidden_dim]

        # Compute correlation map
        correlation_map = self.compute_correlation(template_feat_proj, search_feat_proj)

        # Get search tokens without CLS
        search_tokens = search_feat_proj[:, 1:, :]  # [B, N_s, hidden_dim]
        N_s = search_tokens.shape[1]
        h_s = w_s = int(N_s ** 0.5)

        # Expand search features to 2D for conv processing
        B = search_tokens.shape[0]
        search_2d = search_tokens.transpose(1, 2).view(B, -1, h_s, w_s)

        # Weighted search features
        weighted_search = search_2d * correlation_map

        # Process correlation
        corr_feat = self.correlation_head(weighted_search)

        # Predict bbox and confidence
        pred_boxes = self.bbox_head(corr_feat)  # [B, 4, H, W]
        pred_conf = self.conf_head(corr_feat)  # [B, 1, H, W]

        return pred_boxes, pred_conf


class DSTARKTracker(nn.Module):
    """
    Complete DSTARK Tracker.

    Combines a flexible backbone with a tracking head for visual object tracking.

    Key improvements over standard STARK:
    - No fixed template/search size (uses flexible position encoding)
    - Better feature extraction with DINOv3 self-supervised pretraining
    - Handles occlusion better with rich semantic features
    - Can track objects at varying scales

    Design Patterns:
    - Dependency Injection: Backbone and head are injected
    - Facade: Provides simple interface for complex tracking
    - Strategy: Components can be swapped without changing interface
    """

    def __init__(
        self,
        backbone: BaseBackbone,
        head: CorrelationHead,
    ):
        """
        Initialize DSTARK tracker with dependency injection.

        Args:
            backbone: Feature extraction backbone (e.g., DINOv3Backbone)
            head: Tracking head (e.g., CorrelationHead)
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.feat_dim = backbone.embed_dim

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or offline tracking.

        Args:
            template: Template image [B, 3, H_t, W_t] - flexible size!
            search: Search image [B, 3, H_s, W_s] - flexible size!
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - pred_boxes: Predicted bounding boxes [B, 4, H, W]
                - pred_conf: Confidence scores [B, 1, H, W]
                - template_feat: (optional) Template features
                - search_feat: (optional) Search features
        """
        # Extract features with flexible sizes
        template_feat = self.backbone(template)  # [B, N_t+1, C]
        search_feat = self.backbone(search)  # [B, N_s+1, C]

        # Tracking prediction
        pred_boxes, pred_conf = self.head(template_feat, search_feat)

        output = {
            'pred_boxes': pred_boxes,
            'pred_conf': pred_conf,
        }

        if return_features:
            output['template_feat'] = template_feat
            output['search_feat'] = search_feat

        return output

    def template(self, z: torch.Tensor) -> torch.Tensor:
        """
        Extract template features (for online tracking).

        This should be called once per tracked object to extract
        and cache template features. Subsequently use track() for
        efficient inference.

        Args:
            z: Template image [B, 3, H, W]

        Returns:
            Template features [B, N+1, C]
        """
        return self.backbone(z)

    def track(
        self,
        template_feat: torch.Tensor,
        search: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Track with pre-extracted template features (online tracking).

        More efficient than forward() as template features are reused.

        Args:
            template_feat: Pre-extracted template features [B, N_t+1, C]
            search: Search image [B, 3, H, W]

        Returns:
            Dictionary containing:
                - pred_boxes: Predicted bounding boxes [B, 4, H, W]
                - pred_conf: Confidence scores [B, 1, H, W]
        """
        search_feat = self.backbone(search)
        pred_boxes, pred_conf = self.head(template_feat, search_feat)

        return {
            'pred_boxes': pred_boxes,
            'pred_conf': pred_conf,
        }

    def get_box_and_score(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract final box and confidence from predictions.

        Selects the bounding box at the location with highest confidence.

        Args:
            predictions: Model output dictionary containing pred_boxes and pred_conf

        Returns:
            final_boxes: Bounding boxes [B, 4] (cx, cy, w, h)
            scores: Confidence scores [B]
        """
        pred_boxes = predictions['pred_boxes']  # [B, 4, H, W]
        pred_conf = predictions['pred_conf']  # [B, 1, H, W]

        B, _, H, W = pred_boxes.shape

        # Find max confidence location
        conf_flat = pred_conf.view(B, -1)
        max_idx = conf_flat.argmax(dim=1)

        # Get box at max confidence location
        boxes_flat = pred_boxes.view(B, 4, -1)
        final_boxes = []

        for b in range(B):
            idx = max_idx[b]
            box = boxes_flat[b, :, idx]  # [4]
            final_boxes.append(box)

        final_boxes = torch.stack(final_boxes)  # [B, 4]

        # Get confidence scores
        scores = conf_flat.gather(1, max_idx.unsqueeze(1)).squeeze(1)  # [B]
        scores = torch.sigmoid(scores)

        return final_boxes, scores


def build_dstark(config: Dict) -> DSTARKTracker:
    """
    Build DSTARK model from dictionary config (legacy support).

    DEPRECATED: Use ModelFactory.create_tracker() for better type safety.

    Args:
        config: Dictionary containing model configuration

    Returns:
        Initialized DSTARKTracker
    """
    from .dinov3_backbone import DINOv3Backbone

    # Extract backbone config
    backbone_config = config.get('backbone', {})
    if 'pretrained_path' not in backbone_config:
        backbone_config['pretrained_path'] = config.get('pretrained_path', None)

    # Default to DINOv3 Small if no config provided
    if not backbone_config:
        backbone_config = {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
        }

    # Build components
    backbone = DINOv3Backbone(**backbone_config)
    feat_dim = backbone_config.get('embed_dim', 384)
    hidden_dim = config.get('hidden_dim', 256)
    head = CorrelationHead(feat_dim=feat_dim, hidden_dim=hidden_dim)

    # Build tracker
    model = DSTARKTracker(backbone=backbone, head=head)

    return model

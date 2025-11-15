"""
DSTARK Tracker Head

Main tracking model that combines DINOv3 backbone with correlation-based tracking.
Supports flexible template and search sizes without hardcoded constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CorrelationHead(nn.Module):
    """Correlation-based tracking head"""

    def __init__(self, feat_dim=384, hidden_dim=256):
        super().__init__()

        self.feat_dim = feat_dim

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

    def forward(self, template_feat, search_feat):
        """
        Args:
            template_feat: [B, N_template, C] - Template features (flexible size)
            search_feat: [B, N_search, C] - Search features (flexible size)

        Returns:
            pred_boxes: Predicted bounding boxes
            pred_conf: Confidence scores
        """
        B = template_feat.shape[0]

        # Project features
        template_feat = self.template_proj(template_feat)  # [B, N_t, hidden_dim]
        search_feat = self.search_proj(search_feat)  # [B, N_s, hidden_dim]

        # Compute correlation (using dot product)
        # Remove CLS token for spatial correlation
        template_tokens = template_feat[:, 1:, :]  # Remove CLS
        search_tokens = search_feat[:, 1:, :]  # Remove CLS

        # Calculate spatial dimensions dynamically
        N_t = template_tokens.shape[1]
        N_s = search_tokens.shape[1]
        h_s = w_s = int(N_s ** 0.5)  # Assume square for simplicity

        # Normalize for stable correlation
        template_tokens = F.normalize(template_tokens, dim=-1)
        search_tokens = F.normalize(search_tokens, dim=-1)

        # Correlation via matrix multiplication
        # Average template tokens to create a prototype
        template_proto = template_tokens.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Compute similarity
        correlation = torch.matmul(search_tokens, template_proto.transpose(1, 2))  # [B, N_s, 1]
        correlation = correlation.squeeze(-1)  # [B, N_s]

        # Reshape to spatial
        correlation_map = correlation.view(B, 1, h_s, w_s)

        # Expand search features to 2D for conv processing
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
    Complete DSTARK Tracker

    Key improvements over standard STARK:
    - No fixed template/search size (uses DINOv3 RoPE)
    - Better feature extraction with DINOv3
    - Handles occlusion better with rich features
    - Can track varying object sizes
    """

    def __init__(
        self,
        backbone_config: Optional[Dict] = None,
        hidden_dim: int = 256,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from .dinov3_backbone import DINOv3Backbone

        # Default backbone config for DINOv3 Small
        if backbone_config is None:
            backbone_config = {
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
                'pretrained_path': pretrained_path
            }

        # Backbone
        self.backbone = DINOv3Backbone(**backbone_config)
        feat_dim = backbone_config.get('embed_dim', 384)

        # Tracking head
        self.head = CorrelationHead(feat_dim=feat_dim, hidden_dim=hidden_dim)

        self.feat_dim = feat_dim

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            template: Template image [B, 3, H_t, W_t] - flexible size!
            search: Search image [B, 3, H_s, W_s] - flexible size!
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing predictions
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
        """Extract template features (for online tracking)"""
        return self.backbone(z)

    def track(
        self,
        template_feat: torch.Tensor,
        search: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Track with pre-extracted template features

        Args:
            template_feat: Pre-extracted template features [B, N_t+1, C]
            search: Search image [B, 3, H, W]

        Returns:
            Tracking predictions
        """
        search_feat = self.backbone(search)
        pred_boxes, pred_conf = self.head(template_feat, search_feat)

        return {
            'pred_boxes': pred_boxes,
            'pred_conf': pred_conf,
        }

    def get_box_and_score(self, predictions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Extract final box and confidence from predictions

        Args:
            predictions: Model output dictionary

        Returns:
            bbox: [x, y, w, h] in original image coordinates
            score: Confidence score
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
    """Build DSTARK model from config"""

    model = DSTARKTracker(
        backbone_config=config.get('backbone', None),
        hidden_dim=config.get('hidden_dim', 256),
        pretrained_path=config.get('pretrained_path', None)
    )

    return model

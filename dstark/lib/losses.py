"""
Loss functions for DSTARK training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TrackingLoss(nn.Module):
    """
    Combined loss for tracking:
    - Bounding box regression loss (L1 + GIoU)
    - Confidence loss (focal loss)
    """

    def __init__(
        self,
        bbox_loss_weight: float = 5.0,
        giou_loss_weight: float = 2.0,
        conf_loss_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.bbox_loss_weight = bbox_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def compute_giou_loss(self, pred_boxes, target_boxes):
        """
        Compute GIoU loss

        Args:
            pred_boxes: [B, 4, H, W] predicted boxes
            target_boxes: [B, 4] target boxes in cxcywh format

        Returns:
            giou_loss: Scalar loss
        """
        from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

        B, _, H, W = pred_boxes.shape

        # Get center prediction
        center_pred = pred_boxes[:, :, H // 2, W // 2]  # [B, 4]

        # Convert to xyxy
        pred_boxes_xyxy = box_cxcywh_to_xyxy(center_pred)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

        # Compute GIoU
        giou = generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
        giou_loss = 1 - giou.diagonal().mean()

        return giou_loss

    def compute_bbox_loss(self, pred_boxes, target_boxes):
        """
        Compute L1 bbox loss

        Args:
            pred_boxes: [B, 4, H, W]
            target_boxes: [B, 4]

        Returns:
            l1_loss: Scalar loss
        """
        B, _, H, W = pred_boxes.shape

        # Get center prediction
        center_pred = pred_boxes[:, :, H // 2, W // 2]  # [B, 4]

        # L1 loss
        l1_loss = F.l1_loss(center_pred, target_boxes)

        return l1_loss

    def compute_conf_loss(self, pred_conf, target_boxes, pred_boxes):
        """
        Compute focal loss for confidence

        Args:
            pred_conf: [B, 1, H, W]
            target_boxes: [B, 4]
            pred_boxes: [B, 4, H, W]

        Returns:
            conf_loss: Scalar loss
        """
        B, _, H, W = pred_conf.shape

        # Create target confidence map (Gaussian around target center)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=pred_conf.device),
            torch.arange(W, device=pred_conf.device),
            indexing='ij'
        )

        # Target center in feature map coordinates
        target_conf = torch.zeros_like(pred_conf)

        for b in range(B):
            cx, cy = H // 2, W // 2  # Assume center for simplicity
            sigma = min(H, W) / 4

            # Gaussian
            gauss = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
            target_conf[b, 0] = gauss

        # Focal loss
        pred_conf_sigmoid = torch.sigmoid(pred_conf)

        pt = pred_conf_sigmoid * target_conf + (1 - pred_conf_sigmoid) * (1 - target_conf)
        focal_weight = (1 - pt) ** self.focal_gamma

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_conf, target_conf, reduction='none'
        )

        conf_loss = (focal_weight * bce_loss).mean()

        return conf_loss

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of losses
        """
        pred_boxes = predictions['pred_boxes']
        pred_conf = predictions['pred_conf']
        target_boxes = targets['search_bbox']

        # Compute losses
        bbox_loss = self.compute_bbox_loss(pred_boxes, target_boxes)
        giou_loss = self.compute_giou_loss(pred_boxes, target_boxes)
        conf_loss = self.compute_conf_loss(pred_conf, target_boxes, pred_boxes)

        # Total loss
        total_loss = (
            self.bbox_loss_weight * bbox_loss +
            self.giou_loss_weight * giou_loss +
            self.conf_loss_weight * conf_loss
        )

        return {
            'total_loss': total_loss,
            'bbox_loss': bbox_loss,
            'giou_loss': giou_loss,
            'conf_loss': conf_loss,
        }

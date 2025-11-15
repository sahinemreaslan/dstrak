"""
Bounding box operations
"""

import torch
from typing import Tuple


def box_cxcywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bbox from (center_x, center_y, width, height) to (x1, y1, x2, y2)

    Args:
        bbox: [..., 4] tensor in cxcywh format

    Returns:
        bbox in xyxy format
    """
    cx, cy, w, h = bbox.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bbox from (x1, y1, x2, y2) to (center_x, center_y, width, height)

    Args:
        bbox: [..., 4] tensor in xyxy format

    Returns:
        bbox in cxcywh format
    """
    x1, y1, x2, y2 = bbox.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute IoU between two sets of boxes

    Args:
        boxes1: [N, 4] in xyxy format
        boxes2: [M, 4] in xyxy format

    Returns:
        iou: [N, M] IoU matrix
        union: [N, M] Union area matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)

    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (GIoU) between two sets of boxes

    Args:
        boxes1: [N, 4] in xyxy format
        boxes2: [M, 4] in xyxy format

    Returns:
        giou: [N, M] GIoU matrix
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / (area + 1e-6)

    return giou

"""
Miscellaneous utility functions
"""

import torch
from typing import List


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """
    Create a nested tensor from a list of tensors with different sizes

    Args:
        tensor_list: List of tensors

    Returns:
        Batched tensor with padding
    """
    if tensor_list[0].ndim == 3:
        # Image tensors [C, H, W]
        max_size = [max([img.shape[i] for img in tensor_list]) for i in range(len(tensor_list[0].shape))]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3D tensors supported")

    return tensor, mask

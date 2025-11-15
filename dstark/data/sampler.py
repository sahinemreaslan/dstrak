"""
Custom sampler for tracking datasets
"""

import torch
from torch.utils.data import Sampler
import math
from typing import Iterator


class TrackingSampler(Sampler):
    """
    Sampler for tracking dataset that ensures diverse sampling
    """

    def __init__(self, dataset, batch_size: int = 16, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            indices = torch.randperm(self.num_samples).tolist()
        else:
            indices = list(range(self.num_samples))

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

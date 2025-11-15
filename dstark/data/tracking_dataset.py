"""
Tracking dataset with flexible template and search sizes

Key difference from standard trackers:
- No fixed size constraints
- Dynamic sizing based on object scale
- Better handling of occlusion scenarios
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import random


class TrackingDataset(Dataset):
    """
    Generic tracking dataset

    Supports various tracking benchmarks (GOT-10k, LaSOT, TrackingNet, etc.)
    with flexible template and search region sizes
    """

    def __init__(
        self,
        root_dir: str,
        dataset_name: str = 'GOT10k',
        split: str = 'train',
        template_size: int = 128,  # Base size, can be dynamic
        search_size: int = 256,    # Base size, can be dynamic
        max_template_size: int = 256,  # Maximum template size
        max_search_size: int = 512,    # Maximum search size
        template_factor: float = 2.0,
        search_factor: float = 4.0,
        scale_jitter: bool = True,
        center_jitter: bool = True,
        augment: bool = True,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split

        # Flexible sizing parameters
        self.template_size = template_size
        self.search_size = search_size
        self.max_template_size = max_template_size
        self.max_search_size = max_search_size
        self.template_factor = template_factor
        self.search_factor = search_factor

        # Augmentation flags
        self.scale_jitter = scale_jitter
        self.center_jitter = center_jitter
        self.augment = augment

        # Load dataset
        self.videos = self._load_videos()

    def _load_videos(self) -> List[Dict]:
        """Load video sequences from dataset"""
        videos = []

        if self.dataset_name == 'GOT10k':
            split_dir = self.root_dir / self.split
            if split_dir.exists():
                for video_dir in sorted(split_dir.iterdir()):
                    if video_dir.is_dir():
                        groundtruth_file = video_dir / 'groundtruth.txt'
                        if groundtruth_file.exists():
                            videos.append({
                                'name': video_dir.name,
                                'path': video_dir,
                                'gt_file': groundtruth_file
                            })

        print(f"Loaded {len(videos)} videos from {self.dataset_name} {self.split}")
        return videos

    def __len__(self) -> int:
        return len(self.videos) * 100 if self.split == 'train' else len(self.videos)

    def _read_anno(self, gt_file: Path) -> np.ndarray:
        """Read ground truth annotations"""
        with open(gt_file, 'r') as f:
            lines = f.readlines()

        anno = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                x, y, w, h = map(float, parts[:4])
                anno.append([x, y, w, h])

        return np.array(anno)

    def _crop_image(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        crop_size: int,
        factor: float,
        jitter_scale: bool = True,
        jitter_center: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop image region around bbox

        Args:
            image: Input image [H, W, C]
            bbox: Bounding box [x, y, w, h]
            crop_size: Target crop size
            factor: Crop factor relative to bbox
            jitter_scale: Apply scale jittering
            jitter_center: Apply center jittering

        Returns:
            crop: Cropped and resized image
            bbox_crop: Bbox in crop coordinates
        """
        x, y, w, h = bbox

        # Calculate crop size based on object size and factor
        crop_w = w * factor
        crop_h = h * factor

        # Scale jittering (helps with varying object sizes)
        if jitter_scale and self.split == 'train':
            scale_factor = random.uniform(0.8, 1.2)
            crop_w *= scale_factor
            crop_h *= scale_factor

        # Center jittering
        cx = x + w / 2
        cy = y + h / 2

        if jitter_center and self.split == 'train':
            cx += random.uniform(-0.2, 0.2) * w
            cy += random.uniform(-0.2, 0.2) * h

        # Calculate crop region
        x1 = max(0, int(cx - crop_w / 2))
        y1 = max(0, int(cy - crop_h / 2))
        x2 = min(image.shape[1], int(cx + crop_w / 2))
        y2 = min(image.shape[0], int(cy + crop_h / 2))

        # Crop image
        crop = image[y1:y2, x1:x2]

        # Resize to target size
        crop = cv2.resize(crop, (crop_size, crop_size))

        # Update bbox coordinates
        bbox_crop = np.array([
            (x - x1) / (x2 - x1) * crop_size,
            (y - y1) / (y2 - y1) * crop_size,
            w / (x2 - x1) * crop_size,
            h / (y2 - y1) * crop_size
        ])

        return crop, bbox_crop

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply color augmentation"""
        if not self.augment or self.split != 'train':
            return image

        # Random brightness
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            image = np.clip(image * alpha, 0, 255).astype(np.uint8)

        # Random contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * alpha + mean, 0, 255).astype(np.uint8)

        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training/testing sample

        Returns:
            Dictionary containing:
            - template: Template image
            - search: Search image
            - bbox: Target bbox in search region
            - template_bbox: Bbox in template
        """
        if self.split == 'train':
            video_idx = idx % len(self.videos)
        else:
            video_idx = idx

        video = self.videos[video_idx]

        # Load annotations
        anno = self._read_anno(video['gt_file'])

        # Get image list
        image_files = sorted((video['path']).glob('*.jpg'))
        if len(image_files) == 0:
            image_files = sorted((video['path']).glob('*.png'))

        # Sample template and search frames
        if self.split == 'train':
            # Random sampling for training
            template_idx = random.randint(0, min(len(anno) - 1, len(image_files) - 1))
            search_idx = random.randint(
                template_idx,
                min(template_idx + 100, len(anno) - 1, len(image_files) - 1)
            )
        else:
            # First frame as template for testing
            template_idx = 0
            search_idx = min(1, len(anno) - 1, len(image_files) - 1)

        # Load images
        template_img = cv2.imread(str(image_files[template_idx]))
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

        search_img = cv2.imread(str(image_files[search_idx]))
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)

        # Get bboxes
        template_bbox = anno[template_idx]
        search_bbox = anno[search_idx]

        # Dynamic sizing based on object scale
        obj_area = template_bbox[2] * template_bbox[3]
        img_area = template_img.shape[0] * template_img.shape[1]
        scale = np.sqrt(obj_area / img_area)

        # Adjust sizes based on object scale
        if scale > 0.5:  # Large object
            template_size = min(self.max_template_size, int(self.template_size * 1.5))
            search_size = min(self.max_search_size, int(self.search_size * 1.5))
        elif scale < 0.1:  # Small object
            template_size = self.template_size
            search_size = self.search_size
        else:  # Medium object
            template_size = int(self.template_size * 1.2)
            search_size = int(self.search_size * 1.2)

        # Crop template and search regions
        template_crop, template_bbox_crop = self._crop_image(
            template_img, template_bbox, template_size,
            self.template_factor, self.scale_jitter, False
        )

        search_crop, search_bbox_crop = self._crop_image(
            search_img, search_bbox, search_size,
            self.search_factor, self.scale_jitter, self.center_jitter
        )

        # Augmentation
        template_crop = self._augment_image(template_crop)
        search_crop = self._augment_image(search_crop)

        # Convert to tensors
        template_tensor = torch.from_numpy(template_crop).permute(2, 0, 1).float() / 255.0
        search_tensor = torch.from_numpy(search_crop).permute(2, 0, 1).float() / 255.0

        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        template_tensor = (template_tensor - mean) / std
        search_tensor = (search_tensor - mean) / std

        return {
            'template': template_tensor,
            'search': search_tensor,
            'search_bbox': torch.from_numpy(search_bbox_crop).float(),
            'template_bbox': torch.from_numpy(template_bbox_crop).float(),
        }

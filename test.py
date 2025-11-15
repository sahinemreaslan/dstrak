"""
DSTARK Testing Script

Test DSTARK tracker on various benchmarks
Supports flexible template and search sizes
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import json

# Add dstark to path
sys.path.insert(0, str(Path(__file__).parent))

from dstark.models import DSTARKTracker


class Tracker:
    """Online tracker for inference"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.template_feat = None

        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def initialize(self, image, bbox):
        """
        Initialize tracker with first frame

        Args:
            image: RGB image [H, W, 3]
            bbox: Initial bbox [x, y, w, h]
        """
        # Crop template
        template = self._crop_image(image, bbox, size=128, factor=2.0)

        # Convert to tensor
        template = torch.from_numpy(template).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        template = (template.to(self.device) - self.mean) / self.std

        # Extract template features
        with torch.no_grad():
            self.template_feat = self.model.template(template)

        self.last_bbox = bbox

    def track(self, image):
        """
        Track in new frame

        Args:
            image: RGB image [H, W, 3]

        Returns:
            bbox: Predicted bbox [x, y, w, h]
            score: Confidence score
        """
        # Crop search region around last bbox
        search = self._crop_image(image, self.last_bbox, size=256, factor=4.0)

        # Convert to tensor
        search = torch.from_numpy(search).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        search = (search.to(self.device) - self.mean) / self.std

        # Track
        with torch.no_grad():
            predictions = self.model.track(self.template_feat, search)
            bbox, score = self.model.get_box_and_score(predictions)

        bbox = bbox[0].cpu().numpy()
        score = score[0].cpu().item()

        # Convert bbox from search region to image coordinates
        x, y, w, h = self.last_bbox
        search_cx = x + w / 2
        search_cy = y + h / 2
        search_size = max(w, h) * 4.0

        # Map predicted bbox to image coordinates
        pred_cx = search_cx + (bbox[0] - 128) * search_size / 256
        pred_cy = search_cy + (bbox[1] - 128) * search_size / 256
        pred_w = bbox[2] * search_size / 256
        pred_h = bbox[3] * search_size / 256

        final_bbox = np.array([
            pred_cx - pred_w / 2,
            pred_cy - pred_h / 2,
            pred_w,
            pred_h
        ])

        # Update last bbox
        self.last_bbox = final_bbox

        return final_bbox, score

    def _crop_image(self, image, bbox, size, factor):
        """Crop image region"""
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

        crop_size = max(w, h) * factor

        x1 = max(0, int(cx - crop_size / 2))
        y1 = max(0, int(cy - crop_size / 2))
        x2 = min(image.shape[1], int(cx + crop_size / 2))
        y2 = min(image.shape[0], int(cy + crop_size / 2))

        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (size, size))

        return crop


def load_model(checkpoint_path, device):
    """Load trained model"""
    print(f'Loading model from {checkpoint_path}')

    # Build model
    model = DSTARKTracker(
        backbone_config={
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
        },
        hidden_dim=256,
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def test_got10k(model, data_root, output_dir, device):
    """Test on GOT-10k benchmark"""
    print('Testing on GOT-10k...')

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = data_root / 'test'
    if not test_dir.exists():
        print(f'GOT-10k test directory not found: {test_dir}')
        return

    video_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    results = {}
    times = []

    for video_dir in tqdm(video_dirs, desc='Testing'):
        video_name = video_dir.name

        # Load groundtruth for first frame
        gt_file = video_dir / 'groundtruth.txt'
        if not gt_file.exists():
            continue

        with open(gt_file, 'r') as f:
            first_line = f.readline().strip()
            init_bbox = list(map(float, first_line.split(',')))

        # Get image files
        image_files = sorted(video_dir.glob('*.jpg'))
        if len(image_files) == 0:
            image_files = sorted(video_dir.glob('*.png'))

        # Initialize tracker
        first_image = cv2.imread(str(image_files[0]))
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)

        tracker = Tracker(model, device)
        tracker.initialize(first_image, init_bbox)

        # Track through video
        pred_bboxes = [init_bbox]
        pred_times = []

        for img_file in image_files[1:]:
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            import time
            start_time = time.time()
            bbox, score = tracker.track(image)
            end_time = time.time()

            pred_bboxes.append(bbox.tolist())
            pred_times.append(end_time - start_time)

        # Save results
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        with open(video_output_dir / f'{video_name}_001.txt', 'w') as f:
            for bbox in pred_bboxes:
                f.write(','.join(map(str, bbox)) + '\n')

        with open(video_output_dir / f'{video_name}_time.txt', 'w') as f:
            f.write(str(np.mean(pred_times)))

        times.extend(pred_times)

    avg_fps = 1.0 / np.mean(times) if times else 0
    print(f'\nAverage FPS: {avg_fps:.2f}')

    return results


def test_otb(model, data_root, output_dir, device):
    """Test on OTB benchmark"""
    print('Testing on OTB...')

    # Similar structure to GOT-10k testing
    print('OTB testing not implemented yet')


def parse_args():
    parser = argparse.ArgumentParser(description='Test DSTARK tracker')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data/GOT10k',
                        help='Root directory for test dataset')
    parser.add_argument('--benchmark', type=str, default='GOT10k',
                        choices=['GOT10k', 'OTB', 'LaSOT'],
                        help='Benchmark to test on')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize tracking results')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = load_model(args.checkpoint, device)

    # Test on benchmark
    if args.benchmark == 'GOT10k':
        results = test_got10k(model, args.data_root, args.output_dir, device)
    elif args.benchmark == 'OTB':
        results = test_otb(model, args.data_root, args.output_dir, device)
    else:
        print(f'Benchmark {args.benchmark} not implemented yet')

    print(f'\nResults saved to {args.output_dir}')


if __name__ == '__main__':
    main()

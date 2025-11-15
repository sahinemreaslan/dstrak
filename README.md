# DSTARK - DINOv3-based STARK Tracker

DSTARK is a visual object tracker that combines the STARK tracking architecture with DINOv3 backbone, enabling flexible template and search region sizes without hardcoded constraints.

## Key Features

🚀 **Flexible Input Sizes**: Unlike traditional trackers with fixed 128×128 template and 256×256 search sizes, DSTARK supports dynamic sizing thanks to DINOv3's RoPE (Rotary Position Embeddings).

🎯 **Better Occlusion Handling**: DINOv3's rich feature extraction helps maintain target identity even during occlusions.

📈 **Adaptive Scaling**: Automatically adjusts template and search region sizes based on object scale.

💪 **Robust Tracking**: Handles varying object sizes throughout video sequences.

## Architecture

```
DSTARK
├── DINOv3 Backbone (Small)
│   ├── Patch size: 16×16
│   ├── Embedding dim: 384
│   ├── Depth: 12 layers
│   └── RoPE for flexible sizing
│
└── Correlation Head
    ├── Feature projection
    ├── Template-search correlation
    ├── Bounding box regression
    └── Confidence prediction
```

## Installation

```bash
# Clone repository
git clone https://github.com/sahinemreaslan/dstrak.git
cd dstrak

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- OpenCV
- NumPy
- tqdm
- PyYAML

## Dataset Preparation

### GOT-10k

```bash
# Download GOT-10k dataset
mkdir -p data/GOT10k
cd data/GOT10k

# Download from http://got-10k.aitestunion.com/downloads
# Extract train and test splits

# Expected structure:
# data/GOT10k/
#   ├── train/
#   │   ├── GOT-10k_Train_000001/
#   │   ├── GOT-10k_Train_000002/
#   │   └── ...
#   └── test/
#       ├── GOT-10k_Test_000001/
#       └── ...
```

## Pretrained Weights

Download DINOv3 small pretrained weights:

```bash
# DINOv3 small weights (already included in repo)
# dinov3_vits16_pretrain.pth
```

## Training

```bash
# Train DSTARK on GOT-10k
python train.py \
    --config configs/dstark_train.yaml \
    --data_root data/GOT10k \
    --output_dir output/dstark \
    --pretrained dinov3_vits16_pretrain.pth \
    --epochs 300 \
    --batch_size 16 \
    --gpu 0

# Resume training from checkpoint
python train.py \
    --config configs/dstark_train.yaml \
    --resume output/dstark/checkpoint_epoch_100.pth
```

### Training Configuration

Key parameters in `configs/dstark_train.yaml`:

- `template_size`: Base template size (default: 128)
- `search_size`: Base search size (default: 256)
- `max_template_size`: Maximum template size (default: 256)
- `max_search_size`: Maximum search size (default: 512)
- `backbone_lr`: Learning rate for backbone (default: 1e-5)
- `head_lr`: Learning rate for tracking head (default: 1e-4)

## Testing

```bash
# Test on GOT-10k
python test.py \
    --checkpoint output/dstark/best_model.pth \
    --data_root data/GOT10k \
    --benchmark GOT10k \
    --output_dir results/got10k \
    --gpu 0

# Test with visualization
python test.py \
    --checkpoint output/dstark/best_model.pth \
    --benchmark GOT10k \
    --visualize
```

## Advantages over Standard STARK

| Feature | STARK | DSTARK |
|---------|-------|--------|
| Template Size | Fixed (128×128) | Flexible (128-256) |
| Search Size | Fixed (256×256) | Flexible (256-512) |
| Backbone | ResNet-50 | DINOv3 Small |
| Position Encoding | Learnable | RoPE (rotation-based) |
| Feature Quality | Standard | Rich self-supervised |
| Occlusion Handling | Moderate | Improved |
| Scale Variation | Limited | Adaptive |

## Why DINOv3 with RoPE?

Traditional trackers struggle with:
- **Fixed size constraints**: 128×128 template, 256×256 search
- **Poor occlusion handling**: Features not distinctive enough
- **Scale variation issues**: Can't adapt to changing object sizes
- **Target switching**: May lock onto wrong object after occlusion

DINOv3 solves these with:
- **RoPE (Rotary Position Embeddings)**: Enables flexible input sizes
- **Rich features**: Better discrimination between objects
- **Self-supervised learning**: More robust representations
- **No size constraints**: Can process varying dimensions

## Project Structure

```
dstrak/
├── dstark/                    # Main package
│   ├── models/                # Model definitions
│   │   ├── dinov3_backbone.py # DINOv3 backbone
│   │   └── dstark_tracker.py  # DSTARK tracker
│   ├── data/                  # Data loading
│   │   ├── tracking_dataset.py
│   │   └── sampler.py
│   ├── lib/                   # Training utilities
│   │   ├── losses.py
│   │   └── train_utils.py
│   └── utils/                 # Utility functions
│       ├── box_ops.py
│       └── misc.py
├── configs/                   # Configuration files
│   ├── dstark_train.yaml
│   └── dstark_test.yaml
├── train.py                   # Training script
├── test.py                    # Testing script
├── dinov3_vits16_pretrain.pth # Pretrained weights
└── README.md
```

## Citation

If you use DSTARK in your research, please cite:

```bibtex
@article{dstark2024,
  title={DSTARK: DINOv3-based STARK Tracker with Flexible Template and Search Sizes},
  author={Your Name},
  year={2024}
}
```

## Acknowledgements

- [STARK](https://github.com/researchmm/Stark) - Original STARK tracker
- [DINOv3](https://github.com/facebookresearch/dinov3) - DINOv3 self-supervised learning

## License

This project is released under the MIT License.

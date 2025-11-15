"""
Test script to verify the refactored architecture.

This script tests:
1. Import structure
2. Factory pattern usage
3. Configuration system
4. Model creation
5. Forward pass
"""

import torch
import sys


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        # Base classes
        from dstark.models import BaseBackbone, FlexibleBackbone
        from dstark.models import BaseTrackingHead, CorrelationBasedHead

        # Concrete implementations
        from dstark.models import DINOv3Backbone, CorrelationHead, DSTARKTracker

        # Configuration
        from dstark.models import (
            DSTARKConfig,
            DINOv3Config,
            CorrelationHeadConfig,
            TrainingConfig
        )

        # Factory
        from dstark.models import ModelFactory, BackboneFactory, HeadFactory

        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration system."""
    print("\n" + "=" * 60)
    print("Testing configuration system...")
    print("=" * 60)

    try:
        from dstark.models import DSTARKConfig, DINOv3Config, CorrelationHeadConfig

        # Test default config
        config = DSTARKConfig.default()
        print(f"✓ Default config created")
        print(f"  - Backbone type: {config.backbone.type}")
        print(f"  - Embed dim: {config.backbone.embed_dim}")
        print(f"  - Head type: {config.head.type}")
        print(f"  - Hidden dim: {config.head.hidden_dim}")

        # Test config variants
        dinov3_small = DINOv3Config.small()
        dinov3_base = DINOv3Config.base()
        dinov3_large = DINOv3Config.large()

        print(f"✓ DINOv3 variants created")
        print(f"  - Small embed_dim: {dinov3_small.embed_dim}")
        print(f"  - Base embed_dim: {dinov3_base.embed_dim}")
        print(f"  - Large embed_dim: {dinov3_large.embed_dim}")

        # Test config to dict
        config_dict = config.to_dict()
        print(f"✓ Config serialization successful")

        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_pattern():
    """Test factory pattern."""
    print("\n" + "=" * 60)
    print("Testing factory pattern...")
    print("=" * 60)

    try:
        from dstark.models import (
            ModelFactory,
            BackboneFactory,
            HeadFactory,
            DSTARKConfig,
            DINOv3Config,
            CorrelationHeadConfig
        )

        # Test backbone factory
        backbone_config = DINOv3Config.small()
        backbone = BackboneFactory.create_backbone(backbone_config)
        print(f"✓ Backbone created via factory")
        print(f"  - Type: {type(backbone).__name__}")
        print(f"  - Embed dim: {backbone.embed_dim}")
        print(f"  - Patch size: {backbone.patch_size}")

        # Test head factory
        head_config = CorrelationHeadConfig()
        head = HeadFactory.create_head(head_config, feat_dim=384)
        print(f"✓ Head created via factory")
        print(f"  - Type: {type(head).__name__}")
        print(f"  - Feat dim: {head.feat_dim}")
        print(f"  - Hidden dim: {head.hidden_dim}")

        # Test complete tracker factory
        config = DSTARKConfig.default()
        tracker = ModelFactory.create_tracker(config)
        print(f"✓ Complete tracker created via factory")
        print(f"  - Type: {type(tracker).__name__}")
        print(f"  - Feat dim: {tracker.feat_dim}")

        return True
    except Exception as e:
        print(f"✗ Factory pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Testing model forward pass...")
    print("=" * 60)

    try:
        from dstark.models import ModelFactory, DSTARKConfig

        # Create model
        config = DSTARKConfig.default()
        tracker = ModelFactory.create_tracker(config)
        tracker.eval()

        print(f"✓ Model created and set to eval mode")

        # Test with dummy data
        batch_size = 2
        template = torch.randn(batch_size, 3, 128, 128)
        search = torch.randn(batch_size, 3, 256, 256)

        print(f"✓ Created dummy inputs:")
        print(f"  - Template: {template.shape}")
        print(f"  - Search: {search.shape}")

        # Forward pass
        with torch.no_grad():
            output = tracker(template, search)

        print(f"✓ Forward pass successful")
        print(f"  - pred_boxes: {output['pred_boxes'].shape}")
        print(f"  - pred_conf: {output['pred_conf'].shape}")

        # Test online tracking mode
        with torch.no_grad():
            # Extract template features
            template_feat = tracker.template(template)
            print(f"✓ Template features extracted: {template_feat.shape}")

            # Track
            track_output = tracker.track(template_feat, search)
            print(f"✓ Online tracking successful")

            # Get box and score
            boxes, scores = tracker.get_box_and_score(track_output)
            print(f"✓ Box extraction successful")
            print(f"  - Boxes: {boxes.shape}")
            print(f"  - Scores: {scores.shape}")

        return True
    except Exception as e:
        print(f"✗ Model forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_legacy_compatibility():
    """Test backward compatibility with legacy code."""
    print("\n" + "=" * 60)
    print("Testing legacy compatibility...")
    print("=" * 60)

    try:
        from dstark.models import build_dstark

        # Test legacy build function
        config = {'hidden_dim': 256}
        tracker = build_dstark(config)

        print(f"✓ Legacy build_dstark() works")
        print(f"  - Type: {type(tracker).__name__}")

        # Test forward pass
        template = torch.randn(1, 3, 128, 128)
        search = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            output = tracker(template, search)

        print(f"✓ Legacy model forward pass works")

        return True
    except Exception as e:
        print(f"✗ Legacy compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flexible_sizes():
    """Test flexible input sizes."""
    print("\n" + "=" * 60)
    print("Testing flexible input sizes...")
    print("=" * 60)

    try:
        from dstark.models import ModelFactory, DSTARKConfig

        config = DSTARKConfig.default()
        tracker = ModelFactory.create_tracker(config)
        tracker.eval()

        # Test different sizes
        sizes = [
            (128, 128, 256, 256),  # Standard
            (160, 160, 320, 320),  # Larger
            (96, 96, 192, 192),    # Smaller
        ]

        for t_h, t_w, s_h, s_w in sizes:
            template = torch.randn(1, 3, t_h, t_w)
            search = torch.randn(1, 3, s_h, s_w)

            with torch.no_grad():
                output = tracker(template, search)

            print(f"✓ Size ({t_h}×{t_w}, {s_h}×{s_w}) works")

        return True
    except Exception as e:
        print(f"✗ Flexible sizes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DSTARK Architecture Validation Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Factory Pattern", test_factory_pattern),
        ("Model Forward", test_model_forward),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Flexible Sizes", test_flexible_sizes),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Architecture is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# DSTARK Architecture Documentation

## Overview

DSTARK (DINOv3-based STARK Tracker) now follows modern software design principles and patterns for better maintainability, extensibility, and testability.

## Design Principles Applied

### SOLID Principles

1. **Single Responsibility Principle (SRP)**
   - Each class has one clear responsibility
   - `DINOv3Backbone`: Feature extraction only
   - `CorrelationHead`: Tracking prediction only
   - `DSTARKTracker`: Orchestrates components

2. **Open/Closed Principle (OCP)**
   - Open for extension, closed for modification
   - New backbones can be added via `BackboneFactory.register_backbone()`
   - New heads can be added via `HeadFactory.register_head()`

3. **Liskov Substitution Principle (LSP)**
   - Any `BaseBackbone` subclass can replace another
   - Any `BaseTrackingHead` subclass can replace another

4. **Interface Segregation Principle (ISP)**
   - `BaseBackbone`: Minimal required interface
   - `FlexibleBackbone`: Extended interface for flexible sizes
   - `CorrelationBasedHead`: Specialized for correlation tracking

5. **Dependency Inversion Principle (DIP)**
   - `DSTARKTracker` depends on abstractions (`BaseBackbone`, not `DINOv3Backbone`)
   - Components are injected, not instantiated internally

### Design Patterns

1. **Strategy Pattern**
   - Backbones and heads are interchangeable strategies
   - Allows runtime swapping of algorithms

2. **Factory Pattern**
   - `BackboneFactory`, `HeadFactory`, `ModelFactory`
   - Centralizes object creation logic
   - Simplifies complex object construction

3. **Dependency Injection**
   - `DSTARKTracker` receives dependencies via constructor
   - Improves testability and flexibility

4. **Facade Pattern**
   - `DSTARKTracker` provides simple interface to complex tracking

## Architecture

```
dstark/models/
├── base_backbone.py          # Abstract backbone interfaces
│   ├── BaseBackbone          # Minimal backbone interface
│   └── FlexibleBackbone      # Extended for flexible sizes
│
├── base_head.py              # Abstract head interfaces
│   ├── BaseTrackingHead      # Minimal head interface
│   ├── CorrelationBasedHead  # Correlation-specific interface
│   └── TransformerBasedHead  # Transformer-specific interface
│
├── config.py                 # Type-safe configurations
│   ├── BackboneConfig        # Base config
│   ├── DINOv3Config          # DINOv3 specific (small/base/large)
│   ├── HeadConfig            # Base head config
│   ├── CorrelationHeadConfig # Correlation head config
│   ├── DSTARKConfig          # Complete model config
│   └── TrainingConfig        # Training hyperparameters
│
├── dinov3_backbone.py        # DINOv3 implementation
│   └── DINOv3Backbone        # Extends FlexibleBackbone
│
├── dstark_tracker.py         # Main tracker
│   ├── CorrelationHead       # Implements CorrelationBasedHead
│   └── DSTARKTracker         # Main tracker (uses DI)
│
├── factory.py                # Factory pattern
│   ├── BackboneFactory       # Creates backbones from config
│   ├── HeadFactory           # Creates heads from config
│   └── ModelFactory          # Creates complete trackers
│
└── __init__.py               # Clean public API
```

## Usage Examples

### Recommended: Factory Pattern with Type-Safe Config

```python
from dstark.models import ModelFactory, DSTARKConfig

# Create default configuration
config = DSTARKConfig.default()

# Or create with pretrained weights
config = DSTARKConfig.from_pretrained(
    pretrained_path='weights/dinov3_small.pth',
    backbone_size='small'
)

# Create tracker
tracker = ModelFactory.create_tracker(config)

# Use tracker
output = tracker(template, search)
```

### Alternative: Direct Construction

```python
from dstark.models import (
    DINOv3Backbone,
    CorrelationHead,
    DSTARKTracker,
    DINOv3Config,
    CorrelationHeadConfig
)

# Create components
backbone_config = DINOv3Config.small(pretrained_path='weights.pth')
backbone = DINOv3Backbone(**backbone_config.to_dict())

head = CorrelationHead(feat_dim=384, hidden_dim=256)

# Assemble tracker (Dependency Injection)
tracker = DSTARKTracker(backbone=backbone, head=head)
```

### Legacy: Backward Compatible

```python
from dstark.models import build_dstark

# Old way still works
tracker = build_dstark({
    'hidden_dim': 256,
    'pretrained_path': 'weights.pth'
})
```

## Key Improvements

### 1. Type Safety

- **Before**: Dictionary configs, prone to typos and errors
- **After**: Dataclass configs with type hints and validation

```python
# Before (error-prone)
config = {'embed_dim': '384'}  # Wrong type, no error until runtime

# After (type-safe)
config = DINOv3Config(embed_dim='384')  # Type checker catches this!
```

### 2. Extensibility

- **Before**: Hard to add new backbones or heads
- **After**: Register new components easily

```python
# Add custom backbone
class MyBackbone(BaseBackbone):
    def forward(self, x):
        return custom_feature_extraction(x)

    @property
    def embed_dim(self):
        return 512

# Register it
BackboneFactory.register_backbone('mybackbone', MyBackbone)

# Use it
config.backbone.type = 'mybackbone'
tracker = ModelFactory.create_tracker(config)
```

### 3. Testability

- **Before**: Tight coupling makes testing difficult
- **After**: Dependency injection enables easy mocking

```python
# Mock backbone for testing
class MockBackbone(BaseBackbone):
    def forward(self, x):
        return torch.zeros(x.shape[0], 100, 384)

mock_backbone = MockBackbone()
mock_head = CorrelationHead(384, 256)
tracker = DSTARKTracker(mock_backbone, mock_head)  # Easy to test!
```

### 4. Configuration Management

- **Before**: Scattered configuration across multiple dictionaries
- **After**: Centralized, validated, serializable configs

```python
config = DSTARKConfig.default()
config.validate()  # Checks consistency
config_dict = config.to_dict()  # Easy serialization
```

### 5. Documentation

- **Before**: Minimal docstrings
- **After**: Comprehensive docstrings with type hints

```python
def forward(
    self,
    template: torch.Tensor,  # Clear type
    search: torch.Tensor,    # Clear type
    return_features: bool = False  # Clear default
) -> Dict[str, torch.Tensor]:  # Clear return type
    """
    Detailed documentation of what this does,
    what arguments it takes, and what it returns.
    """
```

## Benefits

1. **Maintainability**: Clear structure, well-documented code
2. **Extensibility**: Easy to add new components
3. **Testability**: Loose coupling enables unit testing
4. **Type Safety**: Catch errors at design time, not runtime
5. **Reusability**: Components can be used independently
6. **Backward Compatibility**: Old code still works

## Migration Guide

### For Existing Code

Your existing code continues to work:

```python
# This still works
from dstark.models import DSTARKTracker, build_dstark
tracker = build_dstark({'hidden_dim': 256})
```

### Recommended Migration

Gradually migrate to the new pattern:

```python
# Step 1: Use config classes
from dstark.models import DSTARKConfig
config = DSTARKConfig.default()

# Step 2: Use factory
from dstark.models import ModelFactory
tracker = ModelFactory.create_tracker(config)
```

## Future Extensions

The new architecture makes it easy to add:

1. **New Backbones**: ResNet, Swin Transformer, etc.
2. **New Heads**: Transformer-based, GNN-based, etc.
3. **Multi-scale Features**: Pyramid features from backbone
4. **Online Learning**: Update tracker during inference
5. **Ensemble Tracking**: Multiple trackers voting

## Validation

Run the architecture validation tests:

```bash
python test_architecture.py
```

This verifies:
- All imports work correctly
- Factory pattern functions properly
- Configuration system is valid
- Models can be created and run
- Backward compatibility is maintained
- Flexible input sizes work

## Version

- **Architecture Version**: 1.0.0
- **Compatible with**: DSTARK v1.x

## Authors

- Original DSTARK implementation
- Architecture refactoring: Applied SOLID principles and design patterns

## License

Same as DSTARK project license

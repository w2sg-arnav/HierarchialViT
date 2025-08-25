# Model Architecture

The HierarchialViT architecture consists of multiple stages of transformer blocks that process visual information at different scales. Each stage operates on a different resolution of the input image, allowing the model to build a hierarchical representation of visual features.

## Key Components

1. **Patch Embedding**
   - Input images are divided into patches
   - Patches are linearly embedded into tokens

2. **Hierarchical Stages**
   - Multiple transformer stages
   - Progressive reduction in spatial dimensions
   - Increasing channel dimensions

3. **Attention Mechanisms**
   - Self-attention at each stage
   - Cross-stage attention for information flow

4. **Feature Pyramid**
   - Multi-scale feature outputs
   - Suitable for various downstream tasks

For implementation details, see the code in `phase2_model/models/hvt.py`.

# HierarchialViT: Hierarchical Vision Transformer for Computer Vision

HierarchialViT is a novel vision transformer architecture that processes visual information hierarchically, offering improved efficiency and performance for computer vision tasks. Our architecture incorporates multi-scale feature learning and progressive dimension reduction, making it particularly effective for complex visual recognition tasks.

## Key Features

1. **Hierarchical Processing**
   - Multi-stage transformer architecture
   - Progressive spatial dimension reduction
   - Increasing channel dimensions for rich feature representation

2. **Efficient Design**
   - Optimized attention mechanisms
   - Memory-efficient implementation
   - Scalable to large datasets

3. **Strong Performance**
   - State-of-the-art results on vision tasks
   - Robust feature learning
   - Effective transfer learning capabilities

## Model Architecture

The HierarchialViT consists of multiple stages where each stage processes the input at a different scale:

1. **Patch Embedding**: Input images are divided into patches and embedded into tokens
2. **Hierarchical Stages**: Multiple transformer stages with progressive reduction in spatial dimensions
3. **Feature Pyramid**: Multi-scale feature outputs suitable for various downstream tasks

For detailed architecture information, see [architecture documentation](docs/architecture.md).

## Project Structure

The project is organized into multiple phases:

1. **Phase 1**: Initial project setup and baseline implementation
   - Located in `phase1_project/`
   - Contains basic data utilities and model implementations

2. **Phase 2**: Model Development
   - Located in `phase2_model/`
   - Implements different model variants including baseline, DFCA, and HVT

3. **Phase 3**: Pre-training
   - Located in `phase3_pretraining/`
   - Contains self-supervised learning implementation and pre-training scripts

4. **Phase 4**: Fine-tuning
   - Located in `phase4_finetuning/`
   - Scripts for fine-tuning the pre-trained models

5. **Phase 5**: Analysis and Ablation Studies
   - Located in `phase5_analysis_and_ablation/`
   - Contains scripts for model analysis, visualization, and robustness testing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/w2sg-arnav/HierarchialViT.git
cd HierarchialViT
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Pre-training
```bash
cd phase3_pretraining
python run_ssl_pretraining.py
```

### Fine-tuning
```bash
cd phase4_finetuning
python main.py --config config.yaml
```

### Analysis
```bash
cd phase5_analysis_and_ablation
python analyze_best_model.py
```

## Results

Our model demonstrates:
- Improved efficiency compared to standard ViT
- Better hierarchical feature learning
- Robust performance across different datasets

Detailed results and analysis can be found in the `phase5_analysis_and_ablation/analysis_results/` directory.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hierarchialvit2025,
  title={HierarchialViT: A Hierarchical Vision Transformer for Efficient Computer Vision},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

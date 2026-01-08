# 🤝 Contributing to HierarchicalViT

Thank you for your interest in contributing to HierarchicalViT! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. We are committed to providing a welcoming and inclusive environment for everyone.

---

## Getting Started

### Types of Contributions

We welcome many types of contributions:

| Type | Description |
|------|-------------|
| 🐛 **Bug Fixes** | Fix issues and improve stability |
| ✨ **Features** | Add new functionality |
| 📚 **Documentation** | Improve docs, examples, tutorials |
| 🧪 **Tests** | Add or improve test coverage |
| ⚡ **Performance** | Optimize speed or memory usage |
| 🔧 **Refactoring** | Improve code quality without changing behavior |

### First-Time Contributors

Look for issues labeled:
- `good first issue` - Simple issues for newcomers
- `help wanted` - We need community help
- `documentation` - Documentation improvements

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- CUDA-capable GPU (recommended for training)

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/HierarchialViT.git
cd HierarchialViT

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 5. Verify installation
python -c "from hvit.models import DiseaseAwareHVT; print('✅ Setup complete!')"
pytest tests/ -v
```

### Development Dependencies

```bash
# requirements-dev.txt includes:
# - pytest, pytest-cov (testing)
# - black, isort (formatting)
# - flake8, mypy (linting)
# - pre-commit (git hooks)
```

---

## Making Changes

### Branch Naming

Use descriptive branch names:

| Type | Format | Example |
|------|--------|---------|
| Feature | `feature/description` | `feature/add-attention-visualization` |
| Bug Fix | `fix/description` | `fix/memory-leak-in-dataloader` |
| Docs | `docs/description` | `docs/improve-training-guide` |
| Refactor | `refactor/description` | `refactor/simplify-loss-functions` |

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change, no new feature
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(models): add support for spectral input channels"
git commit -m "fix(training): resolve NaN loss in focal loss computation"
git commit -m "docs(readme): add installation instructions for Windows"
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=hvit --cov-report=html

# Run only fast tests (skip slow/GPU tests)
pytest tests/ -v -m "not slow"
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_my_feature.py

import pytest
import torch
from hvit.models import create_disease_aware_hvt


class TestMyFeature:
    """Tests for my new feature."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return create_disease_aware_hvt(
            current_img_size=(256, 256),
            num_classes=7,
            model_params_dict={"embed_dim_rgb": 96}
        )
    
    def test_forward_pass(self, model):
        """Test that forward pass works correctly."""
        x = torch.randn(2, 3, 256, 256)
        output = model(x, mode='classify')
        assert output.shape == (2, 7)
    
    def test_gradient_flow(self, model):
        """Test that gradients flow correctly."""
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        output = model(x, mode='classify')
        output.sum().backward()
        assert x.grad is not None
```

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Maintain >80% code coverage

---

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Format code
   black hvit/ tests/ scripts/
   isort hvit/ tests/ scripts/
   
   # Run linting
   flake8 hvit/ tests/ scripts/
   
   # Run tests
   pytest tests/ -v
   ```

3. **Update documentation** if needed

### Submitting PR

1. Push your branch to your fork
2. Open a Pull Request against `main`
3. Fill out the PR template
4. Wait for CI checks to pass
5. Request review from maintainers

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
Describe how you tested the changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
```

### Review Process

1. Automated CI checks run
2. At least one maintainer reviews
3. Address any feedback
4. Maintainer approves and merges

---

## Code Style

### Python Style Guide

We follow PEP 8 with these specifications:

| Rule | Value |
|------|-------|
| Line length | 100 characters |
| Indentation | 4 spaces |
| Quotes | Double quotes for strings |
| Imports | Sorted with isort |

### Formatting Tools

```bash
# Auto-format code
black hvit/ tests/ scripts/ --line-length 100

# Sort imports
isort hvit/ tests/ scripts/

# Check style (without modifying)
flake8 hvit/ tests/ scripts/
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor


def create_model(
    img_size: Tuple[int, int],
    num_classes: int,
    config: Optional[Dict[str, any]] = None
) -> torch.nn.Module:
    """Create a model instance.
    
    Args:
        img_size: Input image size as (height, width).
        num_classes: Number of output classes.
        config: Optional configuration dictionary.
    
    Returns:
        Initialized model.
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_loss(
    predictions: Tensor,
    targets: Tensor,
    class_weights: Optional[Tensor] = None
) -> Tensor:
    """Compute the classification loss.
    
    Args:
        predictions: Model predictions of shape (B, C).
        targets: Ground truth labels of shape (B,).
        class_weights: Optional per-class weights of shape (C,).
    
    Returns:
        Scalar loss tensor.
    
    Raises:
        ValueError: If predictions and targets have mismatched batch sizes.
    
    Example:
        >>> preds = torch.randn(32, 7)
        >>> labels = torch.randint(0, 7, (32,))
        >>> loss = compute_loss(preds, labels)
    """
    ...
```

---

## Documentation

### Updating Documentation

1. **README.md**: Update for major features
2. **docs/**: Update technical documentation
3. **Docstrings**: Add to all public functions
4. **Examples**: Include usage examples

### Building Docs Locally

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve

# Open http://localhost:8000
```

---

## Reporting Issues

### Bug Reports

Include:

1. **Environment info**:
   ```bash
   python --version
   pip show torch hvit
   nvidia-smi
   ```

2. **Steps to reproduce**

3. **Expected vs actual behavior**

4. **Error messages/stack traces**

5. **Minimal code example**

### Feature Requests

Include:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How would you implement it?
3. **Alternatives**: What alternatives did you consider?
4. **Use case**: Who would benefit from this?

---

## Getting Help

- 📖 **Documentation**: Check [docs/](docs/)
- 💬 **Discussions**: Use [GitHub Discussions](https://github.com/w2sg-arnav/HierarchialViT/discussions)
- 🐛 **Issues**: Report bugs on [GitHub Issues](https://github.com/w2sg-arnav/HierarchialViT/issues)

---

## Recognition

Contributors will be:
- Listed in the README
- Credited in release notes
- Added to the AUTHORS file

Thank you for contributing to HierarchicalViT! 🎉

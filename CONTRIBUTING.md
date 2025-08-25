# Contributing to HierarchialViT

We welcome contributions to the HierarchialViT project! This document provides guidelines for contributing to the project.

## Development Setup

1. **Create Environment**
```bash
# Using mamba (recommended)
mamba env create -p ./env -f environment.yml
mamba env update -f environment-dev.yml

# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Install Pre-commit Hooks**
```bash
pre-commit install
```

## Code Style

We follow PEP 8 guidelines with these tools:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run formatting:
```bash
make format
```

Check style:
```bash
make lint
```

## Testing

Write tests for new features:
```python
def test_new_feature():
    model = HierarchialViT()
    output = model(input_tensor)
    assert output.shape == expected_shape
```

Run tests:
```bash
make test
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Documentation

- Add docstrings to all functions
- Update relevant README files
- Include benchmark results if applicable

## Reporting Issues

- Use the issue tracker
- Include reproduction steps
- Attach relevant logs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="hvit",
    version="0.1.0",
    description="Hierarchical Vision Transformer for Computer Vision",
    author="w2sg-arnav",
    author_email="",  # Add your email
    url="https://github.com/w2sg-arnav/HierarchialViT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "pillow",
        "pyyaml",
        "tensorboard",
        "matplotlib",
        "scikit-learn",
        "pandas",
        "tqdm",
        "wandb",
        "timm",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)

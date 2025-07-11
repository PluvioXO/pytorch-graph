from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torch-vis",
    version="0.1.0",
    author="Torch Vis Team",
    author_email="contact@torchvis.com",
    description="A PyTorch-specific package for 3D visualization of neural network architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/torch-vis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Framework :: PyTorch",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
        "networkx>=2.5",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "kaleido>=0.2.1",
        "torchinfo>=1.6.0",  # For enhanced model analysis
        "torchsummary>=1.5.1",  # For model summaries
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8.0",
        ],
        "examples": [
            "torchvision>=0.9.0",
            "timm>=0.6.0",  # PyTorch Image Models
            "transformers>=4.0.0",  # Hugging Face transformers
        ],
    },
    entry_points={
        "console_scripts": [
            "torch-vis=torch_vis.cli:main",
        ],
    },
) 
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch-graph",
    version="0.2.1",
    author="Max",
    author_email="max@ahmresearch.com",
    description="Enhanced PyTorch neural network architecture visualization with flowchart diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pluvioxo/pytorch-graph",
    project_urls={
        "Bug Tracker": "https://github.com/pluvioxo/pytorch-graph/issues",
        "Documentation": "https://pytorch-graph.readthedocs.io/",
        "Source Code": "https://github.com/pluvioxo/pytorch-graph",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="pytorch neural-network visualization flowchart architecture diagram machine-learning deep-learning",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
        "networkx>=2.5",
        "pandas>=1.2.0",
    ],
    extras_require={
        "full": [
            "torchvision>=0.9.0",
            "scipy>=1.6.0",
            "dash>=2.0.0",
            "dash-bootstrap-components>=1.0.0",
            "kaleido>=0.2.1",
            "torchinfo>=1.6.0",
            "torchsummary>=1.5.1",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "jupyter>=1.0.0",
            "twine>=4.0.0",
            "build>=0.8.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8.0",
        ],
        "examples": [
            "torchvision>=0.9.0",
            "timm>=0.6.0",
            "transformers>=4.0.0",
        ],
    },
    package_data={
        "torch_vis": ["*.py"],
    },
    include_package_data=True,
    zip_safe=False,
) 
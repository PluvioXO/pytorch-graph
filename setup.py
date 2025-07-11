from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-viz-3d",
    version="0.1.0",
    author="Neural Viz Team",
    author_email="contact@neuralviz.com",
    description="A Python package for 3D visualization of neural network architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-viz-3d",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
        "networkx>=2.5",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "tensorflow>=2.4.0",
        "keras>=2.4.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "kaleido>=0.2.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "neural-viz=neural_viz.cli:main",
        ],
    },
) 
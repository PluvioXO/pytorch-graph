# Contributing to PyTorch Graph

We welcome contributions to PyTorch Graph! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)
- [Testing](#testing)
- [Code Style](#code-style)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- Git
- Basic understanding of PyTorch and neural networks

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/pytorch-graph.git
   cd pytorch-graph
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks** (optional but recommended)
   ```bash
   pre-commit install
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix existing issues
- **Feature additions**: Add new functionality
- **Documentation improvements**: Enhance docs, examples, or README
- **Performance optimizations**: Improve speed or memory usage
- **Test coverage**: Add or improve tests
- **Code refactoring**: Improve code quality and maintainability

### Before You Start

1. **Check existing issues** to see if your contribution is already being worked on
2. **Open an issue** for significant changes to discuss the approach
3. **Keep changes focused** - one feature or fix per pull request
4. **Ensure backward compatibility** when possible

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

- Write clean, readable code
- Add appropriate comments and docstrings
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run the test suite
python -m pytest tests/

# Run the demo to ensure everything works
python demo.py

# Check code style
flake8 pytorch-graph/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add: brief description of changes

- Detailed description of what was changed
- Why the change was made
- Any relevant context"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots for visual changes
- Test results

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   - Python version
   - PyTorch version
   - Operating system
   - Package version

2. **Reproduction steps**:
   - Minimal code example
   - Expected behavior
   - Actual behavior
   - Error messages or stack traces

3. **Additional context**:
   - Screenshots if applicable
   - Related issues
   - Workarounds if any

### Feature Requests

For feature requests, please include:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: What other approaches were considered?
4. **Additional context**: Any other relevant information

## Documentation

### Documentation Standards

- **Docstrings**: Follow Google style docstrings
- **Type hints**: Use type annotations for function parameters and returns
- **Examples**: Include usage examples in docstrings
- **API documentation**: Keep API docs up to date

### Documentation Structure

```
docs/
├── api/                 # API reference
├── examples/            # Usage examples
├── guides/              # How-to guides
└── conf.py             # Sphinx configuration
```

### Building Documentation

```bash
cd docs/
make html
# Open _build/html/index.html in your browser
```

## Testing

### Test Structure

```
tests/
├── unit/               # Unit tests
├── integration/        # Integration tests
├── fixtures/           # Test data and models
└── conftest.py        # Pytest configuration
```

### Writing Tests

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows
- **Visual tests**: Test diagram generation (save outputs for comparison)
- **Performance tests**: Test execution time and memory usage

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/unit/test_parser.py

# Run with coverage
python -m pytest --cov=pytorch-graph

# Run visual tests (generates output files)
python -m pytest tests/integration/test_visualization.py
```

## Code Style

### Python Style Guide

- Follow **PEP 8** for Python code style
- Use **Black** for code formatting
- Use **isort** for import sorting
- Maximum line length: 88 characters

### Code Formatting

```bash
# Format code
black pytorch-graph/ tests/

# Sort imports
isort pytorch-graph/ tests/

# Check style
flake8 pytorch-graph/ tests/
```

### Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### File Organization

```
pytorch-graph/
├── __init__.py         # Public API
├── core/               # Core functionality
│   ├── parser.py       # Model parsing
│   └── visualizer.py   # Visualization logic
├── renderers/          # Output renderers
│   ├── diagram_renderer.py
│   └── plotly_renderer.py
└── utils/              # Utilities
    ├── computational_graph.py
    ├── layer_info.py
    ├── model_analyzer.py
    ├── position_calculator.py
    └── pytorch_hooks.py
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** to ensure everything works
4. **Update documentation** if needed
5. **Create release tag** on GitHub
6. **Build and upload** to PyPI

### Building for Release

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI (test first)
twine upload --repository testpypi dist/*
twine upload dist/*
```

## Development Tips

### Common Tasks

1. **Adding a new diagram style**:
   - Add style to `DiagramStyle` enum
   - Implement rendering logic in `DiagramRenderer`
   - Add tests and examples

2. **Adding a new model layer type**:
   - Update `LayerInfo` class
   - Add parsing logic in `PyTorchModelParser`
   - Update visualization logic

3. **Improving computational graph tracking**:
   - Modify `ComputationalGraphTracker`
   - Update graph traversal algorithms
   - Enhance visualization rendering

### Debugging

- Use `pdb` or IDE debugger for step-by-step debugging
- Add logging statements for complex operations
- Use `torch.jit.trace` to understand model execution
- Check tensor shapes and gradients during tracking

### Performance Optimization

- Profile code with `cProfile` or `line_profiler`
- Use `torch.profiler` for PyTorch-specific profiling
- Optimize matplotlib rendering for large graphs
- Consider memory usage for large models

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the full documentation at [pytorch-graph.readthedocs.io](https://pytorch-graph.readthedocs.io/)

## Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to PyTorch Graph! Your efforts help make PyTorch model visualization more accessible and powerful for the entire community.

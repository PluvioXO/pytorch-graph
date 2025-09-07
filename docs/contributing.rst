Contributing to PyTorch Graph
==============================

We welcome contributions to PyTorch Graph! This document provides guidelines for contributing to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/your-username/pytorch-graph.git
      cd pytorch-graph

3. **Install in development mode**:

   .. code-block:: bash

      pip install -e .[dev]

4. **Create a new branch** for your feature:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Setup
-----------------

Install development dependencies:

.. code-block:: bash

   pip install -e .[dev]

This installs:
* The package in editable mode
* Development dependencies (testing, linting, etc.)
* Documentation tools

Running Tests
-------------

Run the test suite:

.. code-block:: bash

   pytest

Run with coverage:

.. code-block:: bash

   pytest --cov=pytorch_graph

Code Quality
------------

We use several tools to maintain code quality:

**Linting:**
.. code-block:: bash

   flake8 pytorch_graph/
   black pytorch_graph/
   isort pytorch_graph/

**Type checking:**
.. code-block:: bash

   mypy pytorch_graph/

**Format code:**
.. code-block:: bash

   black pytorch_graph/
   isort pytorch_graph/

Building Documentation
----------------------

Build documentation locally:

.. code-block:: bash

   cd docs
   make html

View documentation:

.. code-block:: bash

   cd docs
   make serve

Then open http://localhost:8000 in your browser.

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

* Python version
* PyTorch version
* Operating system
* Minimal code example that reproduces the issue
* Expected vs actual behavior
* Error messages and stack traces

Feature Requests
~~~~~~~~~~~~~~~~

For feature requests, please:

* Check existing issues first
* Provide a clear description of the feature
* Explain the use case and benefits
* Consider implementation complexity

Code Contributions
~~~~~~~~~~~~~~~~~~

We welcome contributions for:

* Bug fixes
* New features
* Documentation improvements
* Performance optimizations
* Test coverage improvements

Pull Request Process
--------------------

1. **Create a feature branch** from the main branch
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run the test suite** and ensure all tests pass
6. **Submit a pull request** with a clear description

Pull Request Guidelines
-----------------------

* **Clear title and description**
* **Reference related issues**
* **Include tests** for new features
* **Update documentation** as needed
* **Keep changes focused** (one feature per PR)
* **Follow coding standards**

Coding Standards
----------------

* Follow PEP 8 style guidelines
* Use type hints where appropriate
* Write docstrings for all public functions
* Add tests for new functionality
* Keep functions focused and small
* Use meaningful variable names

Documentation Standards
-----------------------

* Write clear, concise docstrings
* Include examples in docstrings
* Update README.md for significant changes
* Add documentation for new features
* Use proper Sphinx formatting

Testing Guidelines
------------------

* Write tests for all new functionality
* Aim for high test coverage
* Test edge cases and error conditions
* Use descriptive test names
* Keep tests simple and focused

Release Process
---------------

Releases are managed by maintainers:

1. Update version numbers
2. Update CHANGELOG.md
3. Create release notes
4. Tag the release
5. Build and upload to PyPI

Community Guidelines
--------------------

* Be respectful and inclusive
* Help others learn and grow
* Provide constructive feedback
* Follow the code of conduct
* Welcome newcomers

Getting Help
------------

* **GitHub Issues**: For bug reports and feature requests
* **GitHub Discussions**: For questions and general discussion
* **Documentation**: Check the docs for usage examples
* **Code Examples**: Look at the demo scripts

Recognition
-----------

Contributors will be recognized in:

* CONTRIBUTORS.md file
* Release notes
* Documentation acknowledgments

Thank you for contributing to PyTorch Graph! 🚀

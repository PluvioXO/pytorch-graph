PyTorch Graph Documentation
============================

**Professional PyTorch neural network visualization toolkit with complete computational graph analysis**. Transform your PyTorch models into publication-ready diagrams with comprehensive architecture visualization and computational graph tracking.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/PyTorch-1.8+-red.svg
   :target: https://pytorch.org/
   :alt: PyTorch 1.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

.. image:: https://badge.fury.io/py/pytorch-graph.svg
   :target: https://badge.fury.io/py/pytorch-graph
   :alt: PyPI version

Quick Start
-----------

Install PyTorch Graph:

.. code-block:: bash

   pip install pytorch-graph

Generate a professional architecture diagram:

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_graph import generate_architecture_diagram

   # Define your model
   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   # Generate diagram
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="model_architecture.png",
       title="Neural Network Architecture"
   )

Track complete computational graph:

.. code-block:: python

   from pytorch_graph import track_computational_graph

   # Track computational graph
   tracker = track_computational_graph(
       model=model,
       input_tensor=torch.randn(1, 784, requires_grad=True)
   )

   # Save high-quality graph
   tracker.save_graph_png("computational_graph.png", dpi=300)

Features
--------

🚀 **Key Features:**

* **Architecture Visualization**: Professional flowchart diagrams with multiple styles
* **Complete Computational Graph Analysis**: Maximal traversal without artificial limits
* **Full Method Names**: Complete operation names without truncation
* **Smart Arrow Positioning**: Proper edge connections without crossing over boxes
* **Compact Layout**: Eliminates gaps and breaks for continuous flow
* **Professional Quality**: High-resolution output up to 300 DPI
* **Comprehensive Analysis**: Memory tracking, execution timing, and performance metrics

📊 **Architecture Diagrams:**
* Enhanced flowchart visualization (default)
* Research paper style for publications
* Standard neural network visualization
* High-quality PNG export with customizable DPI

🔍 **Computational Graph Tracking:**
* Complete autograd graph traversal
* Full operation coverage without limits
* Real-time memory and timing analysis
* Professional visualization with proper arrow positioning
* JSON data export for further analysis

📈 **Model Analysis:**
* Parameter counting and memory estimation
* Performance metrics and execution timing
* Layer-wise analysis and breakdown
* Model complexity assessment

Installation
------------

Basic installation:

.. code-block:: bash

   pip install pytorch-graph

With enhanced features:

.. code-block:: bash

   pip install pytorch-graph[full]

Development version:

.. code-block:: bash

   pip install pytorch-graph[dev]

Requirements
------------

* Python ≥ 3.8
* PyTorch ≥ 1.8.0
* matplotlib ≥ 3.3.0
* numpy ≥ 1.19.0

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   architecture_visualization
   computational_graph_tracking
   model_analysis
   advanced_features
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/main_module
   api/architecture
   api/computational_graph
   api/model_analysis
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   changelog
   license

Examples
--------

Architecture Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import generate_architecture_diagram

   # Generate flowchart style
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="flowchart.png",
       style="flowchart"
   )

   # Generate research paper style
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="research.png",
       style="research_paper"
   )

Computational Graph Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import ComputationalGraphTracker

   # Create tracker
   tracker = ComputationalGraphTracker(
       model=model,
       track_memory=True,
       track_timing=True,
       track_tensor_ops=True
   )

   # Track execution
   tracker.start_tracking()
   output = model(input_tensor)
   loss = output.sum()
   loss.backward()
   tracker.stop_tracking()

   # Save visualization
   tracker.save_graph_png(
       "complete_graph.png",
       width=1600,
       height=1200,
       dpi=300
   )

Model Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import analyze_model, analyze_computational_graph

   # Analyze model structure
   analysis = analyze_model(model, input_shape=(1, 784))
   print(f"Parameters: {analysis['total_parameters']:,}")

   # Analyze computational graph
   graph_analysis = analyze_computational_graph(
       model, input_tensor, detailed=True
   )
   print(f"Operations: {graph_analysis['summary']['total_nodes']:,}")

Support
-------

* **GitHub Issues**: `Report bugs or request features <https://github.com/your-username/pytorch-graph/issues>`_
* **GitHub Discussions**: `Ask questions or discuss ideas <https://github.com/your-username/pytorch-graph/discussions>`_
* **Documentation**: `Full documentation <https://pytorch-graph.readthedocs.io>`_

License
-------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/your-username/pytorch-graph/blob/main/LICENSE>`_ file for details.

Contributing
------------

Contributions are welcome! Please see our `Contributing Guidelines <contributing.html>`_ for details.

Acknowledgments
---------------

* Built for the PyTorch community
* Inspired by the need for better model visualization tools
* Designed for researchers, practitioners, and educators

---

**PyTorch Graph** - Professional PyTorch model visualization made simple, beautiful, and comprehensive! 🚀

Quick Start Guide
==================

This guide will get you up and running with PyTorch Graph in just a few minutes.

Installation
------------

First, install PyTorch Graph:

.. code-block:: bash

   pip install pytorch-graph

Basic Architecture Visualization
--------------------------------

Create a simple neural network and visualize its architecture:

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_graph import generate_architecture_diagram

   # Define a simple model
   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(128, 64),
       nn.ReLU(),
       nn.Linear(64, 10)
   )

   # Generate architecture diagram
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="my_model_architecture.png",
       title="My Neural Network"
   )

This creates a professional flowchart diagram showing your model's architecture.

Computational Graph Tracking
----------------------------

Track the complete computational graph of your model:

.. code-block:: python

   from pytorch_graph import track_computational_graph

   # Create input tensor
   input_tensor = torch.randn(1, 784, requires_grad=True)

   # Track computational graph
   tracker = track_computational_graph(
       model=model,
       input_tensor=input_tensor,
       track_memory=True,
       track_timing=True
   )

   # Save high-quality computational graph
   tracker.save_graph_png(
       "computational_graph.png",
       width=1600,
       height=1200,
       dpi=300
   )

This creates a detailed visualization of every operation in your model's computational graph.

Model Analysis
--------------

Analyze your model's structure and performance:

.. code-block:: python

   from pytorch_graph import analyze_model, analyze_computational_graph

   # Analyze model structure
   model_analysis = analyze_model(model, input_shape=(1, 784))
   print(f"Total parameters: {model_analysis['total_parameters']:,}")
   print(f"Model size: {model_analysis['model_size_mb']:.2f} MB")

   # Analyze computational graph
   graph_analysis = analyze_computational_graph(
       model, input_tensor, detailed=True
   )
   print(f"Total operations: {graph_analysis['summary']['total_nodes']:,}")
   print(f"Execution time: {graph_analysis['summary']['execution_time']:.4f}s")

Advanced Usage
--------------

For more control, use the ComputationalGraphTracker class directly:

.. code-block:: python

   from pytorch_graph import ComputationalGraphTracker

   # Create tracker with custom settings
   tracker = ComputationalGraphTracker(
       model=model,
       track_memory=True,
       track_timing=True,
       track_tensor_ops=True
   )

   # Start tracking
   tracker.start_tracking()

   # Run your model
   output = model(input_tensor)
   loss = output.sum()
   loss.backward()

   # Stop tracking
   tracker.stop_tracking()

   # Get comprehensive analysis
   summary = tracker.get_graph_summary()
   print(f"Operations: {summary['total_nodes']:,}")
   print(f"Execution time: {summary['execution_time']:.4f}s")

   # Save with custom parameters
   tracker.save_graph_png(
       "advanced_graph.png",
       width=2000,
       height=1500,
       dpi=300,
       show_legend=True,
       node_size=30,
       font_size=14
   )

Multiple Diagram Styles
-----------------------

Generate different styles of architecture diagrams:

.. code-block:: python

   # Flowchart style (default)
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="flowchart.png",
       style="flowchart"
   )

   # Research paper style
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="research.png",
       style="research_paper"
   )

   # Standard style
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="standard.png",
       style="standard"
   )

CNN Example
-----------

Here's a complete example with a CNN:

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_graph import generate_architecture_diagram, track_computational_graph

   # Define CNN
   cnn_model = nn.Sequential(
       nn.Conv2d(3, 32, 3, padding=1),
       nn.BatchNorm2d(32),
       nn.ReLU(),
       nn.MaxPool2d(2),
       
       nn.Conv2d(32, 64, 3, padding=1),
       nn.BatchNorm2d(64),
       nn.ReLU(),
       nn.MaxPool2d(2),
       
       nn.Flatten(),
       nn.Linear(64 * 8 * 8, 128),
       nn.ReLU(),
       nn.Dropout(0.5),
       nn.Linear(128, 10)
   )

   # Generate architecture diagram
   generate_architecture_diagram(
       model=cnn_model,
       input_shape=(1, 3, 32, 32),
       output_path="cnn_architecture.png",
       title="CNN Architecture"
   )

   # Track computational graph
   input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
   tracker = track_computational_graph(cnn_model, input_tensor)
   tracker.save_graph_png("cnn_computational_graph.png")

Next Steps
----------

Now that you have the basics, explore:

* :doc:`architecture_visualization` - Detailed architecture diagram features
* :doc:`computational_graph_tracking` - Complete computational graph analysis
* :doc:`model_analysis` - Model analysis and performance metrics
* :doc:`advanced_features` - Advanced customization and features
* :doc:`examples` - More comprehensive examples

Troubleshooting
---------------

If you encounter issues:

1. Make sure PyTorch is installed: ``pip install torch``
2. Check that matplotlib is available: ``pip install matplotlib``
3. Verify your input shapes match your model's expected input
4. For large models, consider using smaller input tensors for testing

For more help, see the `troubleshooting section <troubleshooting.html>`_ or visit our `GitHub Issues <https://github.com/your-username/pytorch-graph/issues>`_.

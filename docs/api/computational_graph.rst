Computational Graph Tracking API
=================================

This module provides comprehensive computational graph tracking and visualization capabilities.

Core Functions
--------------

.. autofunction:: pytorch_graph.track_computational_graph

.. autofunction:: pytorch_graph.analyze_computational_graph

Classes
-------

.. autoclass:: pytorch_graph.ComputationalGraphTracker
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pytorch_graph.GraphNode
   :members:
   :undoc-members:

.. autoclass:: pytorch_graph.GraphEdge
   :members:
   :undoc-members:

.. autoclass:: pytorch_graph.OperationType
   :members:
   :undoc-members:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_graph import track_computational_graph

   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   input_tensor = torch.randn(1, 784, requires_grad=True)

   # Track computational graph
   tracker = track_computational_graph(
       model=model,
       input_tensor=input_tensor,
       track_memory=True,
       track_timing=True,
       track_tensor_ops=True
   )

   # Save visualization
   tracker.save_graph_png("computational_graph.png")

Advanced Usage
~~~~~~~~~~~~~~

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

   # Run model
   output = model(input_tensor)
   loss = output.sum()
   loss.backward()

   # Stop tracking
   tracker.stop_tracking()

   # Get analysis
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

Analysis Functions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import analyze_computational_graph

   # Comprehensive analysis
   analysis = analyze_computational_graph(
       model=model,
       input_tensor=input_tensor,
       detailed=True
   )

   summary = analysis['summary']
   print(f"Total operations: {summary['total_nodes']:,}")
   print(f"Execution time: {summary['execution_time']:.4f}s")

   # Performance metrics
   if 'performance' in analysis:
       perf = analysis['performance']
       print(f"Operations per second: {perf['operations_per_second']:.2f}")
       print(f"Memory usage: {perf['memory_usage']}")

   # Layer-wise analysis
   if 'layer_analysis' in analysis:
       for layer_name, operations in analysis['layer_analysis'].items():
           print(f"{layer_name}: {len(operations)} operations")

Data Export
~~~~~~~~~~~

.. code-block:: python

   # Export graph data to JSON
   tracker.export_graph("graph_data.json")

   # Load and inspect exported data
   import json
   with open("graph_data.json", 'r') as f:
       graph_data = json.load(f)

   print(f"Nodes: {len(graph_data['nodes'])}")
   print(f"Edges: {len(graph_data['edges'])}")

Custom Visualization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save with custom parameters
   tracker.save_graph_png(
       filepath="custom_graph.png",
       width=1600,           # Image width
       height=1200,          # Image height
       dpi=300,              # High DPI
       show_legend=True,     # Show legend
       node_size=25,         # Node size
       font_size=12          # Font size
   )

Performance Tracking
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Track with performance monitoring
   tracker = ComputationalGraphTracker(
       model=model,
       track_memory=True,      # Track memory usage
       track_timing=True,      # Track execution timing
       track_tensor_ops=True   # Track tensor operations
   )

   tracker.start_tracking()
   
   # Your model execution
   output = model(input_tensor)
   loss = output.sum()
   loss.backward()
   
   tracker.stop_tracking()

   # Get performance summary
   summary = tracker.get_graph_summary()
   print(f"Memory usage: {summary['memory_usage']}")
   print(f"Execution time: {summary['execution_time']:.4f}s")

Parameters
----------

track_computational_graph
~~~~~~~~~~~~~~~~~~~~~~~~~

* **model** (torch.nn.Module): PyTorch model to track
* **input_tensor** (torch.Tensor): Input tensor for forward pass
* **track_memory** (bool): Whether to track memory usage (default: True)
* **track_timing** (bool): Whether to track execution timing (default: True)
* **track_tensor_ops** (bool): Whether to track tensor operations (default: True)

Returns: ComputationalGraphTracker object

ComputationalGraphTracker.save_graph_png
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **filepath** (str): Output file path
* **width** (int): Image width in pixels (default: 1200)
* **height** (int): Image height in pixels (default: 800)
* **dpi** (int): Dots per inch for high resolution (default: 300)
* **show_legend** (bool): Whether to show legend (default: True)
* **node_size** (int): Size of nodes in the graph (default: 20)
* **font_size** (int): Font size for labels (default: 10)

Returns: Path to the saved PNG file

Error Handling
--------------

The functions will raise appropriate exceptions for:

* Invalid model types
* Incorrect input tensors
* File system errors
* Memory issues with large models
* Missing dependencies

Performance Tips
----------------

* Use ``track_tensor_ops=False`` for large models to improve performance
* Use smaller input tensors for initial testing
* Monitor memory usage when tracking large models
* Export graph data for offline analysis of complex models

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`model_analysis` - For model analysis functions

Computational Graph Tracking
============================

PyTorch Graph provides comprehensive computational graph tracking and visualization capabilities.

Overview
--------

The computational graph tracking module allows you to:

* **Track Complete Graphs**: Capture every operation in your model's computational graph
* **Analyze Performance**: Monitor execution time, memory usage, and operation counts
* **Visualize Operations**: Create professional diagrams of the computational graph
* **Export Data**: Save graph data in JSON format for further analysis

Key Features
------------

* **Maximal Traversal**: No artificial limits on graph depth or operation count
* **Full Method Names**: Complete operation names without truncation
* **Smart Arrow Positioning**: Arrows connect node edges properly without crossing over boxes
* **Compact Layout**: Eliminates gaps and breaks for continuous flow
* **Professional Quality**: High-resolution output up to 300 DPI

Basic Usage
-----------

Track a computational graph with the convenience function:

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch-graph import track_computational_graph

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
--------------

Use the ComputationalGraphTracker class for full control:

.. code-block:: python

   from pytorch-graph import ComputationalGraphTracker

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

Analysis Functions
------------------

Get detailed analysis of your computational graph:

.. code-block:: python

   from pytorch-graph import analyze_computational_graph

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
-----------

Export graph data for offline analysis:

.. code-block:: python

   # Export to JSON
   tracker.export_graph("graph_data.json")

   # Load and inspect exported data
   import json
   with open("graph_data.json", 'r') as f:
       graph_data = json.load(f)

   print(f"Nodes: {len(graph_data['nodes'])}")
   print(f"Edges: {len(graph_data['edges'])}")

Visualization Features
----------------------

High-Quality Output
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   tracker.save_graph_png(
       filepath="publication_quality.png",
       width=2000,           # Custom width
       height=1500,          # Custom height
       dpi=300,              # High DPI for publication
       show_legend=True,     # Show legend
       node_size=30,         # Node size
       font_size=14          # Font size
   )

Custom Styling
~~~~~~~~~~~~~~

The computational graph visualization includes:

* **Full Method Names**: Complete operation names without truncation
* **Smart Arrow Positioning**: Arrows connect node edges properly
* **Compact Layout**: No gaps or breaks in the graph
* **Professional Styling**: Enhanced colors and typography
* **Intelligent Legends**: Automatic positioning without overlap

Examples
--------

CNN Computational Graph
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cnn_model = nn.Sequential(
       nn.Conv2d(3, 32, 3, padding=1),
       nn.BatchNorm2d(32),
       nn.ReLU(),
       nn.MaxPool2d(2),
       nn.Conv2d(32, 64, 3, padding=1),
       nn.BatchNorm2d(64),
       nn.ReLU(),
       nn.AdaptiveAvgPool2d((1, 1)),
       nn.Flatten(),
       nn.Linear(64, 10)
   )

   input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
   tracker = track_computational_graph(cnn_model, input_tensor)
   tracker.save_graph_png("cnn_computational_graph.png")

Complex Model Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ComplexModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.features = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(128, 256, 3, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d((1, 1))
           )
           self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(128, 10)
           )
           
       def forward(self, x):
           x = self.features(x)
           x = self.classifier(x)
           return x

   model = ComplexModel()
   input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
   
   # Track with performance monitoring
   tracker = ComputationalGraphTracker(
       model=model,
       track_memory=True,
       track_timing=True,
       track_tensor_ops=True
   )
   
   tracker.start_tracking()
   output = model(input_tensor)
   loss = output.sum()
   loss.backward()
   tracker.stop_tracking()
   
   # Get performance summary
   summary = tracker.get_graph_summary()
   print(f"Memory usage: {summary['memory_usage']}")
   print(f"Execution time: {summary['execution_time']:.4f}s")

Best Practices
--------------

* **Use appropriate input tensors** that match your model's expected input
* **Enable memory tracking** for performance analysis
* **Use high DPI** (300) for publication-quality output
* **Export graph data** for offline analysis of complex models
* **Monitor memory usage** when tracking large models

Performance Tips
----------------

* **Disable tensor operation tracking** for very large models to improve performance
* **Use smaller input tensors** for initial testing
* **Export graph data** for offline analysis of complex models
* **Monitor memory usage** when tracking large models

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Memory issues with large models**
   Use ``track_tensor_ops=False`` for better performance

**Long operation names**
   The system automatically handles long names without truncation

**Large graph visualization**
   Increase image size and use high DPI for better quality

**Import errors**
   Ensure all dependencies are installed: ``pip install torch matplotlib``

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`model_analysis` - For model analysis functions
* :doc:`api/computational_graph` - For complete API reference

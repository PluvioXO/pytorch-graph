Computational Graph Tracking API
=================================

This module provides comprehensive computational graph tracking and visualization capabilities for PyTorch models.

Core Functions
--------------

.. autofunction:: pytorch-graph.track_computational_graph

.. autofunction:: pytorch-graph.analyze_computational_graph

.. autofunction:: pytorch-graph.track_computational_graph_execution

.. autofunction:: pytorch-graph.analyze_computational_graph_execution

.. autofunction:: pytorch-graph.visualize_computational_graph

.. autofunction:: pytorch-graph.export_computational_graph

.. autofunction:: pytorch-graph.save_computational_graph_png

Classes
-------

.. autoclass:: pytorch-graph.ComputationalGraphTracker
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.GraphNode
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.GraphEdge
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.OperationType
   :members:
   :undoc-members:

Function Details
----------------

track_computational_graph
~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: track_computational_graph(model, input_tensor, track_memory=True, track_timing=True, track_tensor_ops=True)

   Track the computational graph of a PyTorch model execution.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to track
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **track_memory** (bool, optional): Whether to track memory usage (default: True)
   * **track_timing** (bool, optional): Whether to track execution timing (default: True)
   * **track_tensor_ops** (bool, optional): Whether to track tensor operations (default: True)

   **Returns:** ComputationalGraphTracker - Tracker instance with execution data

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

   **Example:**
   
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
      tracker = track_computational_graph(
          model=model,
          input_tensor=input_tensor,
          track_memory=True,
          track_timing=True,
          track_tensor_ops=True
      )

      # Get summary
      summary = tracker.get_graph_summary()
      print(f"Total operations: {summary['total_nodes']}")

analyze_computational_graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: analyze_computational_graph(model, input_tensor, detailed=True)

   Analyze the computational graph of a PyTorch model execution.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to analyze
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **detailed** (bool, optional): Whether to include detailed analysis (default: True)

   **Returns:** dict - Dictionary containing computational graph analysis

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

   **Example:**
   
   .. code-block:: python

      from pytorch-graph import analyze_computational_graph

      analysis = analyze_computational_graph(
          model=model,
          input_tensor=input_tensor,
          detailed=True
      )

      summary = analysis['summary']
      print(f"Total operations: {summary['total_nodes']}")
      print(f"Execution time: {summary['execution_time']:.4f}s")

track_computational_graph_execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: track_computational_graph_execution(model, input_tensor, track_memory=True, track_timing=True, track_tensor_ops=True)

   Track the computational graph of a PyTorch model execution (alias for track_computational_graph).

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to track
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **track_memory** (bool, optional): Whether to track memory usage (default: True)
   * **track_timing** (bool, optional): Whether to track execution timing (default: True)
   * **track_tensor_ops** (bool, optional): Whether to track tensor operations (default: True)

   **Returns:** ComputationalGraphTracker - Tracker instance with execution data

analyze_computational_graph_execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: analyze_computational_graph_execution(model, input_tensor, detailed=True)

   Analyze the computational graph of a PyTorch model execution (alias for analyze_computational_graph).

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to analyze
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **detailed** (bool, optional): Whether to include detailed analysis (default: True)

   **Returns:** dict - Dictionary containing computational graph analysis

visualize_computational_graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: visualize_computational_graph(model, input_tensor, renderer='plotly')

   Visualize the computational graph of a PyTorch model execution.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to visualize
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **renderer** (str, optional): Rendering backend ('plotly' or 'matplotlib', default: 'plotly')

   **Returns:** Visualization object (Plotly figure or Matplotlib figure)

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

   **Example:**
   
   .. code-block:: python

      from pytorch-graph import visualize_computational_graph

      fig = visualize_computational_graph(
          model=model,
          input_tensor=input_tensor,
          renderer='plotly'
      )
      fig.show()

export_computational_graph
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: export_computational_graph(model, input_tensor, filepath, format='json')

   Export the computational graph of a PyTorch model execution to a file.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to export
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **filepath** (str): Output file path
   * **format** (str, optional): Export format ('json', default: 'json')

   **Returns:** str - Path to the exported file

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

   **Example:**
   
   .. code-block:: python

      from pytorch-graph import export_computational_graph

      filepath = export_computational_graph(
          model=model,
          input_tensor=input_tensor,
          filepath='graph.json'
      )
      print(f"Graph exported to: {filepath}")

save_computational_graph_png
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: save_computational_graph_png(model, input_tensor, filepath="computational_graph.png", width=1200, height=800, dpi=300, show_legend=True, node_size=20, font_size=10)

   Save the computational graph as a high-quality PNG image.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to visualize
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **filepath** (str, optional): Output PNG file path (default: "computational_graph.png")
   * **width** (int, optional): Image width in pixels (default: 1200)
   * **height** (int, optional): Image height in pixels (default: 800)
   * **dpi** (int, optional): Dots per inch for high resolution (default: 300)
   * **show_legend** (bool, optional): Whether to show legend (default: True)
   * **node_size** (int, optional): Size of nodes in the graph (default: 20)
   * **font_size** (int, optional): Font size for labels (default: 10)

   **Returns:** str - Path to the saved PNG file

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

   **Example:**
   
   .. code-block:: python

      from pytorch-graph import save_computational_graph_png

      png_path = save_computational_graph_png(
          model=model,
          input_tensor=input_tensor,
          filepath="graph.png",
          width=1600,
          height=1200,
          dpi=300
      )
      print(f"PNG saved to: {png_path}")

ComputationalGraphTracker Class
-------------------------------

.. class:: ComputationalGraphTracker(model, track_memory=True, track_timing=True, track_tensor_ops=True)

   Tracks the computational graph of PyTorch model execution.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to track
   * **track_memory** (bool, optional): Whether to track memory usage (default: True)
   * **track_timing** (bool, optional): Whether to track execution timing (default: True)
   * **track_tensor_ops** (bool, optional): Whether to track tensor operations (default: True)

   **Methods:**

   .. method:: start_tracking()

      Start tracking the computational graph.

      **Returns:** None

   .. method:: stop_tracking()

      Stop tracking the computational graph.

      **Returns:** None

   .. method:: get_graph_summary()

      Get a summary of the computational graph.

      **Returns:** dict - Dictionary containing graph summary with keys:
      
      * **total_nodes** (int): Total number of nodes in the graph
      * **total_edges** (int): Total number of edges in the graph
      * **execution_time** (float): Total execution time in seconds
      * **memory_usage** (str): Memory usage information
      * **operation_types** (dict): Count of each operation type
      * **model_size_mb** (float): Model size in megabytes

   .. method:: get_graph_data()

      Get the complete graph data for visualization.

      **Returns:** dict - Dictionary containing:
      
      * **nodes** (list): List of GraphNode objects as dictionaries
      * **edges** (list): List of GraphEdge objects as dictionaries

   .. method:: export_graph(filepath, format='json')

      Export the computational graph to a file.

      **Parameters:**
      
      * **filepath** (str): Output file path
      * **format** (str, optional): Export format ('json', default: 'json')

      **Returns:** None

   .. method:: visualize_graph(renderer='plotly')

      Visualize the computational graph.

      **Parameters:**
      
      * **renderer** (str, optional): Rendering backend ('plotly' or 'matplotlib', default: 'plotly')

      **Returns:** Visualization object (Plotly figure or Matplotlib figure)

   .. method:: save_graph_png(filepath, width=1200, height=800, dpi=300, show_legend=True, node_size=20, font_size=10)

      Save the computational graph as a PNG image with enhanced visualization.

      **Parameters:**
      
      * **filepath** (str): Output PNG file path
      * **width** (int, optional): Image width in pixels (default: 1200)
      * **height** (int, optional): Image height in pixels (default: 800)
      * **dpi** (int, optional): Dots per inch for high resolution (default: 300)
      * **show_legend** (bool, optional): Whether to show legend (default: True)
      * **node_size** (int, optional): Size of nodes in the graph (default: 20)
      * **font_size** (int, optional): Font size for labels (default: 10)

      **Returns:** str - Path to the saved PNG file

GraphNode Class
---------------

.. class:: GraphNode(id, name, operation_type, module_name=None, input_shapes=None, output_shapes=None, parameters=None, execution_time=None, memory_usage=None, metadata=None, parent_ids=None, child_ids=None, timestamp=None)

   Represents a node in the computational graph.

   **Parameters:**
   
   * **id** (str): Unique identifier for the node
   * **name** (str): Name of the operation
   * **operation_type** (OperationType): Type of operation
   * **module_name** (str, optional): Name of the PyTorch module
   * **input_shapes** (list, optional): List of input tensor shapes
   * **output_shapes** (list, optional): List of output tensor shapes
   * **parameters** (dict, optional): Operation parameters
   * **execution_time** (float, optional): Execution time in seconds
   * **memory_usage** (int, optional): Memory usage in bytes
   * **metadata** (dict, optional): Additional metadata
   * **parent_ids** (list, optional): List of parent node IDs
   * **child_ids** (list, optional): List of child node IDs
   * **timestamp** (float, optional): Timestamp of execution

GraphEdge Class
---------------

.. class:: GraphEdge(source_id, target_id, edge_type, tensor_shape=None, metadata=None)

   Represents an edge in the computational graph.

   **Parameters:**
   
   * **source_id** (str): ID of the source node
   * **target_id** (str): ID of the target node
   * **edge_type** (str): Type of edge (e.g., 'data_flow', 'gradient_flow')
   * **tensor_shape** (tuple, optional): Shape of the tensor flowing through the edge
   * **metadata** (dict, optional): Additional metadata

OperationType Enum
------------------

.. class:: OperationType

   Types of operations that can be tracked.

   **Values:**
   
   * **FORWARD**: Forward pass operation
   * **BACKWARD**: Backward pass operation
   * **TENSOR_OP**: Tensor operation (add, multiply, etc.)
   * **LAYER_OP**: Layer operation
   * **GRADIENT_OP**: Gradient operation
   * **MEMORY_OP**: Memory operation
   * **CUSTOM**: Custom operation

Examples
--------

Basic Computational Graph Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

   # Get summary
   summary = tracker.get_graph_summary()
   print(f"Total operations: {summary['total_nodes']}")
   print(f"Execution time: {summary['execution_time']:.4f}s")

Advanced Usage with ComputationalGraphTracker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

   # Save visualization
   tracker.save_graph_png(
       "advanced_graph.png",
       width=2000,
       height=1500,
       dpi=300,
       show_legend=True,
       node_size=30,
       font_size=14
   )

Graph Analysis
~~~~~~~~~~~~~~

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
~~~~~~~~~~~

.. code-block:: python

   # Export to JSON
   tracker.export_graph("graph_data.json")

   # Load and inspect exported data
   import json
   with open("graph_data.json", 'r') as f:
       graph_data = json.load(f)

   print(f"Nodes: {len(graph_data['nodes'])}")
   print(f"Edges: {len(graph_data['edges'])}")

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   from pytorch-graph import visualize_computational_graph

   # Create interactive visualization
   fig = visualize_computational_graph(
       model=model,
       input_tensor=input_tensor,
       renderer='plotly'
   )
   fig.show()

   # Save as PNG
   from pytorch-graph import save_computational_graph_png

   png_path = save_computational_graph_png(
       model=model,
       input_tensor=input_tensor,
       filepath="computational_graph.png",
       width=1600,
       height=1200,
       dpi=300
   )

Error Handling
--------------

The functions will raise appropriate exceptions for:

* **ImportError**: If PyTorch is not available
* **RuntimeError**: If model execution fails
* **ValueError**: If invalid parameters are provided
* **FileNotFoundError**: If output directory doesn't exist

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`model_analysis` - For model analysis functions
* :doc:`utils` - For utility classes and functions
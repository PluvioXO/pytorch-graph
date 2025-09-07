Main Module API
===============

This module provides the main public API for PyTorch Graph. All functions and classes are available directly from the `pytorch-graph` package.

Core Functions
--------------

Architecture Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch-graph.generate_architecture_diagram

.. autofunction:: pytorch-graph.save_architecture_diagram

.. autofunction:: pytorch-graph.generate_research_paper_diagram

.. autofunction:: pytorch-graph.generate_flowchart_diagram

.. autofunction:: pytorch-graph.visualize

.. autofunction:: pytorch-graph.visualize_model

.. autofunction:: pytorch-graph.compare_models

.. autofunction:: pytorch-graph.create_architecture_report

Model Analysis
~~~~~~~~~~~~~~

.. autofunction:: pytorch-graph.analyze_model

.. autofunction:: pytorch-graph.profile_model

.. autofunction:: pytorch-graph.extract_activations

Computational Graph Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch-graph.track_computational_graph

.. autofunction:: pytorch-graph.analyze_computational_graph

.. autofunction:: pytorch-graph.track_computational_graph_execution

.. autofunction:: pytorch-graph.analyze_computational_graph_execution

.. autofunction:: pytorch-graph.visualize_computational_graph

.. autofunction:: pytorch-graph.export_computational_graph

.. autofunction:: pytorch-graph.save_computational_graph_png

Classes
-------

Core Classes
~~~~~~~~~~~~

.. autoclass:: pytorch-graph.PyTorchVisualizer
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.ComputationalGraphTracker
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.ModelAnalyzer
   :members:
   :undoc-members:

Utility Classes
~~~~~~~~~~~~~~~

.. autoclass:: pytorch-graph.LayerInfo
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.PositionCalculator
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.HookManager
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.ActivationExtractor
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.FeatureMapExtractor
   :members:
   :undoc-members:

Data Classes
~~~~~~~~~~~~

.. autoclass:: pytorch-graph.GraphNode
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.GraphEdge
   :members:
   :undoc-members:

.. autoclass:: pytorch-graph.OperationType
   :members:
   :undoc-members:

Enums
~~~~~

.. autoclass:: pytorch-graph.OperationType
   :members:
   :undoc-members:

Complete API Reference
----------------------

Architecture Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

generate_architecture_diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: generate_architecture_diagram(model, input_shape, output_path="architecture.png", title=None, format="png", style="flowchart")

   Generate an enhanced flowchart architecture diagram from a PyTorch model and save as PNG.

   **Parameters:**
   
   * **model** (torch.nn.Module): The PyTorch model to visualize
   * **input_shape** (tuple): The input tensor shape for the model
   * **output_path** (str, optional): Output file path (default: "architecture.png")
   * **title** (str, optional): Diagram title (auto-generated if None)
   * **format** (str, optional): Output format ('png' or 'txt', default: "png")
   * **style** (str, optional): Diagram style ('flowchart', 'standard', or 'research_paper', default: "flowchart")

   **Returns:** str - Path to the generated diagram file

save_architecture_diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: save_architecture_diagram(model, input_shape, output_path="architecture.png", **kwargs)

   Generate and save an enhanced flowchart architecture diagram (alias for generate_architecture_diagram).

   **Parameters:**
   
   * **model** (torch.nn.Module): The PyTorch model to visualize
   * **input_shape** (tuple): The input tensor shape for the model
   * **output_path** (str, optional): Output file path (default: "architecture.png")
   * **\*\*kwargs**: Additional arguments passed to generate_architecture_diagram

   **Returns:** str - Path to the generated diagram file

generate_research_paper_diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: generate_research_paper_diagram(model, input_shape, output_path="model_architecture_paper.png", title=None)

   Generate a research paper quality architecture diagram.

   **Parameters:**
   
   * **model** (torch.nn.Module): The PyTorch model to visualize
   * **input_shape** (tuple): The input tensor shape for the model
   * **output_path** (str, optional): Output file path (default: "model_architecture_paper.png")
   * **title** (str, optional): Diagram title (auto-generated if None)

   **Returns:** str - Path to the generated diagram file

generate_flowchart_diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: generate_flowchart_diagram(model, input_shape, output_path="model_flowchart.png", title=None)

   Generate a clean flowchart-style architecture diagram with vertical flow.

   **Parameters:**
   
   * **model** (torch.nn.Module): The PyTorch model to visualize
   * **input_shape** (tuple): The input tensor shape for the model
   * **output_path** (str, optional): Output file path (default: "model_flowchart.png")
   * **title** (str, optional): Diagram title (auto-generated if None)

   **Returns:** str - Path to the generated diagram file

visualize
^^^^^^^^^

.. function:: visualize(model, input_shape=None, renderer='plotly', **kwargs)

   Visualize a PyTorch model in 3D.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to visualize
   * **input_shape** (tuple, optional): Input tensor shape (if None will try to infer)
   * **renderer** (str, optional): Rendering backend ('plotly' or 'matplotlib', default: 'plotly')
   * **\*\*kwargs**: Additional visualization parameters

   **Returns:** Visualization object (Plotly figure or Matplotlib figure)

visualize_model
^^^^^^^^^^^^^^^

.. function:: visualize_model(model, input_shape=None, renderer='plotly', **kwargs)

   Alias for visualize() function for backward compatibility.

   **Parameters:** Same as :func:`visualize`

   **Returns:** Visualization object

compare_models
^^^^^^^^^^^^^^

.. function:: compare_models(models, names=None, input_shapes=None, renderer='plotly', **kwargs)

   Compare multiple PyTorch models in a single visualization.

   **Parameters:**
   
   * **models** (list): List of PyTorch models
   * **names** (list, optional): Optional list of model names
   * **input_shapes** (list, optional): Optional list of input shapes for each model
   * **renderer** (str, optional): Rendering backend ('plotly' or 'matplotlib', default: 'plotly')
   * **\*\*kwargs**: Additional visualization parameters

   **Returns:** Comparison visualization object

create_architecture_report
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: create_architecture_report(model, input_shape=None, output_path="pytorch-graph_report.html")

   Create a comprehensive HTML report of the PyTorch model architecture.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to analyze
   * **input_shape** (tuple, optional): Input tensor shape
   * **output_path** (str, optional): Path for the output HTML file (default: "pytorch-graph_report.html")

   **Returns:** None

Model Analysis Functions
~~~~~~~~~~~~~~~~~~~~~~~~

analyze_model
^^^^^^^^^^^^^

.. function:: analyze_model(model, input_shape=None, detailed=True)

   Analyze a PyTorch model and return detailed statistics.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to analyze
   * **input_shape** (tuple, optional): Input tensor shape
   * **detailed** (bool, optional): Whether to include detailed layer analysis (default: True)

   **Returns:** dict - Dictionary containing model analysis

profile_model
^^^^^^^^^^^^^

.. function:: profile_model(model, input_shape, device='cpu')

   Profile a PyTorch model for performance analysis.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to profile
   * **input_shape** (tuple): Input tensor shape
   * **device** (str, optional): Device to run profiling on ('cpu' or 'cuda', default: 'cpu')

   **Returns:** dict - Profiling results dictionary

extract_activations
^^^^^^^^^^^^^^^^^^^

.. function:: extract_activations(model, input_tensor, layer_names=None)

   Extract intermediate activations from a PyTorch model.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to extract activations from
   * **input_tensor** (torch.Tensor): Input tensor for forward pass
   * **layer_names** (list, optional): Specific layer names to extract (if None, extracts all)

   **Returns:** dict - Dictionary of layer names to activation tensors

Computational Graph Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

track_computational_graph
^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: track_computational_graph(model, input_tensor, track_memory=True, track_timing=True, track_tensor_ops=True)

   Track the computational graph of a PyTorch model execution.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to track
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **track_memory** (bool, optional): Whether to track memory usage (default: True)
   * **track_timing** (bool, optional): Whether to track execution timing (default: True)
   * **track_tensor_ops** (bool, optional): Whether to track tensor operations (default: True)

   **Returns:** ComputationalGraphTracker - Tracker instance with execution data

analyze_computational_graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: analyze_computational_graph(model, input_tensor, detailed=True)

   Analyze the computational graph of a PyTorch model execution.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to analyze
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **detailed** (bool, optional): Whether to include detailed analysis (default: True)

   **Returns:** dict - Dictionary containing computational graph analysis

track_computational_graph_execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: analyze_computational_graph_execution(model, input_tensor, detailed=True)

   Analyze the computational graph of a PyTorch model execution (alias for analyze_computational_graph).

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to analyze
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **detailed** (bool, optional): Whether to include detailed analysis (default: True)

   **Returns:** dict - Dictionary containing computational graph analysis

visualize_computational_graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: visualize_computational_graph(model, input_tensor, renderer='plotly')

   Visualize the computational graph of a PyTorch model execution.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to visualize
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **renderer** (str, optional): Rendering backend ('plotly' or 'matplotlib', default: 'plotly')

   **Returns:** Visualization object (Plotly figure or Matplotlib figure)

export_computational_graph
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: export_computational_graph(model, input_tensor, filepath, format='json')

   Export the computational graph of a PyTorch model execution to a file.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to export
   * **input_tensor** (torch.Tensor): Input tensor for the forward pass
   * **filepath** (str): Output file path
   * **format** (str, optional): Export format ('json', default: 'json')

   **Returns:** str - Path to the exported file

save_computational_graph_png
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Quick Start Examples
--------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch-graph import (
       generate_architecture_diagram,
       analyze_model,
       track_computational_graph
   )

   # Create a simple model
   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   # Generate architecture diagram
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="model_architecture.png",
       title="My Neural Network"
   )

   # Analyze model
   analysis = analyze_model(model, input_shape=(1, 784))
   print(f"Total parameters: {analysis['basic_info']['total_parameters']:,}")

   # Track computational graph
   input_tensor = torch.randn(1, 784, requires_grad=True)
   tracker = track_computational_graph(model, input_tensor)
   tracker.save_graph_png("computational_graph.png")

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch-graph import (
       PyTorchVisualizer,
       ComputationalGraphTracker,
       ModelAnalyzer
   )

   # Create visualizer
   visualizer = PyTorchVisualizer(
       renderer='plotly',
       layout_style='hierarchical',
       spacing=2.5
   )

   # 3D visualization
   fig = visualizer.visualize(
       model=model,
       input_shape=(1, 784),
       show_parameters=True,
       show_activations=True
   )
   fig.show()

   # Comprehensive analysis
   analyzer = ModelAnalyzer()
   analysis = analyzer.analyze(model, input_shape=(1, 784), detailed=True)

   # Computational graph tracking
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

   # Get summary
   summary = tracker.get_graph_summary()
   print(f"Operations: {summary['total_nodes']:,}")
   print(f"Execution time: {summary['execution_time']:.4f}s")

Error Handling
--------------

All functions will raise appropriate exceptions for:

* **ImportError**: If PyTorch is not available
* **RuntimeError**: If model execution fails
* **ValueError**: If invalid parameters are provided
* **TypeError**: If model is not a PyTorch module
* **FileNotFoundError**: If output directory doesn't exist

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`computational_graph_tracking` - For computational graph analysis
* :doc:`model_analysis` - For model analysis functions
* :doc:`utils` - For utility classes and functions

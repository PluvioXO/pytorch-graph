Architecture Visualization API
==============================

This module provides comprehensive functions for generating professional architecture diagrams from PyTorch models.

Core Functions
--------------

.. autofunction:: pytorch_graph.generate_architecture_diagram

.. autofunction:: pytorch_graph.save_architecture_diagram

.. autofunction:: pytorch_graph.generate_research_paper_diagram

.. autofunction:: pytorch_graph.generate_flowchart_diagram

.. autofunction:: pytorch_graph.visualize

.. autofunction:: pytorch_graph.visualize_model

.. autofunction:: pytorch_graph.compare_models

.. autofunction:: pytorch_graph.create_architecture_report

Classes
-------

.. autoclass:: pytorch_graph.PyTorchVisualizer
   :members:
   :undoc-members:

Function Details
----------------

generate_architecture_diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

   **Raises:**
   
   * **ImportError**: If PyTorch is not available
   * **ValueError**: If unsupported format is specified

   **Example:**
   
   .. code-block:: python

      import torch.nn as nn
      from pytorch_graph import generate_architecture_diagram

      model = nn.Sequential(
          nn.Linear(784, 128),
          nn.ReLU(),
          nn.Linear(128, 10)
      )

      # Generate architecture diagram
      path = generate_architecture_diagram(
          model=model,
          input_shape=(1, 784),
          output_path="model_architecture.png",
          title="My Neural Network",
          style="flowchart"
      )
      print(f"Diagram saved to: {path}")

save_architecture_diagram
~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: save_architecture_diagram(model, input_shape, output_path="architecture.png", **kwargs)

   Generate and save an enhanced flowchart architecture diagram (alias for generate_architecture_diagram).

   **Parameters:**
   
   * **model** (torch.nn.Module): The PyTorch model to visualize
   * **input_shape** (tuple): The input tensor shape for the model
   * **output_path** (str, optional): Output file path (default: "architecture.png")
   * **\*\*kwargs**: Additional arguments passed to generate_architecture_diagram

   **Returns:** str - Path to the generated diagram file

generate_research_paper_diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: generate_research_paper_diagram(model, input_shape, output_path="model_architecture_paper.png", title=None)

   Generate a research paper quality architecture diagram.

   **Parameters:**
   
   * **model** (torch.nn.Module): The PyTorch model to visualize
   * **input_shape** (tuple): The input tensor shape for the model
   * **output_path** (str, optional): Output file path (default: "model_architecture_paper.png")
   * **title** (str, optional): Diagram title (auto-generated if None)

   **Returns:** str - Path to the generated diagram file

generate_flowchart_diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: generate_flowchart_diagram(model, input_shape, output_path="model_flowchart.png", title=None)

   Generate a clean flowchart-style architecture diagram with vertical flow.

   **Parameters:**
   
   * **model** (torch.nn.Module): The PyTorch model to visualize
   * **input_shape** (tuple): The input tensor shape for the model
   * **output_path** (str, optional): Output file path (default: "model_flowchart.png")
   * **title** (str, optional): Diagram title (auto-generated if None)

   **Returns:** str - Path to the generated diagram file

visualize
~~~~~~~~~

.. function:: visualize(model, input_shape=None, renderer='plotly', **kwargs)

   Visualize a PyTorch model in 3D.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to visualize
   * **input_shape** (tuple, optional): Input tensor shape (if None will try to infer)
   * **renderer** (str, optional): Rendering backend ('plotly' or 'matplotlib', default: 'plotly')
   * **\*\*kwargs**: Additional visualization parameters

   **Returns:** Visualization object (Plotly figure or Matplotlib figure)

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

visualize_model
~~~~~~~~~~~~~~~

.. function:: visualize_model(model, input_shape=None, renderer='plotly', **kwargs)

   Alias for visualize() function for backward compatibility.

   **Parameters:** Same as :func:`visualize`

   **Returns:** Visualization object

compare_models
~~~~~~~~~~~~~~

.. function:: compare_models(models, names=None, input_shapes=None, renderer='plotly', **kwargs)

   Compare multiple PyTorch models in a single visualization.

   **Parameters:**
   
   * **models** (list): List of PyTorch models
   * **names** (list, optional): Optional list of model names
   * **input_shapes** (list, optional): Optional list of input shapes for each model
   * **renderer** (str, optional): Rendering backend ('plotly' or 'matplotlib', default: 'plotly')
   * **\*\*kwargs**: Additional visualization parameters

   **Returns:** Comparison visualization object

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

create_architecture_report
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: create_architecture_report(model, input_shape=None, output_path="pytorch_graph_report.html")

   Create a comprehensive HTML report of the PyTorch model architecture.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to analyze
   * **input_shape** (tuple, optional): Input tensor shape
   * **output_path** (str, optional): Path for the output HTML file (default: "pytorch_graph_report.html")

   **Returns:** None

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

PyTorchVisualizer Class
-----------------------

.. class:: PyTorchVisualizer(renderer='plotly', layout_style='hierarchical', spacing=2.0, theme='plotly_dark', width=1200, height=800)

   Main class for visualizing PyTorch neural network architectures in 3D.

   **Parameters:**
   
   * **renderer** (str, optional): Rendering backend ('plotly' or 'matplotlib', default: 'plotly')
   * **layout_style** (str, optional): Layout algorithm ('hierarchical', 'circular', 'spring', 'custom', default: 'hierarchical')
   * **spacing** (float, optional): Spacing between layers (default: 2.0)
   * **theme** (str, optional): Color theme for visualization (default: 'plotly_dark')
   * **width** (int, optional): Figure width in pixels (default: 1200)
   * **height** (int, optional): Figure height in pixels (default: 800)

   **Methods:**

   .. method:: visualize(model, input_shape=None, title=None, show_connections=True, show_labels=True, show_parameters=False, show_activations=False, optimize_layout=True, device='auto', export_path=None, **kwargs)

      Visualize a PyTorch model.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model to visualize
      * **input_shape** (tuple, optional): Input tensor shape (required for detailed analysis)
      * **title** (str, optional): Plot title
      * **show_connections** (bool, optional): Whether to show connections between layers (default: True)
      * **show_labels** (bool, optional): Whether to show layer labels (default: True)
      * **show_parameters** (bool, optional): Whether to show parameter count visualization (default: False)
      * **show_activations** (bool, optional): Whether to include activation statistics (default: False)
      * **optimize_layout** (bool, optional): Whether to optimize layer positions (default: True)
      * **device** (str, optional): Device for model analysis ('auto', 'cpu', 'cuda', default: 'auto')
      * **export_path** (str, optional): Path to export the visualization (optional)
      * **\*\*kwargs**: Additional rendering options

      **Returns:** Rendered visualization object

   .. method:: get_model_summary(model, input_shape=None, device='auto')

      Get a comprehensive summary of the PyTorch model.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model
      * **input_shape** (tuple, optional): Input tensor shape
      * **device** (str, optional): Device for analysis (default: 'auto')

      **Returns:** dict - Dictionary containing model summary

   .. method:: analyze_model(model, input_shape=None, device='auto', detailed=True)

      Perform comprehensive analysis of the PyTorch model.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model
      * **input_shape** (tuple, optional): Input tensor shape
      * **device** (str, optional): Device for analysis (default: 'auto')
      * **detailed** (bool, optional): Whether to include detailed analysis (default: True)

      **Returns:** dict - Dictionary containing comprehensive analysis

   .. method:: profile_model(model, input_shape, device='cpu', num_runs=100)

      Profile PyTorch model performance.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model
      * **input_shape** (tuple): Input tensor shape
      * **device** (str, optional): Device for profiling (default: 'cpu')
      * **num_runs** (int, optional): Number of timing runs (default: 100)

      **Returns:** dict - Dictionary with profiling results

   .. method:: compare_models(models, names=None, input_shapes=None, device='auto', **kwargs)

      Compare multiple PyTorch models in a single visualization.

      **Parameters:**
      
      * **models** (list): List of PyTorch models
      * **names** (list, optional): Optional list of model names
      * **input_shapes** (list, optional): Optional list of input shapes for each model
      * **device** (str, optional): Device for analysis (default: 'auto')
      * **\*\*kwargs**: Additional visualization options

      **Returns:** Rendered comparison visualization

   .. method:: visualize_feature_maps(model, input_tensor, layer_names=None, max_channels=16)

      Visualize feature maps from convolutional layers.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch CNN model
      * **input_tensor** (torch.Tensor): Input tensor for feature extraction
      * **layer_names** (list, optional): Specific conv layers to visualize
      * **max_channels** (int, optional): Maximum channels per layer to visualize (default: 16)

      **Returns:** Feature map visualization

   .. method:: create_training_visualization(model, input_shape, num_epochs=1)

      Create visualizations showing model changes during training.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model
      * **input_shape** (tuple): Input tensor shape
      * **num_epochs** (int, optional): Number of training epochs to simulate (default: 1)

      **Returns:** list - List of visualizations for each epoch

   .. method:: export_architecture_report(model, input_shape=None, output_path="pytorch_report.html", include_profiling=True)

      Export a comprehensive HTML report of the PyTorch model architecture.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model
      * **input_shape** (tuple, optional): Input tensor shape
      * **output_path** (str, optional): Path for the output HTML file (default: "pytorch_report.html")
      * **include_profiling** (bool, optional): Whether to include performance profiling (default: True)

      **Returns:** None

   .. method:: set_theme(theme)

      Set the visualization theme.

      **Parameters:**
      
      * **theme** (str): Theme name

      **Returns:** None

   .. method:: set_layout_style(layout_style)

      Set the layout style for positioning layers.

      **Parameters:**
      
      * **layout_style** (str): Layout style name

      **Returns:** None

   .. method:: set_spacing(spacing)

      Set the spacing between layers.

      **Parameters:**
      
      * **spacing** (float): Spacing value

      **Returns:** None

Examples
--------

Basic Architecture Diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   from pytorch_graph import generate_architecture_diagram

   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   # Generate architecture diagram
   path = generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="model_architecture.png",
       title="My Neural Network",
       style="flowchart"
   )
   print(f"Diagram saved to: {path}")

Research Paper Quality Diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import generate_research_paper_diagram

   path = generate_research_paper_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="paper_architecture.png",
       title="Research Model Architecture"
   )

3D Visualization
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import visualize

   fig = visualize(
       model=model,
       input_shape=(1, 784),
       renderer='plotly',
       title="3D Model Visualization",
       show_parameters=True,
       show_activations=True
   )
   fig.show()

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import compare_models

   models = [mlp_model, cnn_model, resnet_model]
   names = ["MLP", "CNN", "ResNet"]
   input_shapes = [(1, 784), (1, 3, 32, 32), (1, 3, 224, 224)]

   fig = compare_models(
       models=models,
       names=names,
       input_shapes=input_shapes,
       renderer='plotly'
   )
   fig.show()

Comprehensive Report
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import create_architecture_report

   create_architecture_report(
       model=model,
       input_shape=(1, 784),
       output_path="my_model_report.html"
   )

Advanced Usage with PyTorchVisualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import PyTorchVisualizer

   # Create visualizer with custom settings
   visualizer = PyTorchVisualizer(
       renderer='plotly',
       layout_style='hierarchical',
       spacing=2.5,
       theme='plotly_white',
       width=1400,
       height=900
   )

   # Visualize with advanced options
   fig = visualizer.visualize(
       model=model,
       input_shape=(1, 784),
       title="Advanced Model Visualization",
       show_connections=True,
       show_labels=True,
       show_parameters=True,
       show_activations=True,
       optimize_layout=True,
       device='auto'
   )

   # Get model analysis
   analysis = visualizer.analyze_model(model, input_shape=(1, 784), detailed=True)
   print(f"Total parameters: {analysis['basic_info']['total_parameters']:,}")

   # Profile performance
   profiling = visualizer.profile_model(model, input_shape=(1, 784), num_runs=50)
   print(f"Average inference time: {profiling['mean_time_ms']:.2f} ms")

Error Handling
--------------

The functions will raise appropriate exceptions for:

* **ImportError**: If PyTorch is not available
* **ValueError**: If unsupported format or style is specified
* **FileNotFoundError**: If output directory doesn't exist
* **RuntimeError**: If model execution fails

See Also
--------

* :doc:`computational_graph_tracking` - For computational graph analysis
* :doc:`model_analysis` - For model analysis functions
* :doc:`utils` - For utility classes and functions

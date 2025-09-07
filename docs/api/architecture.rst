Architecture Visualization API
==============================

This module provides functions for generating professional architecture diagrams of PyTorch models.

Core Functions
--------------

.. autofunction:: pytorch_graph.generate_architecture_diagram

.. autofunction:: pytorch_graph.generate_flowchart_diagram

.. autofunction:: pytorch_graph.generate_research_paper_diagram

.. autofunction:: pytorch_graph.generate_standard_diagram

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   from pytorch_graph import generate_architecture_diagram

   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="model.png",
       title="My Model"
   )

Multiple Styles
~~~~~~~~~~~~~~~

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

Custom Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="custom.png",
       title="Custom Model",
       style="flowchart",
       dpi=300,
       show_legend=True
   )

CNN Example
~~~~~~~~~~~

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

   generate_architecture_diagram(
       model=cnn_model,
       input_shape=(1, 3, 32, 32),
       output_path="cnn_architecture.png",
       title="CNN Architecture"
   )

Parameters
----------

All architecture diagram functions accept the following parameters:

* **model** (torch.nn.Module): The PyTorch model to visualize
* **input_shape** (tuple): The input shape for the model
* **output_path** (str): Path where to save the diagram
* **title** (str, optional): Title for the diagram
* **style** (str, optional): Style of the diagram ("flowchart", "research_paper", "standard")
* **dpi** (int, optional): Dots per inch for the output image
* **show_legend** (bool, optional): Whether to show the legend

Return Values
-------------

All functions return the path to the saved diagram file.

Error Handling
--------------

The functions will raise appropriate exceptions for:

* Invalid model types
* Incorrect input shapes
* File system errors
* Missing dependencies

See Also
--------

* :doc:`computational_graph_tracking` - For computational graph visualization
* :doc:`model_analysis` - For model analysis functions

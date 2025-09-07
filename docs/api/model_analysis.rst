Model Analysis API
==================

This module provides functions for analyzing PyTorch models and their computational graphs.

Core Functions
--------------

.. autofunction:: pytorch_graph.analyze_model

Examples
--------

Basic Model Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   from pytorch_graph import analyze_model

   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   # Analyze model structure
   analysis = analyze_model(model, input_shape=(1, 784))
   print(f"Total parameters: {analysis['total_parameters']:,}")
   print(f"Model size: {analysis['model_size_mb']:.2f} MB")

Parameters
----------

analyze_model
~~~~~~~~~~~~~

* **model** (torch.nn.Module): The PyTorch model to analyze
* **input_shape** (tuple, optional): The input shape for the model
* **detailed** (bool, optional): Whether to include detailed analysis

Returns: Dictionary containing model analysis results

Error Handling
--------------

The functions will raise appropriate exceptions for:

* Invalid model types
* Incorrect input shapes
* Missing dependencies

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`computational_graph_tracking` - For computational graph analysis

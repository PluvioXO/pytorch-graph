Model Analysis API
==================

This module provides comprehensive model analysis capabilities for PyTorch models.

Core Functions
--------------

.. autofunction:: pytorch-graph.analyze_model

.. autofunction:: pytorch-graph.profile_model

.. autofunction:: pytorch-graph.extract_activations

Classes
-------

.. autoclass:: pytorch-graph.ModelAnalyzer
   :members:
   :undoc-members:

Function Details
----------------

analyze_model
~~~~~~~~~~~~~

.. function:: analyze_model(model, input_shape=None, detailed=True)

   Analyze a PyTorch model and return detailed statistics.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to analyze
   * **input_shape** (tuple, optional): Input tensor shape
   * **detailed** (bool, optional): Whether to include detailed layer analysis (default: True)

   **Returns:** dict - Dictionary containing model analysis with keys:
   
   * **basic_info** (dict): Basic model information
     * **total_parameters** (int): Total number of parameters
     * **trainable_parameters** (int): Number of trainable parameters
     * **total_layers** (int): Total number of layers
     * **device** (str): Device the model is on
     * **model_mode** (str): Model mode (training/evaluation)
   * **memory** (dict): Memory usage information
     * **parameters_mb** (float): Parameter memory in MB
     * **activations_mb** (float): Activation memory in MB
     * **input_mb** (float): Input memory in MB
     * **total_memory_mb** (float): Total memory in MB
   * **architecture** (dict): Architecture analysis
     * **depth** (int): Network depth
     * **conv_layers** (int): Number of convolutional layers
     * **linear_layers** (int): Number of linear layers
     * **norm_layers** (int): Number of normalization layers
     * **patterns** (list): Detected architectural patterns

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

   **Example:**
   
   .. code-block:: python

      import torch.nn as nn
      from pytorch-graph import analyze_model

      model = nn.Sequential(
          nn.Linear(784, 128),
          nn.ReLU(),
          nn.Linear(128, 10)
      )

      # Analyze model structure
      analysis = analyze_model(model, input_shape=(1, 784))
      print(f"Total parameters: {analysis['basic_info']['total_parameters']:,}")
      print(f"Model size: {analysis['memory']['total_memory_mb']:.2f} MB")

profile_model
~~~~~~~~~~~~~

.. function:: profile_model(model, input_shape, device='cpu')

   Profile a PyTorch model for performance analysis.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to profile
   * **input_shape** (tuple): Input tensor shape
   * **device** (str, optional): Device to run profiling on ('cpu' or 'cuda', default: 'cpu')

   **Returns:** dict - Profiling results dictionary with keys:
   
   * **mean_time_ms** (float): Mean inference time in milliseconds
   * **std_time_ms** (float): Standard deviation of inference time
   * **min_time_ms** (float): Minimum inference time in milliseconds
   * **max_time_ms** (float): Maximum inference time in milliseconds
   * **fps** (float): Frames per second (throughput)
   * **memory_usage** (dict): Memory usage statistics
   * **execution_times** (list): List of individual execution times

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

   **Example:**
   
   .. code-block:: python

      from pytorch-graph import profile_model

      profiling_results = profile_model(
          model=model,
          input_shape=(1, 784),
          device='cpu'
      )

      print(f"Average inference time: {profiling_results['mean_time_ms']:.2f} ms")
      print(f"Throughput: {profiling_results['fps']:.1f} FPS")

extract_activations
~~~~~~~~~~~~~~~~~~~

.. function:: extract_activations(model, input_tensor, layer_names=None)

   Extract intermediate activations from a PyTorch model.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to extract activations from
   * **input_tensor** (torch.Tensor): Input tensor for forward pass
   * **layer_names** (list, optional): Specific layer names to extract (if None, extracts all)

   **Returns:** dict - Dictionary of layer names to activation tensors

   **Raises:**
   
   * **ImportError**: If PyTorch is not available

   **Example:**
   
   .. code-block:: python

      import torch
      from pytorch-graph import extract_activations

      input_tensor = torch.randn(1, 784)
      activations = extract_activations(
          model=model,
          input_tensor=input_tensor,
          layer_names=['0', '2']  # Extract from specific layers
      )

      for layer_name, activation in activations.items():
          print(f"{layer_name}: {activation.shape}")

ModelAnalyzer Class
-------------------

.. class:: ModelAnalyzer()

   Comprehensive model analysis and profiling class.

   **Methods:**

   .. method:: analyze(model, input_shape=None, detailed=True, device='auto')

      Perform comprehensive analysis of the PyTorch model.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model to analyze
      * **input_shape** (tuple, optional): Input tensor shape
      * **detailed** (bool, optional): Whether to include detailed analysis (default: True)
      * **device** (str, optional): Device for analysis ('auto', 'cpu', 'cuda', default: 'auto')

      **Returns:** dict - Dictionary containing comprehensive analysis

   .. method:: profile_model(model, input_shape, device='cpu', num_runs=100)

      Profile PyTorch model performance.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model to profile
      * **input_shape** (tuple): Input tensor shape
      * **device** (str, optional): Device for profiling (default: 'cpu')
      * **num_runs** (int, optional): Number of timing runs (default: 100)

      **Returns:** dict - Dictionary with profiling results

   .. method:: get_layer_info(model, input_shape=None)

      Get detailed information about each layer in the model.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model
      * **input_shape** (tuple, optional): Input tensor shape

      **Returns:** list - List of layer information dictionaries

   .. method:: calculate_memory_usage(model, input_shape)

      Calculate memory usage for the model.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model
      * **input_shape** (tuple): Input tensor shape

      **Returns:** dict - Memory usage information

   .. method:: detect_architecture_patterns(model)

      Detect common architectural patterns in the model.

      **Parameters:**
      
      * **model** (torch.nn.Module): PyTorch model

      **Returns:** list - List of detected patterns

Examples
--------

Basic Model Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   from pytorch-graph import analyze_model

   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(128, 64),
       nn.ReLU(),
       nn.Linear(64, 10)
   )

   # Analyze model structure
   analysis = analyze_model(model, input_shape=(1, 784))
   
   # Basic information
   basic_info = analysis['basic_info']
   print(f"Total parameters: {basic_info['total_parameters']:,}")
   print(f"Trainable parameters: {basic_info['trainable_parameters']:,}")
   print(f"Total layers: {basic_info['total_layers']}")
   
   # Memory usage
   memory_info = analysis['memory']
   print(f"Model size: {memory_info['total_memory_mb']:.2f} MB")
   
   # Architecture analysis
   arch_info = analysis['architecture']
   print(f"Network depth: {arch_info['depth']}")
   print(f"Linear layers: {arch_info['linear_layers']}")

Performance Profiling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch-graph import profile_model

   # Profile model performance
   profiling_results = profile_model(
       model=model,
       input_shape=(1, 784),
       device='cpu'
   )

   print(f"Average inference time: {profiling_results['mean_time_ms']:.2f} ms")
   print(f"Standard deviation: {profiling_results['std_time_ms']:.2f} ms")
   print(f"Min time: {profiling_results['min_time_ms']:.2f} ms")
   print(f"Max time: {profiling_results['max_time_ms']:.2f} ms")
   print(f"Throughput: {profiling_results['fps']:.1f} FPS")

Activation Extraction
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from pytorch-graph import extract_activations

   input_tensor = torch.randn(1, 784)
   
   # Extract activations from all layers
   all_activations = extract_activations(model, input_tensor)
   
   for layer_name, activation in all_activations.items():
       print(f"Layer {layer_name}: {activation.shape}")
   
   # Extract from specific layers
   specific_activations = extract_activations(
       model=model,
       input_tensor=input_tensor,
       layer_names=['0', '3', '5']  # First, fourth, and sixth layers
   )

Advanced Usage with ModelAnalyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch-graph import ModelAnalyzer

   # Create analyzer
   analyzer = ModelAnalyzer()

   # Comprehensive analysis
   analysis = analyzer.analyze(
       model=model,
       input_shape=(1, 784),
       detailed=True,
       device='auto'
   )

   # Get layer information
   layer_info = analyzer.get_layer_info(model, input_shape=(1, 784))
   for layer in layer_info:
       print(f"Layer: {layer['name']}, Type: {layer['type']}, Parameters: {layer['parameters']}")

   # Calculate memory usage
   memory_usage = analyzer.calculate_memory_usage(model, input_shape=(1, 784))
   print(f"Parameter memory: {memory_usage['parameters_mb']:.2f} MB")
   print(f"Activation memory: {memory_usage['activations_mb']:.2f} MB")

   # Detect patterns
   patterns = analyzer.detect_architecture_patterns(model)
   print(f"Detected patterns: {patterns}")

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   def compare_models(models, input_shapes, names=None):
       """Compare multiple models."""
       if names is None:
           names = [f"Model_{i+1}" for i in range(len(models))]
       
       results = {}
       
       for name, (model, input_shape) in zip(names, zip(models, input_shapes)):
           # Analyze model
           analysis = analyze_model(model, input_shape=input_shape)
           
           # Profile performance
           profiling = profile_model(model, input_shape, device='cpu')
           
           results[name] = {
               'parameters': analysis['basic_info']['total_parameters'],
               'model_size': analysis['memory']['total_memory_mb'],
               'inference_time': profiling['mean_time_ms'],
               'throughput': profiling['fps']
           }
       
       # Print comparison
       print("Model Comparison:")
       print("-" * 50)
       for name, metrics in results.items():
           print(f"{name}:")
           print(f"  Parameters: {metrics['parameters']:,}")
           print(f"  Model Size: {metrics['model_size']:.2f} MB")
           print(f"  Inference Time: {metrics['inference_time']:.2f} ms")
           print(f"  Throughput: {metrics['throughput']:.1f} FPS")
           print()
       
       return results

   # Example usage
   models = [mlp_model, cnn_model, resnet_model]
   input_shapes = [(1, 784), (1, 3, 32, 32), (1, 3, 224, 224)]
   names = ["MLP", "CNN", "ResNet"]
   
   comparison_results = compare_models(models, input_shapes, names)

Memory Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_memory_usage(model, input_shape):
       """Analyze memory usage patterns."""
       analysis = analyze_model(model, input_shape=input_shape)
       memory_info = analysis['memory']
       
       print("Memory Analysis:")
       print(f"  Parameter memory: {memory_info['parameters_mb']:.2f} MB")
       print(f"  Activation memory: {memory_info['activations_mb']:.2f} MB")
       print(f"  Input memory: {memory_info['input_mb']:.2f} MB")
       print(f"  Total memory: {memory_info['total_memory_mb']:.2f} MB")
       
       # Memory efficiency
       param_ratio = memory_info['parameters_mb'] / memory_info['total_memory_mb']
       activation_ratio = memory_info['activations_mb'] / memory_info['total_memory_mb']
       
       print(f"  Parameter ratio: {param_ratio:.1%}")
       print(f"  Activation ratio: {activation_ratio:.1%}")
       
       return memory_info

   # Analyze memory
   memory_analysis = analyze_memory_usage(model, input_shape=(1, 784))

Layer-wise Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_layers(model, input_shape):
       """Get detailed layer analysis."""
       analyzer = ModelAnalyzer()
       layer_info = analyzer.get_layer_info(model, input_shape)
       
       print("Layer-wise Analysis:")
       print("-" * 30)
       
       for layer in layer_info:
           print(f"\nLayer: {layer['name']}")
           print(f"  Type: {layer['type']}")
           print(f"  Parameters: {layer['parameters']:,}")
           print(f"  Input shape: {layer.get('input_shape', 'N/A')}")
           print(f"  Output shape: {layer.get('output_shape', 'N/A')}")
           if 'memory_usage' in layer:
               print(f"  Memory: {layer['memory_usage']:.2f} MB")
       
       return layer_info

   # Analyze layers
   layer_analysis = analyze_layers(model, input_shape=(1, 784))

Error Handling
--------------

The functions will raise appropriate exceptions for:

* **ImportError**: If PyTorch is not available
* **RuntimeError**: If model execution fails
* **ValueError**: If invalid parameters are provided
* **TypeError**: If model is not a PyTorch module

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`computational_graph_tracking` - For computational graph analysis
* :doc:`utils` - For utility classes and functions
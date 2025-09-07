Utilities API
=============

This module provides utility functions and classes for PyTorch Graph.

Classes
-------

.. autoclass:: pytorch_graph.LayerInfo
   :members:
   :undoc-members:

.. autoclass:: pytorch_graph.PositionCalculator
   :members:
   :undoc-members:

.. autoclass:: pytorch_graph.HookManager
   :members:
   :undoc-members:

.. autoclass:: pytorch_graph.ActivationExtractor
   :members:
   :undoc-members:

.. autoclass:: pytorch_graph.FeatureMapExtractor
   :members:
   :undoc-members:

LayerInfo Class
---------------

.. class:: LayerInfo(name, layer_type, input_shape=None, output_shape=None, parameters=0, position=(0, 0, 0), metadata=None)

   Represents information about a neural network layer.

   **Parameters:**
   
   * **name** (str): Name of the layer
   * **layer_type** (str): Type of the layer (e.g., 'Linear', 'Conv2d', 'ReLU')
   * **input_shape** (tuple, optional): Input tensor shape
   * **output_shape** (tuple, optional): Output tensor shape
   * **parameters** (int, optional): Number of parameters in the layer (default: 0)
   * **position** (tuple, optional): 3D position for visualization (default: (0, 0, 0))
   * **metadata** (dict, optional): Additional metadata

   **Attributes:**
   
   * **name** (str): Layer name
   * **layer_type** (str): Layer type
   * **input_shape** (tuple): Input shape
   * **output_shape** (tuple): Output shape
   * **parameters** (int): Parameter count
   * **position** (tuple): 3D position
   * **metadata** (dict): Additional metadata

   **Methods:**

   .. method:: to_dict()

      Convert layer info to dictionary.

      **Returns:** dict - Dictionary representation of the layer info

   .. method:: from_dict(data)

      Create LayerInfo from dictionary.

      **Parameters:**
      
      * **data** (dict): Dictionary containing layer information

      **Returns:** LayerInfo - LayerInfo instance

   .. method:: get_memory_usage()

      Calculate memory usage for this layer.

      **Returns:** float - Memory usage in MB

PositionCalculator Class
------------------------

.. class:: PositionCalculator(layout_style='hierarchical', spacing=2.0)

   Calculates 3D positions for layers in neural network visualizations.

   **Parameters:**
   
   * **layout_style** (str, optional): Layout algorithm ('hierarchical', 'circular', 'spring', 'custom', default: 'hierarchical')
   * **spacing** (float, optional): Spacing between layers (default: 2.0)

   **Methods:**

   .. method:: calculate_positions(layers, connections)

      Calculate 3D positions for layers based on connections.

      **Parameters:**
      
      * **layers** (list): List of LayerInfo objects
      * **connections** (dict): Dictionary of layer connections

      **Returns:** list - List of LayerInfo objects with updated positions

   .. method:: optimize_positions(layers, connections)

      Optimize layer positions to minimize overlaps and improve readability.

      **Parameters:**
      
      * **layers** (list): List of LayerInfo objects
      * **connections** (dict): Dictionary of layer connections

      **Returns:** list - List of LayerInfo objects with optimized positions

   .. method:: set_layout_style(layout_style)

      Set the layout style.

      **Parameters:**
      
      * **layout_style** (str): Layout style name

      **Returns:** None

   .. method:: set_spacing(spacing)

      Set the spacing between layers.

      **Parameters:**
      
      * **spacing** (float): Spacing value

      **Returns:** None

HookManager Class
-----------------

.. class:: HookManager(model)

   Manages PyTorch hooks for model analysis.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to manage hooks for

   **Methods:**

   .. method:: register_forward_hooks(hook_fn)

      Register forward hooks for all modules.

      **Parameters:**
      
      * **hook_fn** (callable): Hook function to register

      **Returns:** list - List of hook handles

   .. method:: register_backward_hooks(hook_fn)

      Register backward hooks for all modules.

      **Parameters:**
      
      * **hook_fn** (callable): Hook function to register

      **Returns:** list - List of hook handles

   .. method:: remove_hooks()

      Remove all registered hooks.

      **Returns:** None

   .. method:: get_activations(input_tensor)

      Get activations from all layers.

      **Parameters:**
      
      * **input_tensor** (torch.Tensor): Input tensor

      **Returns:** dict - Dictionary of layer names to activations

ActivationExtractor Class
-------------------------

.. class:: ActivationExtractor(model)

   Extracts intermediate activations from PyTorch models.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch model to extract activations from

   **Methods:**

   .. method:: extract(input_tensor, layer_names=None)

      Extract activations from the model.

      **Parameters:**
      
      * **input_tensor** (torch.Tensor): Input tensor for forward pass
      * **layer_names** (list, optional): Specific layer names to extract (if None, extracts all)

      **Returns:** dict - Dictionary of layer names to activation tensors

   .. method:: register_hooks(layer_names=None)

      Register hooks for activation extraction.

      **Parameters:**
      
      * **layer_names** (list, optional): Specific layer names to hook

      **Returns:** None

   .. method:: remove_hooks()

      Remove all registered hooks.

      **Returns:** None

   .. method:: get_activation_stats(activations)

      Get statistics for extracted activations.

      **Parameters:**
      
      * **activations** (dict): Dictionary of activations

      **Returns:** dict - Statistics for each activation

FeatureMapExtractor Class
-------------------------

.. class:: FeatureMapExtractor(model)

   Extracts and visualizes feature maps from convolutional layers.

   **Parameters:**
   
   * **model** (torch.nn.Module): PyTorch CNN model

   **Attributes:**
   
   * **conv_layers** (list): List of convolutional layer names

   **Methods:**

   .. method:: extract_feature_maps(input_tensor, layer_names=None)

      Extract feature maps from convolutional layers.

      **Parameters:**
      
      * **input_tensor** (torch.Tensor): Input tensor
      * **layer_names** (list, optional): Specific conv layers to extract

      **Returns:** dict - Dictionary of layer names to feature maps

   .. method:: visualize_feature_maps(feature_maps, max_channels=16)

      Visualize feature maps as images.

      **Parameters:**
      
      * **feature_maps** (dict): Dictionary of feature maps
      * **max_channels** (int, optional): Maximum channels per layer to visualize (default: 16)

      **Returns:** dict - Processed feature maps for visualization

   .. method:: get_feature_map_stats(feature_maps)

      Get statistics for feature maps.

      **Parameters:**
      
      * **feature_maps** (dict): Dictionary of feature maps

      **Returns:** dict - Statistics for each feature map

Examples
--------

Using LayerInfo
~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import LayerInfo

   # Create a layer info object
   layer = LayerInfo(
       name="linear_1",
       layer_type="Linear",
       input_shape=(1, 784),
       output_shape=(1, 128),
       parameters=100480,
       position=(0, 0, 0),
       metadata={"activation": "ReLU"}
   )

   print(f"Layer: {layer.name}")
   print(f"Type: {layer.layer_type}")
   print(f"Parameters: {layer.parameters:,}")
   print(f"Memory usage: {layer.get_memory_usage():.2f} MB")

   # Convert to dictionary
   layer_dict = layer.to_dict()
   print(f"Dictionary: {layer_dict}")

   # Create from dictionary
   new_layer = LayerInfo.from_dict(layer_dict)

Using PositionCalculator
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import PositionCalculator, LayerInfo

   # Create position calculator
   calculator = PositionCalculator(
       layout_style='hierarchical',
       spacing=2.5
   )

   # Create some layers
   layers = [
       LayerInfo("input", "Input", output_shape=(1, 784)),
       LayerInfo("linear1", "Linear", input_shape=(1, 784), output_shape=(1, 128)),
       LayerInfo("relu1", "ReLU", input_shape=(1, 128), output_shape=(1, 128)),
       LayerInfo("linear2", "Linear", input_shape=(1, 128), output_shape=(1, 10))
   ]

   # Define connections
   connections = {
       "input": ["linear1"],
       "linear1": ["relu1"],
       "relu1": ["linear2"]
   }

   # Calculate positions
   positioned_layers = calculator.calculate_positions(layers, connections)

   for layer in positioned_layers:
       print(f"{layer.name}: {layer.position}")

   # Optimize positions
   optimized_layers = calculator.optimize_positions(positioned_layers, connections)

Using HookManager
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_graph import HookManager

   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   # Create hook manager
   hook_manager = HookManager(model)

   # Define hook function
   def forward_hook(module, input, output):
       print(f"Layer {module.__class__.__name__}: {output.shape}")

   # Register hooks
   hooks = hook_manager.register_forward_hooks(forward_hook)

   # Run forward pass
   input_tensor = torch.randn(1, 784)
   output = model(input_tensor)

   # Get activations
   activations = hook_manager.get_activations(input_tensor)

   # Remove hooks
   hook_manager.remove_hooks()

Using ActivationExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_graph import ActivationExtractor

   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   # Create activation extractor
   extractor = ActivationExtractor(model)

   # Extract activations
   input_tensor = torch.randn(1, 784)
   activations = extractor.extract(input_tensor)

   # Print activation shapes
   for layer_name, activation in activations.items():
       print(f"{layer_name}: {activation.shape}")

   # Get activation statistics
   stats = extractor.get_activation_stats(activations)
   for layer_name, stat in stats.items():
       print(f"{layer_name}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")

   # Extract from specific layers
   specific_activations = extractor.extract(
       input_tensor,
       layer_names=['0', '2']  # First and third layers
   )

Using FeatureMapExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_graph import FeatureMapExtractor

   # Create a CNN model
   cnn_model = nn.Sequential(
       nn.Conv2d(3, 32, 3, padding=1),
       nn.ReLU(),
       nn.Conv2d(32, 64, 3, padding=1),
       nn.ReLU(),
       nn.AdaptiveAvgPool2d((1, 1)),
       nn.Flatten(),
       nn.Linear(64, 10)
   )

   # Create feature map extractor
   extractor = FeatureMapExtractor(cnn_model)

   # Print available conv layers
   print(f"Convolutional layers: {extractor.conv_layers}")

   # Extract feature maps
   input_tensor = torch.randn(1, 3, 32, 32)
   feature_maps = extractor.extract_feature_maps(input_tensor)

   # Visualize feature maps
   processed_maps = extractor.visualize_feature_maps(
       feature_maps,
       max_channels=8
   )

   # Get feature map statistics
   stats = extractor.get_feature_map_stats(feature_maps)
   for layer_name, stat in stats.items():
       print(f"{layer_name}: {stat}")

Advanced Usage
~~~~~~~~~~~~~~

Custom Layer Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_model_layers(model, input_shape):
       """Comprehensive layer analysis."""
       from pytorch_graph import LayerInfo, ActivationExtractor
       
       # Create layer info objects
       layers = []
       for name, module in model.named_modules():
           if len(list(module.children())) == 0:  # Leaf modules
               layer = LayerInfo(
                   name=name,
                   layer_type=module.__class__.__name__,
                   parameters=sum(p.numel() for p in module.parameters())
               )
               layers.append(layer)
       
       # Extract activations to get shapes
       extractor = ActivationExtractor(model)
       input_tensor = torch.randn(1, *input_shape)
       activations = extractor.extract(input_tensor)
       
       # Update layer info with shapes
       for layer in layers:
           if layer.name in activations:
               activation = activations[layer.name]
               layer.output_shape = activation.shape
               layer.metadata = {
                   'activation_mean': float(activation.mean()),
                   'activation_std': float(activation.std())
               }
       
       return layers

   # Analyze model
   layers = analyze_model_layers(model, input_shape=(784,))
   
   for layer in layers:
       print(f"Layer: {layer.name}")
       print(f"  Type: {layer.layer_type}")
       print(f"  Parameters: {layer.parameters:,}")
       print(f"  Output shape: {layer.output_shape}")
       if layer.metadata:
           print(f"  Activation mean: {layer.metadata['activation_mean']:.4f}")

Memory Usage Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_memory_usage(model, input_shape):
       """Analyze memory usage of model layers."""
       from pytorch_graph import LayerInfo, ActivationExtractor
       
       # Get layer information
       layers = []
       for name, module in model.named_modules():
           if len(list(module.children())) == 0:
               layer = LayerInfo(
                   name=name,
                   layer_type=module.__class__.__name__,
                   parameters=sum(p.numel() for p in module.parameters())
               )
               layers.append(layer)
       
       # Extract activations
       extractor = ActivationExtractor(model)
       input_tensor = torch.randn(1, *input_shape)
       activations = extractor.extract(input_tensor)
       
       # Calculate memory usage
       total_param_memory = 0
       total_activation_memory = 0
       
       for layer in layers:
           param_memory = layer.get_memory_usage()
           total_param_memory += param_memory
           
           if layer.name in activations:
               activation = activations[layer.name]
               activation_memory = activation.numel() * 4 / (1024 * 1024)  # 4 bytes per float32
               total_activation_memory += activation_memory
               
               print(f"{layer.name}:")
               print(f"  Parameter memory: {param_memory:.2f} MB")
               print(f"  Activation memory: {activation_memory:.2f} MB")
       
       print(f"\nTotal parameter memory: {total_param_memory:.2f} MB")
       print(f"Total activation memory: {total_activation_memory:.2f} MB")
       print(f"Total memory: {total_param_memory + total_activation_memory:.2f} MB")
       
       return {
           'parameter_memory': total_param_memory,
           'activation_memory': total_activation_memory,
           'total_memory': total_param_memory + total_activation_memory
       }

   # Analyze memory
   memory_analysis = analyze_memory_usage(model, input_shape=(784,))

Error Handling
--------------

The utility classes will raise appropriate exceptions for:

* **ImportError**: If PyTorch is not available
* **RuntimeError**: If model execution fails
* **ValueError**: If invalid parameters are provided
* **TypeError**: If model is not a PyTorch module

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`computational_graph_tracking` - For computational graph analysis
* :doc:`model_analysis` - For model analysis functions
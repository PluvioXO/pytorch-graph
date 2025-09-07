Model Analysis
==============

PyTorch Graph provides comprehensive model analysis capabilities for understanding your PyTorch models.

Overview
--------

The model analysis module allows you to:

* **Analyze Model Structure**: Parameter counts, layer information, and model size
* **Performance Metrics**: Execution time, memory usage, and operation counts
* **Layer-wise Analysis**: Detailed breakdown by layer and operation type
* **Model Comparison**: Compare different models side by side

Features
--------

* **Parameter Counting**: Total and trainable parameter counts
* **Memory Analysis**: Model size estimation and memory usage
* **Performance Metrics**: Execution timing and operation analysis
* **Layer Breakdown**: Detailed analysis by layer type
* **Model Complexity**: Automatic assessment of model complexity

Basic Usage
-----------

Analyze a simple model:

.. code-block:: python

   import torch.nn as nn
   from pytorch_graph import analyze_model

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
   print(f"Total parameters: {analysis['total_parameters']:,}")
   print(f"Trainable parameters: {analysis['trainable_parameters']:,}")
   print(f"Model size: {analysis['model_size_mb']:.2f} MB")

Advanced Analysis
-----------------

Get detailed analysis with computational graph tracking:

.. code-block:: python

   import torch
   from pytorch_graph import analyze_computational_graph

   input_tensor = torch.randn(1, 784, requires_grad=True)

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

Model Comparison
----------------

Compare different models:

.. code-block:: python

   def compare_models(models, input_shapes):
       results = {}
       
       for name, (model, input_shape) in models.items():
           # Analyze model structure
           model_analysis = analyze_model(model, input_shape=input_shape)
           
           # Analyze computational graph
           input_tensor = torch.randn(*input_shape, requires_grad=True)
           graph_analysis = analyze_computational_graph(model, input_tensor)
           
           results[name] = {
               'parameters': model_analysis['total_parameters'],
               'model_size': model_analysis['model_size_mb'],
               'operations': graph_analysis['summary']['total_nodes'],
               'execution_time': graph_analysis['summary']['execution_time']
           }
       
       # Print comparison
       print("Model Comparison:")
       print("-" * 50)
       for name, metrics in results.items():
           print(f"{name}:")
           print(f"  Parameters: {metrics['parameters']:,}")
           print(f"  Model Size: {metrics['model_size']:.2f} MB")
           print(f"  Operations: {metrics['operations']:,}")
           print(f"  Execution Time: {metrics['execution_time']:.4f}s")
           print()
       
       return results

   # Example usage
   models = {
       'MLP': (mlp_model, (1, 784)),
       'CNN': (cnn_model, (1, 3, 32, 32)),
       'ResNet': (resnet_model, (1, 3, 224, 224))
   }
   
   comparison_results = compare_models(models, input_shapes)

Performance Analysis
--------------------

Analyze model performance in detail:

.. code-block:: python

   from pytorch_graph import ComputationalGraphTracker

   def analyze_performance(model, input_tensor, num_runs=5):
       execution_times = []
       memory_usage = []
       
       for i in range(num_runs):
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
           
           summary = tracker.get_graph_summary()
           execution_times.append(summary['execution_time'])
           if summary['memory_usage']:
               memory_usage.append(summary['memory_usage'])
       
       # Calculate statistics
       avg_time = sum(execution_times) / len(execution_times)
       min_time = min(execution_times)
       max_time = max(execution_times)
       
       print(f"Performance Analysis ({num_runs} runs):")
       print(f"  Average execution time: {avg_time:.4f}s")
       print(f"  Min execution time: {min_time:.4f}s")
       print(f"  Max execution time: {max_time:.4f}s")
       
       if memory_usage:
           avg_memory = sum(memory_usage) / len(memory_usage)
           print(f"  Average memory usage: {avg_memory}")
       
       return {
           'execution_times': execution_times,
           'memory_usage': memory_usage,
           'average_time': avg_time,
           'min_time': min_time,
           'max_time': max_time
       }

   # Analyze performance
   input_tensor = torch.randn(1, 784, requires_grad=True)
   performance_results = analyze_performance(model, input_tensor)

Layer-wise Analysis
-------------------

Get detailed breakdown by layer:

.. code-block:: python

   def analyze_layers(model, input_tensor):
       tracker = track_computational_graph(model, input_tensor)
       analysis = analyze_computational_graph(model, input_tensor, detailed=True)
       
       if 'layer_analysis' in analysis:
           print("Layer-wise Analysis:")
           print("-" * 30)
           
           for layer_name, operations in analysis['layer_analysis'].items():
               print(f"\n{layer_name}:")
               print(f"  Operations: {len(operations)}")
               
               for op in operations:
                   print(f"    - {op['operation_type']}: {op['input_shapes']} -> {op['output_shapes']}")
       
       return analysis['layer_analysis']

   # Analyze layers
   layer_analysis = analyze_layers(model, input_tensor)

Memory Analysis
---------------

Analyze memory usage patterns:

.. code-block:: python

   def analyze_memory_usage(model, input_tensor):
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
       
       summary = tracker.get_graph_summary()
       
       print("Memory Analysis:")
       print(f"  Model size: {summary.get('model_size_mb', 'N/A')} MB")
       print(f"  Memory usage: {summary.get('memory_usage', 'N/A')}")
       print(f"  Execution time: {summary['execution_time']:.4f}s")
       
       return summary

   # Analyze memory
   memory_analysis = analyze_memory_usage(model, input_tensor)

Examples
--------

CNN Analysis
~~~~~~~~~~~~

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

   # Analyze CNN
   cnn_analysis = analyze_model(cnn_model, input_shape=(1, 3, 32, 32))
   print(f"CNN Parameters: {cnn_analysis['total_parameters']:,}")
   print(f"CNN Size: {cnn_analysis['model_size_mb']:.2f} MB")

ResNet Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   class ResNetBlock(nn.Module):
       def __init__(self, in_channels, out_channels):
           super().__init__()
           self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
           self.bn1 = nn.BatchNorm2d(out_channels)
           self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
           self.bn2 = nn.BatchNorm2d(out_channels)
           self.relu = nn.ReLU()
           
       def forward(self, x):
           residual = x
           out = self.relu(self.bn1(self.conv1(x)))
           out = self.bn2(self.conv2(out))
           out += residual
           return self.relu(out)

   class ResNetModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
           self.bn1 = nn.BatchNorm2d(64)
           self.relu = nn.ReLU()
           self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
           
           self.res_block1 = ResNetBlock(64, 64)
           self.res_block2 = ResNetBlock(64, 64)
           
           self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
           self.fc = nn.Linear(64, 1000)
           
       def forward(self, x):
           x = self.relu(self.bn1(self.conv1(x)))
           x = self.maxpool(x)
           x = self.res_block1(x)
           x = self.res_block2(x)
           x = self.avgpool(x)
           x = torch.flatten(x, 1)
           x = self.fc(x)
           return x

   resnet_model = ResNetModel()
   
   # Analyze ResNet
   resnet_analysis = analyze_model(resnet_model, input_shape=(1, 3, 224, 224))
   print(f"ResNet Parameters: {resnet_analysis['total_parameters']:,}")
   print(f"ResNet Size: {resnet_analysis['model_size_mb']:.2f} MB")

Best Practices
--------------

* **Use appropriate input shapes** that match your model's expected input
* **Analyze multiple runs** for performance metrics
* **Compare models** to understand trade-offs
* **Monitor memory usage** for large models
* **Export analysis data** for further processing

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'torch'**
   Install PyTorch: ``pip install torch``

**Memory issues with large models**
   Use smaller input tensors for initial testing

**Analysis takes too long**
   Consider using fewer runs or smaller models

**Missing analysis data**
   Ensure the model runs successfully with the given input

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`computational_graph_tracking` - For computational graph analysis
* :doc:`api/model_analysis` - For complete API reference

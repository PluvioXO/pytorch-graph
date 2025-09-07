Examples
========

This section provides comprehensive examples of using PyTorch Graph for various use cases.

Basic Examples
--------------

Simple MLP
~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from pytorch_graph import generate_architecture_diagram, track_computational_graph

   # Define a simple MLP
   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(128, 64),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(64, 10)
   )

   # Generate architecture diagram
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="mlp_architecture.png",
       title="Simple MLP Architecture"
   )

   # Track computational graph
   input_tensor = torch.randn(1, 784, requires_grad=True)
   tracker = track_computational_graph(model, input_tensor)
   tracker.save_graph_png("mlp_computational_graph.png")

CNN Example
~~~~~~~~~~~

.. code-block:: python

   # Define a CNN
   cnn_model = nn.Sequential(
       nn.Conv2d(3, 32, 3, padding=1),
       nn.BatchNorm2d(32),
       nn.ReLU(),
       nn.MaxPool2d(2),
       
       nn.Conv2d(32, 64, 3, padding=1),
       nn.BatchNorm2d(64),
       nn.ReLU(),
       nn.MaxPool2d(2),
       
       nn.Conv2d(64, 128, 3, padding=1),
       nn.BatchNorm2d(128),
       nn.ReLU(),
       nn.AdaptiveAvgPool2d((1, 1)),
       
       nn.Flatten(),
       nn.Linear(128, 256),
       nn.ReLU(),
       nn.Dropout(0.5),
       nn.Linear(256, 10)
   )

   # Generate architecture diagram
   generate_architecture_diagram(
       model=cnn_model,
       input_shape=(1, 3, 32, 32),
       output_path="cnn_architecture.png",
       title="CNN Architecture"
   )

   # Track computational graph
   input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
   tracker = track_computational_graph(cnn_model, input_tensor)
   tracker.save_graph_png("cnn_computational_graph.png")

Advanced Examples
-----------------

ResNet-like Model
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ResidualBlock(nn.Module):
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
           
           self.res_block1 = ResidualBlock(64, 64)
           self.res_block2 = ResidualBlock(64, 64)
           
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
   
   # Generate architecture diagram
   generate_architecture_diagram(
       model=resnet_model,
       input_shape=(1, 3, 224, 224),
       output_path="resnet_architecture.png",
       title="ResNet-like Architecture"
   )

   # Track computational graph
   input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
   tracker = track_computational_graph(resnet_model, input_tensor)
   tracker.save_graph_png("resnet_computational_graph.png")

Transformer-like Model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiHeadAttention(nn.Module):
       def __init__(self, d_model, num_heads):
           super().__init__()
           self.d_model = d_model
           self.num_heads = num_heads
           self.d_k = d_model // num_heads
           
           self.W_q = nn.Linear(d_model, d_model)
           self.W_k = nn.Linear(d_model, d_model)
           self.W_v = nn.Linear(d_model, d_model)
           self.W_o = nn.Linear(d_model, d_model)
           
       def forward(self, x):
           batch_size, seq_len, d_model = x.size()
           
           Q = self.W_q(x)
           K = self.W_k(x)
           V = self.W_v(x)
           
           # Simplified attention (without actual attention computation)
           attention_output = self.W_o(V)
           return attention_output

   class TransformerBlock(nn.Module):
       def __init__(self, d_model, num_heads):
           super().__init__()
           self.attention = MultiHeadAttention(d_model, num_heads)
           self.norm1 = nn.LayerNorm(d_model)
           self.norm2 = nn.LayerNorm(d_model)
           self.feed_forward = nn.Sequential(
               nn.Linear(d_model, d_model * 4),
               nn.ReLU(),
               nn.Linear(d_model * 4, d_model)
           )
           
       def forward(self, x):
           # Self-attention
           attn_output = self.attention(x)
           x = self.norm1(x + attn_output)
           
           # Feed forward
           ff_output = self.feed_forward(x)
           x = self.norm2(x + ff_output)
           
           return x

   class TransformerModel(nn.Module):
       def __init__(self, vocab_size, d_model, num_heads, num_layers):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
           
           self.transformer_blocks = nn.ModuleList([
               TransformerBlock(d_model, num_heads)
               for _ in range(num_layers)
           ])
           
           self.output_projection = nn.Linear(d_model, vocab_size)
           
       def forward(self, x):
           x = self.embedding(x)
           x = x + self.pos_encoding[:x.size(1)]
           
           for transformer_block in self.transformer_blocks:
               x = transformer_block(x)
           
           x = self.output_projection(x)
           return x

   transformer_model = TransformerModel(
       vocab_size=10000,
       d_model=512,
       num_heads=8,
       num_layers=6
   )
   
   # Generate architecture diagram
   generate_architecture_diagram(
       model=transformer_model,
       input_shape=(1, 100),  # batch_size, seq_len
       output_path="transformer_architecture.png",
       title="Transformer Architecture"
   )

   # Track computational graph
   input_tensor = torch.randint(0, 10000, (1, 100), requires_grad=True)
   tracker = track_computational_graph(transformer_model, input_tensor)
   tracker.save_graph_png("transformer_computational_graph.png")

Real-world Examples
-------------------

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   def compare_models(models, input_shapes, output_dir="model_comparison"):
       """Compare multiple models comprehensively."""
       import os
       import json
       
       os.makedirs(output_dir, exist_ok=True)
       results = {}
       
       for name, (model, input_shape) in models.items():
           print(f"Analyzing {name}...")
           
           # Architecture visualization
           generate_architecture_diagram(
               model=model,
               input_shape=input_shape,
               output_path=f"{output_dir}/{name}_architecture.png",
               title=f"{name} Architecture"
           )
           
           # Computational graph tracking
           input_tensor = torch.randn(*input_shape, requires_grad=True)
           tracker = track_computational_graph(model, input_tensor)
           
           tracker.save_graph_png(
               f"{output_dir}/{name}_computational_graph.png",
               width=1600,
               height=1200,
               dpi=300
           )
           
           # Analysis
           from pytorch_graph import analyze_model, analyze_computational_graph
           
           model_analysis = analyze_model(model, input_shape=input_shape)
           graph_analysis = analyze_computational_graph(model, input_tensor)
           
           results[name] = {
               'parameters': model_analysis['total_parameters'],
               'model_size': model_analysis['model_size_mb'],
               'operations': graph_analysis['summary']['total_nodes'],
               'execution_time': graph_analysis['summary']['execution_time']
           }
       
       # Save comparison results
       with open(f"{output_dir}/comparison_results.json", 'w') as f:
           json.dump(results, f, indent=2)
       
       # Print comparison
       print("\nModel Comparison Results:")
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
   models_to_compare = {
       'MLP': (mlp_model, (1, 784)),
       'CNN': (cnn_model, (1, 3, 32, 32)),
       'ResNet': (resnet_model, (1, 3, 224, 224))
   }
   
   comparison_results = compare_models(models_to_compare, input_shapes)

Training Loop Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def train_with_graph_tracking(model, dataloader, num_epochs=10, output_dir="training_graphs"):
       """Training loop with computational graph tracking."""
       import os
       os.makedirs(output_dir, exist_ok=True)
       
       for epoch in range(num_epochs):
           for batch_idx, (data, target) in enumerate(dataloader):
               # Track computational graph for first batch of each epoch
               if batch_idx == 0:
                   tracker = track_computational_graph(model, data)
                   
                   # Save graph for this epoch
                   tracker.save_graph_png(
                       f"{output_dir}/epoch_{epoch}_computational_graph.png",
                       width=1600,
                       height=1200,
                       dpi=300
                   )
                   
                   # Get performance metrics
                   summary = tracker.get_graph_summary()
                   print(f"Epoch {epoch}: {summary['total_nodes']} operations, "
                         f"{summary['execution_time']:.4f}s")
               
               # Your existing training code
               optimizer.zero_grad()
               output = model(data)
               loss = criterion(output, target)
               loss.backward()
               optimizer.step()

Research Paper Workflow
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def research_paper_workflow(model, input_shape, model_name, output_dir="research_figures"):
       """Complete workflow for research paper figures."""
       import os
       os.makedirs(output_dir, exist_ok=True)
       
       print(f"Generating research figures for {model_name}...")
       
       # Architecture diagram (research style)
       generate_architecture_diagram(
           model=model,
           input_shape=input_shape,
           output_path=f"{output_dir}/{model_name}_architecture_research.png",
           style="research_paper",
           title=f"{model_name} Architecture",
           dpi=300
       )
       
       # Standard architecture diagram
       generate_architecture_diagram(
           model=model,
           input_shape=input_shape,
           output_path=f"{output_dir}/{model_name}_architecture_flowchart.png",
           style="flowchart",
           title=f"{model_name} Architecture (Flowchart)",
           dpi=300
       )
       
       # Computational graph
       input_tensor = torch.randn(*input_shape, requires_grad=True)
       tracker = track_computational_graph(model, input_tensor)
       
       tracker.save_graph_png(
           f"{output_dir}/{model_name}_computational_graph.png",
           width=2000,
           height=1500,
           dpi=300,
           show_legend=True,
           node_size=25,
           font_size=12
       )
       
       # Analysis data
       from pytorch_graph import analyze_model, analyze_computational_graph
       
       model_analysis = analyze_model(model, input_shape=input_shape)
       graph_analysis = analyze_computational_graph(model, input_tensor, detailed=True)
       
       # Save analysis results
       analysis_data = {
           'model_analysis': model_analysis,
           'graph_analysis': graph_analysis
       }
       
       with open(f"{output_dir}/{model_name}_analysis.json", 'w') as f:
           json.dump(analysis_data, f, indent=2, default=str)
       
       print(f"Research figures generated for {model_name}")
       print(f"  - Architecture diagrams: 2 styles")
       print(f"  - Computational graph: 1 diagram")
       print(f"  - Analysis data: JSON export")

Performance Profiling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def profile_model_performance(model, input_tensor, num_runs=10):
       """Detailed performance profiling."""
       import time
       
       execution_times = []
       memory_usage = []
       
       for i in range(num_runs):
           start_time = time.time()
           
           tracker = track_computational_graph(
               model=model,
               input_tensor=input_tensor,
               track_memory=True,
               track_timing=True,
               track_tensor_ops=True
           )
           
           end_time = time.time()
           execution_times.append(end_time - start_time)
           
           summary = tracker.get_graph_summary()
           if summary['memory_usage']:
               memory_usage.append(summary['memory_usage'])
       
       # Calculate statistics
       avg_time = sum(execution_times) / len(execution_times)
       std_time = (sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5
       
       print(f"Performance Profiling ({num_runs} runs):")
       print(f"  Average execution time: {avg_time:.4f}s ± {std_time:.4f}s")
       print(f"  Min execution time: {min(execution_times):.4f}s")
       print(f"  Max execution time: {max(execution_times):.4f}s")
       
       if memory_usage:
           avg_memory = sum(memory_usage) / len(memory_usage)
           print(f"  Average memory usage: {avg_memory}")
       
       return {
           'execution_times': execution_times,
           'memory_usage': memory_usage,
           'statistics': {
               'average_time': avg_time,
               'std_time': std_time,
               'min_time': min(execution_times),
               'max_time': max(execution_times)
           }
       }

   # Example usage
   input_tensor = torch.randn(1, 784, requires_grad=True)
   performance_results = profile_model_performance(mlp_model, input_tensor)

Best Practices
--------------

* **Start with simple models** to understand the output format
* **Use appropriate input shapes** that match your model's expected input
* **Generate multiple styles** for different use cases
* **Export data** for offline analysis
* **Monitor memory usage** when working with large models
* **Use high DPI** for publication-quality output

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'torch'**
   Install PyTorch: ``pip install torch``

**ImportError: No module named 'matplotlib'**
   Install matplotlib: ``pip install matplotlib``

**Memory issues with large models**
   Use smaller input tensors or disable tensor operation tracking

**Slow rendering with complex graphs**
   Reduce DPI or use smaller canvas sizes

**File not found errors**
   Ensure output directories exist

See Also
--------

* :doc:`quickstart` - For getting started quickly
* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`computational_graph_tracking` - For computational graph analysis
* :doc:`model_analysis` - For model analysis functions
* :doc:`advanced_features` - For advanced customization options

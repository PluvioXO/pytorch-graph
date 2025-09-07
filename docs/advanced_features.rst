Advanced Features
=================

PyTorch Graph provides advanced features for power users and researchers.

Overview
--------

Advanced features include:

* **Custom Visualization Parameters**: Fine-tune output appearance
* **Data Export and Analysis**: Export graph data for offline analysis
* **Performance Optimization**: Optimize for large models and complex graphs
* **Integration with Workflows**: Seamless integration with existing PyTorch workflows
* **Custom Styling**: Advanced customization options

Custom Visualization Parameters
-------------------------------

Fine-tune your visualizations with custom parameters:

.. code-block:: python

   from pytorch_graph import ComputationalGraphTracker

   # Create tracker with custom settings
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

   # Save with custom parameters
   tracker.save_graph_png(
       filepath="custom_visualization.png",
       width=2000,           # Custom width
       height=1500,          # Custom height
       dpi=300,              # High DPI for publication
       show_legend=True,     # Show legend
       node_size=30,         # Node size
       font_size=14          # Font size
   )

High-Resolution Output
~~~~~~~~~~~~~~~~~~~~~~

For publication-quality images:

.. code-block:: python

   # Ultra-high resolution for large displays
   tracker.save_graph_png(
       filepath="ultra_hd_graph.png",
       width=4000,
       height=3000,
       dpi=300,
       node_size=50,
       font_size=20
   )

Data Export and Analysis
------------------------

Export graph data for offline analysis:

.. code-block:: python

   # Export complete graph data
   tracker.export_graph("complete_graph_data.json")

   # Load and analyze exported data
   import json
   with open("complete_graph_data.json", 'r') as f:
       graph_data = json.load(f)

   print(f"Total nodes: {len(graph_data['nodes'])}")
   print(f"Total edges: {len(graph_data['edges'])}")
   
   # Analyze node types
   node_types = {}
   for node in graph_data['nodes']:
       node_type = node['operation_type']
       node_types[node_type] = node_types.get(node_type, 0) + 1
   
   print("Node type distribution:")
   for node_type, count in node_types.items():
       print(f"  {node_type}: {count}")

Custom Analysis Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom analysis functions:

.. code-block:: python

   def analyze_graph_complexity(graph_data):
       """Analyze the complexity of a computational graph."""
       nodes = graph_data['nodes']
       edges = graph_data['edges']
       
       # Calculate metrics
       total_nodes = len(nodes)
       total_edges = len(edges)
       avg_connections = total_edges / total_nodes if total_nodes > 0 else 0
       
       # Find most connected nodes
       node_connections = {}
       for edge in edges:
           source = edge['source_id']
           target = edge['target_id']
           node_connections[source] = node_connections.get(source, 0) + 1
           node_connections[target] = node_connections.get(target, 0) + 1
       
       most_connected = max(node_connections.items(), key=lambda x: x[1]) if node_connections else None
       
       return {
           'total_nodes': total_nodes,
           'total_edges': total_edges,
           'average_connections': avg_connections,
           'most_connected_node': most_connected
       }

   # Use custom analysis
   complexity_analysis = analyze_graph_complexity(graph_data)
   print(f"Graph complexity: {complexity_analysis}")

Performance Optimization
------------------------

Optimize for large models and complex graphs:

.. code-block:: python

   # Optimize for large models
   tracker = ComputationalGraphTracker(
       model=large_model,
       track_memory=True,      # Keep memory tracking
       track_timing=True,      # Keep timing
       track_tensor_ops=False  # Disable for performance
   )

   # Use smaller input for testing
   test_input = torch.randn(1, 3, 224, 224, requires_grad=True)
   
   tracker.start_tracking()
   output = large_model(test_input)
   loss = output.sum()
   loss.backward()
   tracker.stop_tracking()

   # Save with optimized settings
   tracker.save_graph_png(
       "large_model_graph.png",
       width=3000,  # Larger canvas for complex graphs
       height=2000,
       dpi=200,     # Lower DPI for faster rendering
       node_size=20,
       font_size=10
   )

Memory-Efficient Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~

For memory-constrained environments:

.. code-block:: python

   # Minimal tracking for memory efficiency
   tracker = ComputationalGraphTracker(
       model=model,
       track_memory=False,     # Disable memory tracking
       track_timing=False,     # Disable timing
       track_tensor_ops=False  # Disable tensor operations
   )

   # Process in chunks for very large models
   def process_large_model_in_chunks(model, input_tensor, chunk_size=1000):
       tracker = ComputationalGraphTracker(model, track_memory=False)
       tracker.start_tracking()
       
       # Process model
       output = model(input_tensor)
       loss = output.sum()
       loss.backward()
       
       tracker.stop_tracking()
       return tracker

Integration with Workflows
--------------------------

Seamlessly integrate with existing PyTorch workflows:

.. code-block:: python

   def train_with_graph_tracking(model, dataloader, num_epochs=10):
       """Training loop with computational graph tracking."""
       for epoch in range(num_epochs):
           for batch_idx, (data, target) in enumerate(dataloader):
               # Track computational graph for first batch of each epoch
               if batch_idx == 0:
                   tracker = track_computational_graph(model, data)
                   
                   # Save graph for this epoch
                   tracker.save_graph_png(
                       f"epoch_{epoch}_computational_graph.png",
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

Model Comparison Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

Compare multiple models systematically:

.. code-block:: python

   def compare_models_comprehensive(models, input_shapes, output_dir="comparison"):
       """Comprehensive model comparison with visualizations."""
       import os
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
       
       return results

Custom Styling
--------------

Create custom visualization styles:

.. code-block:: python

   def create_custom_style_graph(tracker, output_path, style_config):
       """Create a graph with custom styling."""
       # This would be implemented in the library
       # For now, we use the standard method with custom parameters
       
       tracker.save_graph_png(
           filepath=output_path,
           width=style_config.get('width', 1600),
           height=style_config.get('height', 1200),
           dpi=style_config.get('dpi', 300),
           show_legend=style_config.get('show_legend', True),
           node_size=style_config.get('node_size', 25),
           font_size=style_config.get('font_size', 12)
       )

   # Custom style configuration
   custom_style = {
       'width': 2000,
       'height': 1500,
       'dpi': 300,
       'show_legend': True,
       'node_size': 30,
       'font_size': 14
   }
   
   create_custom_style_graph(tracker, "custom_style_graph.png", custom_style)

Batch Processing
----------------

Process multiple models in batch:

.. code-block:: python

   def batch_process_models(models, input_shapes, output_dir="batch_output"):
       """Process multiple models in batch."""
       import os
       os.makedirs(output_dir, exist_ok=True)
       
       for name, (model, input_shape) in models.items():
           print(f"Processing {name}...")
           
           # Create output subdirectory
           model_dir = os.path.join(output_dir, name)
           os.makedirs(model_dir, exist_ok=True)
           
           # Generate all visualizations
           generate_architecture_diagram(
               model=model,
               input_shape=input_shape,
               output_path=os.path.join(model_dir, "architecture.png"),
               title=f"{name} Architecture"
           )
           
           input_tensor = torch.randn(*input_shape, requires_grad=True)
           tracker = track_computational_graph(model, input_tensor)
           
           tracker.save_graph_png(
               os.path.join(model_dir, "computational_graph.png"),
               width=1600,
               height=1200,
               dpi=300
           )
           
           # Export data
           tracker.export_graph(os.path.join(model_dir, "graph_data.json"))
           
           print(f"Completed {name}")

Advanced Examples
-----------------

Research Paper Workflow
~~~~~~~~~~~~~~~~~~~~~~~

Complete workflow for research papers:

.. code-block:: python

   def research_paper_workflow(model, input_shape, model_name):
       """Complete workflow for research paper figures."""
       print(f"Generating research figures for {model_name}...")
       
       # Architecture diagram (research style)
       generate_architecture_diagram(
           model=model,
           input_shape=input_shape,
           output_path=f"{model_name}_architecture_research.png",
           style="research_paper",
           title=f"{model_name} Architecture",
           dpi=300
       )
       
       # Computational graph
       input_tensor = torch.randn(*input_shape, requires_grad=True)
       tracker = track_computational_graph(model, input_tensor)
       
       tracker.save_graph_png(
           f"{model_name}_computational_graph.png",
           width=2000,
           height=1500,
           dpi=300,
           show_legend=True,
           node_size=25,
           font_size=12
       )
       
       # Analysis data
       analysis = analyze_computational_graph(model, input_tensor, detailed=True)
       
       # Save analysis results
       with open(f"{model_name}_analysis.json", 'w') as f:
           json.dump(analysis, f, indent=2, default=str)
       
       print(f"Research figures generated for {model_name}")

Performance Profiling
~~~~~~~~~~~~~~~~~~~~~

Detailed performance profiling:

.. code-block:: python

   def profile_model_performance(model, input_tensor, num_runs=10):
       """Detailed performance profiling."""
       import time
       
       execution_times = []
       memory_usage = []
       
       for i in range(num_runs):
           start_time = time.time()
           
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

Best Practices
--------------

* **Use appropriate parameters** for your use case
* **Optimize for performance** when working with large models
* **Export data** for offline analysis
* **Batch process** multiple models efficiently
* **Monitor memory usage** in memory-constrained environments
* **Use high DPI** for publication-quality output

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Memory issues with large models**
   Use ``track_tensor_ops=False`` and smaller input tensors

**Slow rendering with complex graphs**
   Reduce DPI or use smaller canvas sizes

**Export file too large**
   Consider filtering the exported data

**Integration issues**
   Ensure proper error handling in your workflows

See Also
--------

* :doc:`architecture_visualization` - For architecture diagram generation
* :doc:`computational_graph_tracking` - For computational graph analysis
* :doc:`model_analysis` - For model analysis functions

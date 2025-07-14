# PyTorch Computational Graph Tracking

This module provides comprehensive tracking and visualization of PyTorch's computational graph, allowing you to see exactly what methods PyTorch calls when executing your model.

## Features

- **Complete Graph Tracking**: Track forward and backward passes through all layers
- **Tensor Operation Monitoring**: Monitor tensor operations (add, multiply, matmul, etc.)
- **Memory Usage Tracking**: Track GPU/CPU memory usage during execution
- **Performance Analysis**: Measure execution timing and operation frequencies
- **Interactive Visualization**: Visualize the computational graph with Plotly or Matplotlib
- **Data Export**: Export graph data to JSON format for further analysis

## Quick Start

```python
import torch
import torch.nn as nn
from torch_vis import track_computational_graph_execution, visualize_computational_graph

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Create input tensor
input_tensor = torch.randn(1, 10, requires_grad=True)

# Track the computational graph
tracker = track_computational_graph_execution(model, input_tensor)

# Get summary
summary = tracker.get_graph_summary()
print(f"Total operations: {summary['total_nodes']}")
print(f"Execution time: {summary['execution_time']:.4f}s")

# Visualize the graph
fig = visualize_computational_graph(model, input_tensor, renderer='plotly')
fig.show()
```

## API Reference

### `track_computational_graph_execution(model, input_tensor, track_memory=True, track_timing=True, track_tensor_ops=True)`

Track the computational graph of a PyTorch model execution.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to track
- `input_tensor` (torch.Tensor): Input tensor for the forward pass
- `track_memory` (bool): Whether to track memory usage (default: True)
- `track_timing` (bool): Whether to track execution timing (default: True)
- `track_tensor_ops` (bool): Whether to track tensor operations (default: True)

**Returns:**
- `ComputationalGraphTracker`: Tracker object with execution data

**Example:**
```python
tracker = track_computational_graph_execution(model, input_tensor)

# Get summary
summary = tracker.get_graph_summary()
print(f"Operations: {summary['total_nodes']}")

# Get detailed data
graph_data = tracker.get_graph_data()
print(f"Nodes: {len(graph_data['nodes'])}")
print(f"Edges: {len(graph_data['edges'])}")
```

### `analyze_computational_graph_execution(model, input_tensor, detailed=True)`

Analyze the computational graph with comprehensive metrics.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to analyze
- `input_tensor` (torch.Tensor): Input tensor for the forward pass
- `detailed` (bool): Whether to include detailed analysis (default: True)

**Returns:**
- `dict`: Dictionary containing computational graph analysis

**Example:**
```python
analysis = analyze_computational_graph_execution(model, input_tensor)

print(f"Total operations: {analysis['summary']['total_nodes']}")
print(f"Execution time: {analysis['summary']['execution_time']:.4f}s")

# Performance metrics
if 'performance' in analysis:
    perf = analysis['performance']
    print(f"Operations per second: {perf['operations_per_second']:.2f}")

# Layer-wise analysis
if 'layer_analysis' in analysis:
    for layer_name, operations in analysis['layer_analysis'].items():
        print(f"{layer_name}: {len(operations)} operations")
```

### `visualize_computational_graph(model, input_tensor, renderer='plotly')`

Visualize the computational graph as an interactive plot.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to visualize
- `input_tensor` (torch.Tensor): Input tensor for the forward pass
- `renderer` (str): Rendering backend ('plotly' or 'matplotlib')

**Returns:**
- Visualization object (Plotly figure or Matplotlib figure)

**Example:**
```python
# Create interactive visualization
fig = visualize_computational_graph(model, input_tensor, renderer='plotly')
fig.show()

# Save to HTML file
fig.write_html("computational_graph.html")

# Use matplotlib instead
fig = visualize_computational_graph(model, input_tensor, renderer='matplotlib')
fig.savefig("computational_graph.png")
```

### `export_computational_graph(model, input_tensor, filepath, format='json')`

Export the computational graph to a file.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to export
- `input_tensor` (torch.Tensor): Input tensor for the forward pass
- `filepath` (str): Output file path
- `format` (str): Export format ('json')

**Returns:**
- `str`: Path to the exported file

**Example:**
```python
filepath = export_computational_graph(model, input_tensor, "graph.json")
print(f"Graph exported to: {filepath}")

# Load and inspect the exported data
import json
with open(filepath, 'r') as f:
    graph_data = json.load(f)

print(f"Nodes: {len(graph_data['nodes'])}")
print(f"Edges: {len(graph_data['edges'])}")
```

## Data Structures

### GraphNode

Represents a node in the computational graph.

```python
@dataclass
class GraphNode:
    id: str                    # Unique node identifier
    name: str                  # Human-readable name
    operation_type: OperationType  # Type of operation
    module_name: Optional[str] = None  # Associated module name
    input_shapes: Optional[List[Tuple[int, ...]]] = None  # Input tensor shapes
    output_shapes: Optional[List[Tuple[int, ...]]] = None  # Output tensor shapes
    parameters: Optional[Dict[str, Any]] = None  # Module parameters
    execution_time: Optional[float] = None  # Execution time in seconds
    memory_usage: Optional[int] = None  # Memory usage in bytes
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    parent_ids: Optional[List[str]] = None  # Parent node IDs
    child_ids: Optional[List[str]] = None  # Child node IDs
    timestamp: Optional[float] = None  # Timestamp relative to start
```

### GraphEdge

Represents an edge in the computational graph.

```python
@dataclass
class GraphEdge:
    source_id: str  # Source node ID
    target_id: str  # Target node ID
    edge_type: str  # Type of edge (data_flow, gradient_flow, etc.)
    tensor_shape: Optional[Tuple[int, ...]] = None  # Tensor shape
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
```

### OperationType

Enumeration of operation types.

```python
class OperationType(Enum):
    FORWARD = "forward"        # Forward pass operations
    BACKWARD = "backward"      # Backward pass operations
    TENSOR_OP = "tensor_op"   # Tensor operations
    LAYER_OP = "layer_op"     # Layer operations
    GRADIENT_OP = "gradient_op"  # Gradient operations
    MEMORY_OP = "memory_op"   # Memory operations
    CUSTOM = "custom"          # Custom operations
```

## Advanced Usage

### Custom Tracking Options

```python
# Track with specific options
tracker = track_computational_graph_execution(
    model, 
    input_tensor,
    track_memory=True,      # Track memory usage
    track_timing=True,      # Track execution timing
    track_tensor_ops=False  # Disable tensor operation tracking for performance
)
```

### Manual Tracker Usage

```python
from torch_vis import ComputationalGraphTracker

# Create tracker manually
tracker = ComputationalGraphTracker(model, track_memory=True)

# Start tracking
tracker.start_tracking()

# Run your model
output = model(input_tensor)
loss = output.sum()
loss.backward()

# Stop tracking
tracker.stop_tracking()

# Get results
summary = tracker.get_graph_summary()
graph_data = tracker.get_graph_data()
```

### Performance Analysis

```python
# Get detailed performance analysis
analysis = analyze_computational_graph_execution(model, input_tensor, detailed=True)

# Access performance metrics
performance = analysis['performance']
print(f"Operations per second: {performance['operations_per_second']:.2f}")
print(f"Memory usage: {performance['memory_usage']}")

# Access layer-wise analysis
layer_analysis = analysis['layer_analysis']
for layer_name, operations in layer_analysis.items():
    print(f"\n{layer_name}:")
    for op in operations:
        print(f"  - {op['operation_type']}: {op['input_shapes']} -> {op['output_shapes']}")
```

## Visualization Examples

### Interactive Plotly Visualization

```python
import plotly.graph_objects as go

# Create visualization
fig = visualize_computational_graph(model, input_tensor, renderer='plotly')

# Customize the plot
fig.update_layout(
    title="PyTorch Computational Graph",
    width=1200,
    height=800
)

# Save as HTML
fig.write_html("computational_graph.html")

# Display in Jupyter notebook
fig.show()
```

### Matplotlib Visualization

```python
import matplotlib.pyplot as plt

# Create visualization
fig = visualize_computational_graph(model, input_tensor, renderer='matplotlib')

# Customize the plot
plt.title("PyTorch Computational Graph")
plt.tight_layout()

# Save as PNG
plt.savefig("computational_graph.png", dpi=300, bbox_inches='tight')
plt.show()
```

## Troubleshooting

### Common Issues

1. **ImportError: PyTorch not available**
   ```bash
   pip install torch
   ```

2. **ImportError: Plotly not available**
   ```bash
   pip install plotly
   ```

3. **ImportError: Matplotlib not available**
   ```bash
   pip install matplotlib
   ```

4. **Memory issues with large models**
   ```python
   # Disable tensor operation tracking for better performance
   tracker = track_computational_graph_execution(
       model, input_tensor, track_tensor_ops=False
   )
   ```

### Performance Tips

1. **Disable tensor operation tracking** for large models to improve performance
2. **Use smaller input tensors** for initial testing
3. **Monitor memory usage** when tracking large models
4. **Export graph data** for offline analysis of complex models

## Integration with Existing Code

The computational graph tracking can be easily integrated into existing PyTorch workflows:

```python
# Wrap your existing training loop
def train_with_tracking(model, dataloader, num_epochs=10):
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            # Track computational graph for first batch of each epoch
            if batch_idx == 0:
                tracker = track_computational_graph_execution(model, data)
                summary = tracker.get_graph_summary()
                print(f"Epoch {epoch}: {summary['total_nodes']} operations")
            
            # Your existing training code
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

This provides comprehensive tracking and visualization of PyTorch's computational graph, helping you understand exactly what methods are called during model execution. 
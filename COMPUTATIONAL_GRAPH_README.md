# PyTorch Computational Graph Tracking

**Complete computational graph analysis and visualization for PyTorch models**. Track every operation, visualize the complete graph structure, and analyze execution performance with professional-quality diagrams.

## 🚀 Enhanced Features

### 🔍 **Complete Graph Traversal**
- **Maximal Traversal**: Captures the entire computational graph without artificial limits
- **No Depth Restrictions**: Removes arbitrary depth and operation count limits
- **Full Operation Coverage**: Shows every operation in the computational graph
- **Cycle Detection**: Prevents infinite recursion while capturing complete structure

### 🎨 **Professional Visualization**
- **Full Method Names**: Displays complete operation names without truncation
- **Smart Arrow Positioning**: Arrows connect node edges properly without crossing over boxes
- **Compact Layout**: Eliminates gaps and breaks for continuous graph flow
- **Enhanced Spacing**: Optimized node positioning and spacing
- **High-Resolution Output**: Up to 300 DPI for publication quality

### 📊 **Comprehensive Analysis**
- **Memory Tracking**: Real-time memory usage monitoring
- **Execution Timing**: Performance analysis and timing
- **Tensor Operations**: Complete tensor operation tracking
- **Layer-wise Analysis**: Detailed breakdown by layer and operation type

## 🏃‍♂️ Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from pytorch-graph import track_computational_graph, ComputationalGraphTracker

# Create a model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Create input tensor
input_tensor = torch.randn(1, 10, requires_grad=True)

# Track the complete computational graph
tracker = track_computational_graph(
    model=model,
    input_tensor=input_tensor,
    track_memory=True,
    track_timing=True,
    track_tensor_ops=True
)

# Save high-quality computational graph
tracker.save_graph_png(
    filepath="complete_computational_graph.png",
    width=1600,
    height=1200,
    dpi=300,
    show_legend=True
)
```

### Advanced Usage

```python
# Create tracker with custom settings
tracker = ComputationalGraphTracker(
    model=model,
    track_memory=True,      # Track memory usage
    track_timing=True,      # Track execution timing
    track_tensor_ops=True   # Track tensor operations
)

# Start tracking
tracker.start_tracking()

# Run your model
output = model(input_tensor)
loss = output.sum()
loss.backward()

# Stop tracking
tracker.stop_tracking()

# Get comprehensive analysis
summary = tracker.get_graph_summary()
graph_data = tracker.get_graph_data()

print(f"Total operations: {summary['total_nodes']}")
print(f"Execution time: {summary['execution_time']:.4f}s")
print(f"Memory usage: {summary['memory_usage']}")
```

## 📋 API Reference

### Core Functions

#### `track_computational_graph(model, input_tensor, track_memory=True, track_timing=True, track_tensor_ops=True)`

Track the complete computational graph of a PyTorch model execution.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to track
- `input_tensor` (torch.Tensor): Input tensor for the forward pass
- `track_memory` (bool): Whether to track memory usage (default: True)
- `track_timing` (bool): Whether to track execution timing (default: True)
- `track_tensor_ops` (bool): Whether to track tensor operations (default: True)

**Returns:**
- `ComputationalGraphTracker`: Tracker object with complete execution data

**Example:**
```python
tracker = track_computational_graph(model, input_tensor)

# Get summary
summary = tracker.get_graph_summary()
print(f"Operations: {summary['total_nodes']}")

# Get detailed data
graph_data = tracker.get_graph_data()
print(f"Nodes: {len(graph_data['nodes'])}")
print(f"Edges: {len(graph_data['edges'])}")
```

#### `analyze_computational_graph(model, input_tensor, detailed=True)`

Analyze the computational graph with comprehensive metrics and performance data.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to analyze
- `input_tensor` (torch.Tensor): Input tensor for the forward pass
- `detailed` (bool): Whether to include detailed analysis (default: True)

**Returns:**
- `dict`: Dictionary containing comprehensive computational graph analysis

**Example:**
```python
analysis = analyze_computational_graph(model, input_tensor, detailed=True)

print(f"Total operations: {analysis['summary']['total_nodes']}")
print(f"Execution time: {analysis['summary']['execution_time']:.4f}s")

# Performance metrics
if 'performance' in analysis:
    perf = analysis['performance']
    print(f"Operations per second: {perf['operations_per_second']:.2f}")
    print(f"Memory usage: {perf['memory_usage']}")

# Layer-wise analysis
if 'layer_analysis' in analysis:
    for layer_name, operations in analysis['layer_analysis'].items():
        print(f"{layer_name}: {len(operations)} operations")
```

### Classes

#### `ComputationalGraphTracker`

Main class for tracking computational graphs with enhanced features.

**Methods:**

##### `save_graph_png(filepath, width=1200, height=800, dpi=300, show_legend=True, node_size=20, font_size=10)`

Save the computational graph as a high-quality PNG image.

**Parameters:**
- `filepath` (str): Output file path
- `width` (int): Image width in pixels (default: 1200)
- `height` (int): Image height in pixels (default: 800)
- `dpi` (int): Dots per inch for high resolution (default: 300)
- `show_legend` (bool): Whether to show legend (default: True)
- `node_size` (int): Size of nodes in the graph (default: 20)
- `font_size` (int): Font size for labels (default: 10)

**Returns:**
- `str`: Path to the saved PNG file

**Example:**
```python
# Save with custom parameters
tracker.save_graph_png(
    filepath="custom_graph.png",
    width=2000,             # Custom width
    height=1500,            # Custom height
    dpi=300,                # High DPI
    show_legend=True,       # Show legend
    node_size=25,           # Node size
    font_size=12            # Font size
)
```

##### `get_graph_summary()`

Get a summary of the computational graph.

**Returns:**
- `dict`: Summary containing total nodes, edges, operation types, execution time, and memory usage

##### `get_graph_data()`

Get the complete graph data for visualization and analysis.

**Returns:**
- `dict`: Complete graph data including nodes, edges, and summary

##### `export_graph(filepath, format='json')`

Export the computational graph to a file.

**Parameters:**
- `filepath` (str): Output file path
- `format` (str): Export format ('json')

**Example:**
```python
# Export to JSON
tracker.export_graph("graph_data.json")

# Load and inspect the exported data
import json
with open("graph_data.json", 'r') as f:
    graph_data = json.load(f)

print(f"Nodes: {len(graph_data['nodes'])}")
print(f"Edges: {len(graph_data['edges'])}")
```

## 🎯 Enhanced Visualization Features

### Complete Graph Display
- **No Truncation**: Full method and object names displayed
- **Smart Layout**: Compact positioning without empty depth levels
- **Proper Connections**: Arrows connect node edges without crossing over boxes
- **Professional Styling**: Enhanced color schemes and typography

### High-Quality Output
- **Publication Ready**: Up to 300 DPI for research papers
- **Customizable Size**: Adjustable width, height, and node sizes
- **Intelligent Legends**: Automatic positioning without overlap
- **Clean Typography**: Professional fonts and text formatting

## 📊 Data Structures

### GraphNode

Represents a node in the computational graph with enhanced information.

```python
@dataclass
class GraphNode:
    id: str                    # Unique node identifier
    name: str                  # Human-readable name (full, not truncated)
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

Enumeration of operation types with enhanced categorization.

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

## 🔬 Advanced Usage Examples

### Complete Model Analysis

```python
# Comprehensive analysis of a complex model
def analyze_complete_model(model, input_tensor):
    # Track the complete computational graph
    tracker = track_computational_graph(
        model=model,
        input_tensor=input_tensor,
        track_memory=True,
        track_timing=True,
        track_tensor_ops=True
    )
    
    # Save high-quality visualization
    tracker.save_graph_png(
        filepath="complete_model_graph.png",
        width=2000,
        height=1500,
        dpi=300,
        show_legend=True
    )
    
    # Get detailed analysis
    analysis = analyze_computational_graph(model, input_tensor, detailed=True)
    
    # Print comprehensive report
    print("=== COMPUTATIONAL GRAPH ANALYSIS ===")
    print(f"Total Operations: {analysis['summary']['total_nodes']:,}")
    print(f"Total Edges: {analysis['summary']['total_edges']:,}")
    print(f"Execution Time: {analysis['summary']['execution_time']:.4f}s")
    
    if 'performance' in analysis:
        perf = analysis['performance']
        print(f"Operations/Second: {perf['operations_per_second']:.2f}")
        print(f"Memory Usage: {perf['memory_usage']}")
    
    # Layer-wise breakdown
    if 'layer_analysis' in analysis:
        print("\n=== LAYER-WISE ANALYSIS ===")
        for layer_name, operations in analysis['layer_analysis'].items():
            print(f"\n{layer_name}:")
            for op in operations:
                print(f"  - {op['operation_type']}: {op['input_shapes']} -> {op['output_shapes']}")
    
    return analysis
```

### Training Loop Integration

```python
# Integrate with training loops
def train_with_complete_tracking(model, dataloader, num_epochs=10):
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
```

### Model Comparison

```python
# Compare computational graphs of different models
def compare_models(models, input_tensor):
    results = {}
    
    for name, model in models.items():
        # Track each model
        tracker = track_computational_graph(model, input_tensor)
        
        # Save visualization
        tracker.save_graph_png(f"{name}_computational_graph.png")
        
        # Get analysis
        analysis = analyze_computational_graph(model, input_tensor)
        
        results[name] = {
            'operations': analysis['summary']['total_nodes'],
            'execution_time': analysis['summary']['execution_time'],
            'memory_usage': analysis['summary']['memory_usage']
        }
    
    # Print comparison
    print("=== MODEL COMPARISON ===")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Operations: {metrics['operations']:,}")
        print(f"  Time: {metrics['execution_time']:.4f}s")
        print(f"  Memory: {metrics['memory_usage']}")
    
    return results
```

## 🛠️ Troubleshooting

### Common Issues

1. **Large Graph Visualization**
   ```python
   # For very large models, increase image size
   tracker.save_graph_png(
       filepath="large_model_graph.png",
       width=3000,
       height=2000,
       dpi=300
   )
   ```

2. **Memory Issues**
   ```python
   # Disable tensor operation tracking for better performance
   tracker = track_computational_graph(
       model, input_tensor, track_tensor_ops=False
   )
   ```

3. **Long Operation Names**
   ```python
   # The system automatically handles long names
   # No truncation occurs - full names are displayed
   ```

### Performance Tips

1. **Use appropriate image sizes** for your model complexity
2. **Disable tensor operation tracking** for very large models
3. **Export graph data** for offline analysis of complex models
4. **Use high DPI** for publication-quality output

## 🎨 Visualization Examples

### Complete Graph with Full Names
- **No Truncation**: All operation names displayed in full
- **Smart Layout**: Compact positioning without gaps
- **Professional Styling**: Enhanced colors and typography
- **Proper Connections**: Arrows connect node edges correctly

### High-Quality Output
- **Publication Ready**: 300 DPI for research papers
- **Customizable**: Adjustable dimensions and styling
- **Clean Layout**: Professional appearance
- **Comprehensive**: Shows complete computational structure

## 🔧 Integration

The enhanced computational graph tracking integrates seamlessly with existing PyTorch workflows:

```python
# Minimal integration
tracker = track_computational_graph(model, input_tensor)
tracker.save_graph_png("my_model_graph.png")

# Advanced integration
tracker = ComputationalGraphTracker(model, track_memory=True)
tracker.start_tracking()
# ... your existing code ...
tracker.stop_tracking()
analysis = tracker.get_graph_summary()
```

This provides comprehensive tracking and visualization of PyTorch's computational graph with professional-quality output and complete operation coverage.
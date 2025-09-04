"""
Computational Graph Tracker for PyTorch Models.

This module provides utilities to track and visualize the computational graph
of PyTorch models, including method calls, tensor operations, and execution flow.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from collections import defaultdict, deque
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import traceback


class OperationType(Enum):
    """Types of operations that can be tracked."""
    FORWARD = "forward"
    BACKWARD = "backward"
    TENSOR_OP = "tensor_op"
    LAYER_OP = "layer_op"
    GRADIENT_OP = "gradient_op"
    MEMORY_OP = "memory_op"

    
    CUSTOM = "custom"


@dataclass
class GraphNode:
    """Represents a node in the computational graph."""
    id: str
    name: str
    operation_type: OperationType
    module_name: Optional[str] = None
    input_shapes: Optional[List[Tuple[int, ...]]] = None
    output_shapes: Optional[List[Tuple[int, ...]]] = None
    parameters: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    parent_ids: Optional[List[str]] = None
    child_ids: Optional[List[str]] = None
    timestamp: Optional[float] = None


@dataclass
class GraphEdge:
    """Represents an edge in the computational graph."""
    source_id: str
    target_id: str
    edge_type: str
    tensor_shape: Optional[Tuple[int, ...]] = None
    metadata: Optional[Dict[str, Any]] = None


class ComputationalGraphTracker:
    """
    Tracks the computational graph of PyTorch model execution.
    
    This class provides comprehensive tracking of:
    - Forward and backward passes
    - Tensor operations
    - Layer computations
    - Memory usage
    - Execution timing
    - Data flow between operations
    """
    
    def __init__(self, model: nn.Module, track_memory: bool = True, 
                 track_timing: bool = True, track_tensor_ops: bool = True):
        """
        Initialize the computational graph tracker.
        
        Args:
            model: PyTorch model to track
            track_memory: Whether to track memory usage
            track_timing: Whether to track execution timing
            track_tensor_ops: Whether to track tensor operations
        """
        self.model = model
        self.track_memory = track_memory
        self.track_timing = track_timing
        self.track_tensor_ops = track_tensor_ops
        
        # Graph data structures
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.node_counter = 0
        
        # Tracking state
        self.is_tracking = False
        self.hooks = []
        self.original_methods = {}
        self.tensor_ops_tracked = set()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance tracking
        self.start_time = None
        self.memory_snapshots = []
        
    def start_tracking(self):
        """Start tracking the computational graph."""
        if self.is_tracking:
            return
            
        self.is_tracking = True
        self.start_time = time.time()
        
        # Register hooks for all modules
        self._register_module_hooks()
        
        # Hook into tensor operations if enabled
        if self.track_tensor_ops:
            self._hook_tensor_operations()
            
        # Track memory if enabled
        if self.track_memory:
            self._start_memory_tracking()
    
    def stop_tracking(self):
        """Stop tracking the computational graph."""
        if not self.is_tracking:
            return
            
        self.is_tracking = False
        
        # Remove all hooks
        self._remove_hooks()
        
        # Restore original methods
        self._restore_original_methods()
        
        # Stop memory tracking
        if self.track_memory:
            self._stop_memory_tracking()
    
    def _register_module_hooks(self):
        """Register hooks for all modules in the model."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                # Forward hook
                forward_hook = module.register_forward_hook(
                    self._create_forward_hook(name)
                )
                self.hooks.append(forward_hook)
                
                # Backward hook
                backward_hook = module.register_backward_hook(
                    self._create_backward_hook(name)
                )
                self.hooks.append(backward_hook)
    
    def _create_forward_hook(self, module_name: str):
        """Create a forward hook for a module."""
        def hook(module, input, output):
            if not self.is_tracking:
                return
                
            with self.lock:
                node_id = f"forward_{module_name}_{self.node_counter}"
                self.node_counter += 1
                
                # Extract shapes
                input_shapes = []
                if isinstance(input, (tuple, list)):
                    input_shapes = [tuple(i.shape) if hasattr(i, 'shape') else None for i in input]
                elif hasattr(input, 'shape'):
                    input_shapes = [tuple(input.shape)]
                
                output_shapes = []
                if isinstance(output, (tuple, list)):
                    output_shapes = [tuple(o.shape) if hasattr(o, 'shape') else None for o in output]
                elif hasattr(output, 'shape'):
                    output_shapes = [tuple(output.shape)]
                
                # Create node
                node = GraphNode(
                    id=node_id,
                    name=f"Forward: {module_name}",
                    operation_type=OperationType.FORWARD,
                    module_name=module_name,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    timestamp=time.time() - self.start_time if self.start_time else None,
                    metadata={
                        'module_type': type(module).__name__,
                        'module_parameters': sum(p.numel() for p in module.parameters()),
                        'input_count': len(input) if isinstance(input, (tuple, list)) else 1,
                        'output_count': len(output) if isinstance(output, (tuple, list)) else 1,
                    }
                )
                
                self.nodes[node_id] = node
                
                # Add edges from inputs to this node
                self._add_input_edges(node_id, input)
        
        return hook
    
    def _create_backward_hook(self, module_name: str):
        """Create a backward hook for a module."""
        def hook(module, grad_input, grad_output):
            if not self.is_tracking:
                return
                
            with self.lock:
                node_id = f"backward_{module_name}_{self.node_counter}"
                self.node_counter += 1
                
                # Extract gradient shapes
                grad_input_shapes = []
                if isinstance(grad_input, (tuple, list)):
                    grad_input_shapes = [tuple(g.shape) if hasattr(g, 'shape') else None for g in grad_input]
                elif hasattr(grad_input, 'shape'):
                    grad_input_shapes = [tuple(grad_input.shape)]
                
                grad_output_shapes = []
                if isinstance(grad_output, (tuple, list)):
                    grad_output_shapes = [tuple(g.shape) if hasattr(g, 'shape') else None for g in grad_output]
                elif hasattr(grad_output, 'shape'):
                    grad_output_shapes = [tuple(grad_output.shape)]
                
                # Create node
                node = GraphNode(
                    id=node_id,
                    name=f"Backward: {module_name}",
                    operation_type=OperationType.BACKWARD,
                    module_name=module_name,
                    input_shapes=grad_output_shapes,  # Gradients flow backward
                    output_shapes=grad_input_shapes,
                    timestamp=time.time() - self.start_time if self.start_time else None,
                    metadata={
                        'module_type': type(module).__name__,
                        'grad_input_count': len(grad_input) if isinstance(grad_input, (tuple, list)) else 1,
                        'grad_output_count': len(grad_output) if isinstance(grad_output, (tuple, list)) else 1,
                    }
                )
                
                self.nodes[node_id] = node
                
                # Add edges from gradient outputs to this node
                self._add_gradient_edges(node_id, grad_output)
        
        return hook
    
    def _hook_tensor_operations(self):
        """Hook into tensor operations to track them."""
        # Store original methods
        self.original_methods['tensor_add'] = torch.Tensor.__add__
        self.original_methods['tensor_mul'] = torch.Tensor.__mul__
        self.original_methods['tensor_matmul'] = torch.Tensor.__matmul__
        
        # Override tensor operations
        def tracked_add(self, other):
            if self.is_tracking:
                self._track_tensor_operation('add', self, other)
            return self.original_methods['tensor_add'](self, other)
        
        def tracked_mul(self, other):
            if self.is_tracking:
                self._track_tensor_operation('mul', self, other)
            return self.original_methods['tensor_mul'](self, other)
        
        def tracked_matmul(self, other):
            if self.is_tracking:
                self._track_tensor_operation('matmul', self, other)
            return self.original_methods['tensor_matmul'](self, other)
        
        # Apply overrides
        torch.Tensor.__add__ = tracked_add
        torch.Tensor.__mul__ = tracked_mul
        torch.Tensor.__matmul__ = tracked_matmul
    
    def _track_tensor_operation(self, op_name: str, tensor1: torch.Tensor, tensor2: torch.Tensor):
        """Track a tensor operation."""
        with self.lock:
            node_id = f"tensor_op_{op_name}_{self.node_counter}"
            self.node_counter += 1
            
            node = GraphNode(
                id=node_id,
                name=f"Tensor {op_name}",
                operation_type=OperationType.TENSOR_OP,
                input_shapes=[tuple(tensor1.shape), tuple(tensor2.shape)],
                timestamp=time.time() - self.start_time if self.start_time else None,
                metadata={
                    'operation': op_name,
                    'tensor1_dtype': str(tensor1.dtype),
                    'tensor2_dtype': str(tensor2.dtype),
                    'tensor1_device': str(tensor1.device),
                    'tensor2_device': str(tensor2.device),
                }
            )
            
            self.nodes[node_id] = node
    
    def _add_input_edges(self, node_id: str, inputs):
        """Add edges from input tensors to a node."""
        if isinstance(inputs, (tuple, list)):
            for i, input_tensor in enumerate(inputs):
                if hasattr(input_tensor, 'shape'):
                    edge = GraphEdge(
                        source_id=f"input_{i}",
                        target_id=node_id,
                        edge_type="data_flow",
                        tensor_shape=tuple(input_tensor.shape)
                    )
                    self.edges.append(edge)
        elif hasattr(inputs, 'shape'):
            edge = GraphEdge(
                source_id="input_0",
                target_id=node_id,
                edge_type="data_flow",
                tensor_shape=tuple(inputs.shape)
            )
            self.edges.append(edge)
    
    def _add_gradient_edges(self, node_id: str, grad_outputs):
        """Add edges from gradient outputs to a node."""
        if isinstance(grad_outputs, (tuple, list)):
            for i, grad_tensor in enumerate(grad_outputs):
                if hasattr(grad_tensor, 'shape'):
                    edge = GraphEdge(
                        source_id=f"grad_output_{i}",
                        target_id=node_id,
                        edge_type="gradient_flow",
                        tensor_shape=tuple(grad_tensor.shape)
                    )
                    self.edges.append(edge)
        elif hasattr(grad_outputs, 'shape'):
            edge = GraphEdge(
                source_id="grad_output_0",
                target_id=node_id,
                edge_type="gradient_flow",
                tensor_shape=tuple(grad_outputs.shape)
            )
            self.edges.append(edge)
    
    def _start_memory_tracking(self):
        """Start tracking memory usage."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def _stop_memory_tracking(self):
        """Stop tracking memory usage."""
        if torch.cuda.is_available():
            memory_stats = torch.cuda.memory_stats()
            self.memory_snapshots.append({
                'peak_allocated': memory_stats.get('allocated_bytes.all.peak', 0),
                'current_allocated': memory_stats.get('allocated_bytes.all.current', 0),
                'peak_reserved': memory_stats.get('reserved_bytes.all.peak', 0),
                'current_reserved': memory_stats.get('reserved_bytes.all.current', 0),
            })
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _restore_original_methods(self):
        """Restore original tensor methods."""
        if 'tensor_add' in self.original_methods:
            torch.Tensor.__add__ = self.original_methods['tensor_add']
        if 'tensor_mul' in self.original_methods:
            torch.Tensor.__mul__ = self.original_methods['tensor_mul']
        if 'tensor_matmul' in self.original_methods:
            torch.Tensor.__matmul__ = self.original_methods['tensor_matmul']
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the computational graph."""
        with self.lock:
            summary = {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'operation_types': defaultdict(int),
                'module_types': defaultdict(int),
                'execution_time': time.time() - self.start_time if self.start_time else None,
                'memory_usage': self.memory_snapshots[-1] if self.memory_snapshots else None,
            }
            
            # Count operation types
            for node in self.nodes.values():
                summary['operation_types'][node.operation_type.value] += 1
                if node.module_name:
                    module_type = node.metadata.get('module_type', 'Unknown') if node.metadata else 'Unknown'
                    summary['module_types'][module_type] += 1
            
            return summary
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get the complete graph data for visualization."""
        with self.lock:
            return {
                'nodes': [asdict(node) for node in self.nodes.values()],
                'edges': [asdict(edge) for edge in self.edges],
                'summary': self.get_graph_summary()
            }
    
    def export_graph(self, filepath: str, format: str = 'json'):
        """Export the computational graph to a file."""
        graph_data = self.get_graph_data()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(graph_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def visualize_graph(self, renderer: str = 'plotly') -> Any:
        """
        Visualize the computational graph.
        
        Args:
            renderer: Rendering backend ('plotly' or 'matplotlib')
            
        Returns:
            Visualization object
        """
        try:
            if renderer == 'plotly':
                return self._visualize_with_plotly()
            elif renderer == 'matplotlib':
                return self._visualize_with_matplotlib()
            else:
                raise ValueError(f"Unsupported renderer: {renderer}")
        except ImportError as e:
            raise ImportError(f"Required dependencies for {renderer} visualization not available: {e}")
    
    def save_graph_png(self, filepath: str, width: int = 1200, height: int = 800, 
                       dpi: int = 300, show_legend: bool = True, 
                       node_size: int = 20, font_size: int = 10) -> str:
        """
        Save the computational graph as a PNG image.
        
        Args:
            filepath: Output file path
            width: Image width in pixels
            height: Image height in pixels
            dpi: Dots per inch for high resolution
            show_legend: Whether to show legend
            node_size: Size of nodes in the graph
            font_size: Font size for labels
            
        Returns:
            Path to the saved PNG file
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import FancyBboxPatch
            import numpy as np
        except ImportError:
            raise ImportError("Matplotlib is required for PNG generation. Install with: pip install matplotlib")
        
        # Get graph data
        graph_data = self.get_graph_data()
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        print(f"Debug: Processing {len(nodes)} nodes and {len(edges)} edges")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=dpi)
        
        # Color scheme for different operation types
        colors = {
            'forward': '#1f77b4',    # Blue
            'backward': '#ff7f0e',   # Orange
            'tensor_op': '#2ca02c',  # Green
            'layer_op': '#d62728',   # Red
            'gradient_op': '#9467bd', # Purple
            'memory_op': '#8c564b',  # Brown
            'custom': '#e377c2'      # Pink
        }
        
        # Simple positioning - just place nodes in a grid
        positions = {}
        if nodes:
            # Calculate grid dimensions
            cols = min(5, len(nodes))  # Max 5 columns
            rows = (len(nodes) + cols - 1) // cols
            
            # Calculate spacing
            x_spacing = width / (cols + 1)
            y_spacing = height / (rows + 1)
            
            for i, node in enumerate(nodes):
                row = i // cols
                col = i % cols
                x = (col + 1) * x_spacing
                y = height - (row + 1) * y_spacing  # Flip Y axis
                positions[node['id']] = (x, y)
        
        print(f"Debug: Positioned {len(positions)} nodes")
        
        # Draw edges (simplified)
        edge_count = 0
        for edge in edges:
            if edge['source_id'] in positions and edge['target_id'] in positions:
                start_pos = positions[edge['source_id']]
                end_pos = positions[edge['target_id']]
                
                # Draw simple line
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                       'gray', alpha=0.5, linewidth=1)
                edge_count += 1
        
        print(f"Debug: Drew {edge_count} edges")
        
        # Draw nodes
        for node in nodes:
            if node['id'] in positions:
                x, y = positions[node['id']]
                op_type = node['operation_type']
                color = colors.get(op_type, '#7f7f7f')
                
                # Create node circle
                circle = plt.Circle((x, y), node_size/2, color=color, 
                                  alpha=0.8, edgecolor='black', linewidth=1)
                ax.add_patch(circle)
                
                # Add text label (shortened)
                label = node['name'][:15] + '...' if len(node['name']) > 15 else node['name']
                ax.text(x, y, label, ha='center', va='center', 
                       fontsize=font_size, weight='bold', color='white')
        
        # Set up the plot
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')
        
        # Add title
        plt.title('PyTorch Computational Graph', fontsize=16, weight='bold', pad=20)
        
        # Add summary text
        summary = graph_data['summary']
        summary_text = f"Operations: {summary['total_nodes']} | Edges: {summary['total_edges']}"
        if summary['execution_time']:
            summary_text += f" | Time: {summary['execution_time']:.4f}s"
        
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        print("Debug: Saving figure...")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Debug: PNG saved to {filepath}")
        return filepath
    
    def _visualize_with_plotly(self):
        """Create a Plotly visualization of the computational graph."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            raise ImportError("Plotly is required for visualization. Install with: pip install plotly")
        
        graph_data = self.get_graph_data()
        
        # Create node positions (simple layout)
        node_positions = {}
        operation_groups = defaultdict(list)
        
        for node in graph_data['nodes']:
            op_type = node['operation_type']
            operation_groups[op_type].append(node['id'])
        
        # Position nodes by operation type
        y_offset = 0
        for op_type, node_ids in operation_groups.items():
            for i, node_id in enumerate(node_ids):
                node_positions[node_id] = (i * 100, y_offset)
            y_offset += 200
        
        # Create edges
        edge_x = []
        edge_y = []
        for edge in graph_data['edges']:
            source_pos = node_positions.get(edge['source_id'], (0, 0))
            target_pos = node_positions.get(edge['target_id'], (0, 0))
            edge_x.extend([source_pos[0], target_pos[0], None])
            edge_y.extend([source_pos[1], target_pos[1], None])
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in graph_data['nodes']:
            pos = node_positions.get(node['id'], (0, 0))
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_text.append(f"{node['name']}<br>Type: {node['operation_type']}")
            node_colors.append(node['operation_type'])
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=[hash(color) % 20 for color in node_colors],
                colorscale='Viridis',
                line=dict(width=2, color='white')
            ),
            text=[node['name'] for node in graph_data['nodes']],
            textposition="middle center",
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title="PyTorch Computational Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _visualize_with_matplotlib(self):
        """Create a Matplotlib visualization of the computational graph."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            raise ImportError("Matplotlib and NetworkX are required for visualization. Install with: pip install matplotlib networkx")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes.values():
            G.add_node(node.id, **asdict(node))
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.source_id, edge.target_id, **asdict(edge))
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=8, font_weight='bold',
                arrows=True, edge_color='gray', arrowsize=20)
        
        plt.title("PyTorch Computational Graph")
        plt.tight_layout()
        
        return plt.gcf()


def track_computational_graph(model: nn.Module, input_tensor: torch.Tensor,
                            track_memory: bool = True, track_timing: bool = True,
                            track_tensor_ops: bool = True) -> ComputationalGraphTracker:
    """
    Track the computational graph of a PyTorch model execution.
    
    Args:
        model: PyTorch model to track
        input_tensor: Input tensor for the forward pass
        track_memory: Whether to track memory usage
        track_timing: Whether to track execution timing
        track_tensor_ops: Whether to track tensor operations
        
    Returns:
        ComputationalGraphTracker with the execution data
    """
    tracker = ComputationalGraphTracker(
        model, track_memory, track_timing, track_tensor_ops
    )
    
    try:
        # Start tracking
        tracker.start_tracking()
        
        # Run forward pass
        output = model(input_tensor)
        
        # Run backward pass if gradients are needed
        if input_tensor.requires_grad:
            if output.numel() > 1:
                loss = output.sum()
            else:
                loss = output
            loss.backward()
        
    finally:
        # Stop tracking
        tracker.stop_tracking()
    
    return tracker


def analyze_computational_graph(model: nn.Module, input_tensor: torch.Tensor,
                              detailed: bool = True) -> Dict[str, Any]:
    """
    Analyze the computational graph of a PyTorch model.
    
    Args:
        model: PyTorch model to analyze
        input_tensor: Input tensor for the forward pass
        detailed: Whether to include detailed analysis
        
    Returns:
        Dictionary containing computational graph analysis
    """
    tracker = track_computational_graph(model, input_tensor)
    
    analysis = {
        'summary': tracker.get_graph_summary(),
        'graph_data': tracker.get_graph_data() if detailed else None,
    }
    
    if detailed:
        # Additional detailed analysis
        analysis['performance'] = {
            'total_execution_time': analysis['summary']['execution_time'],
            'memory_usage': analysis['summary']['memory_usage'],
            'operations_per_second': len(tracker.nodes) / analysis['summary']['execution_time'] if analysis['summary']['execution_time'] else 0,
        }
        
        # Layer-wise analysis
        layer_analysis = defaultdict(list)
        for node in tracker.nodes.values():
            if node.module_name:
                layer_analysis[node.module_name].append({
                    'operation_type': node.operation_type.value,
                    'execution_time': node.execution_time,
                    'input_shapes': node.input_shapes,
                    'output_shapes': node.output_shapes,
                })
        
        analysis['layer_analysis'] = dict(layer_analysis)
    
    return analysis 
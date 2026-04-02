"""
Computational Graph Tracker for PyTorch Models.

This module provides utilities to track and visualize the computational graph
of PyTorch models, including method calls, tensor operations, and execution flow.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Mapping
from collections import defaultdict, deque
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import traceback

from .submission_styles import get_submission_style


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
        self.lock = threading.RLock()
        
        # Performance tracking
        self.start_time = None
        self.execution_time = None
        self.memory_snapshots = []
        self.graph_source = None
        
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

    def _reset_graph_data(self):
        """Reset tracked graph state before a fresh capture."""
        self.nodes = {}
        self.edges = []
        self.node_counter = 0
        self.execution_time = None
        self.memory_snapshots = []
        self.graph_source = None

    def _extract_tensor_shapes(self, value: Any) -> List[Tuple[int, ...]]:
        """Collect tensor shapes from nested tensor outputs."""
        shapes: List[Tuple[int, ...]] = []

        def collect(item: Any):
            if torch.is_tensor(item):
                shapes.append(tuple(item.shape))
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    collect(sub_item)
            elif isinstance(item, dict):
                for sub_item in item.values():
                    collect(sub_item)

        collect(value)
        return shapes

    def _collect_grad_fns(self, value: Any) -> List[Any]:
        """Collect grad_fn handles from nested tensor outputs."""
        grad_fns = []

        def collect(item: Any):
            if torch.is_tensor(item):
                if item.grad_fn is not None:
                    grad_fns.append(item.grad_fn)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    collect(sub_item)
            elif isinstance(item, dict):
                for sub_item in item.values():
                    collect(sub_item)

        collect(value)
        return grad_fns

    def _collect_named_tensors(self, value: Any, prefix: str = "output") -> List[Tuple[str, torch.Tensor]]:
        """Collect tensors from nested outputs while preserving readable names."""
        tensors: List[Tuple[str, torch.Tensor]] = []

        def collect(item: Any, path: str):
            if torch.is_tensor(item):
                tensors.append((path, item))
            elif isinstance(item, (list, tuple)):
                for index, sub_item in enumerate(item):
                    collect(sub_item, f"{path}[{index}]")
            elif isinstance(item, dict):
                for key, sub_item in item.items():
                    collect(sub_item, f"{path}.{key}")

        collect(value, prefix)
        return tensors

    def _clone_input_structure(self, value: Any) -> Any:
        """Clone tensors inside nested model inputs and enable gradients when possible."""
        if torch.is_tensor(value):
            cloned = value.detach().clone()
            if cloned.is_floating_point() or cloned.is_complex():
                cloned.requires_grad_(True)
            return cloned
        if isinstance(value, list):
            return [self._clone_input_structure(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_input_structure(item) for item in value)
        if isinstance(value, dict):
            return {key: self._clone_input_structure(item) for key, item in value.items()}
        return value

    def _forward_model(self, model_inputs: Any) -> Any:
        """Execute the tracked model with tensor, tuple/list, or dict inputs."""
        if isinstance(model_inputs, dict):
            return self.model(**model_inputs)
        if isinstance(model_inputs, (tuple, list)):
            return self.model(*model_inputs)
        return self.model(model_inputs)

    def _normalize_operation_name(self, operation: Any) -> str:
        """Normalize autograd node names for graph labels."""
        name = str(operation).split("(")[0]
        if name.startswith("<") and name.endswith(">"):
            name = name[1:-1]
        if " object at " in name:
            name = name.split(" object at ")[0]
        name = name.replace("Backward0", "Backward")
        name = name.replace("Backward1", "Backward")
        name = name.replace("Function", "")
        return name or "Unknown"

    def _classify_operation_family(self, name: str) -> str:
        """Group autograd operations into a smaller set of visual families."""
        normalized = name.lower()

        if "accumulategrad" in normalized or "grad" in normalized:
            return "gradient"
        if "backward" in normalized:
            return "backward"
        if any(token in normalized for token in ["conv", "convolution"]):
            return "convolution"
        if any(token in normalized for token in ["addmm", "mm", "matmul", "linear"]):
            return "linear"
        if any(token in normalized for token in ["relu", "gelu", "sigmoid", "tanh", "silu", "softmax"]):
            return "activation"
        if "norm" in normalized:
            return "normalization"
        if "pool" in normalized:
            return "pooling"
        if any(token in normalized for token in ["sum", "mean", "mul", "add", "sub", "div"]):
            return "reduction"
        if any(token in normalized for token in ["view", "reshape", "flatten", "transpose", "permute", "cat", "slice"]):
            return "tensor"
        if any(token in normalized for token in ["input", "output"]):
            return "io"
        return "other"

    def _map_family_to_operation_type(self, family: str) -> OperationType:
        """Map rendering families back to the public operation type enum."""
        if family == "gradient":
            return OperationType.GRADIENT_OP
        if family == "backward":
            return OperationType.BACKWARD
        if family in {"linear", "convolution", "activation", "normalization", "pooling"}:
            return OperationType.LAYER_OP
        if family in {"reduction", "tensor", "io"}:
            return OperationType.TENSOR_OP
        return OperationType.CUSTOM

    def _reduce_output_to_scalar(self, output: Any) -> torch.Tensor:
        """Reduce model outputs to a scalar tensor for autograd traversal."""
        if torch.is_tensor(output):
            return output.sum()
        if isinstance(output, (list, tuple)):
            tensors = [item for item in output if torch.is_tensor(item)]
            if tensors:
                total = tensors[0].sum()
                for item in tensors[1:]:
                    total = total + item.sum()
                return total
        if isinstance(output, dict):
            tensors = [item for item in output.values() if torch.is_tensor(item)]
            if tensors:
                total = tensors[0].sum()
                for item in tensors[1:]:
                    total = total + item.sum()
                return total
        raise TypeError("Unsupported output type for computational graph capture")

    def _capture_memory_snapshot(self) -> Optional[Dict[str, int]]:
        """Capture a memory snapshot when CUDA is available."""
        if not (self.track_memory and torch.cuda.is_available()):
            return None

        memory_stats = torch.cuda.memory_stats()
        return {
            'peak_allocated': memory_stats.get('allocated_bytes.all.peak', 0),
            'current_allocated': memory_stats.get('allocated_bytes.all.current', 0),
            'peak_reserved': memory_stats.get('reserved_bytes.all.peak', 0),
            'current_reserved': memory_stats.get('reserved_bytes.all.current', 0),
        }

    def _build_graph_from_autograd(
        self,
        root_functions: List[Any],
        module_metadata: Dict[int, Dict[str, Any]],
        parameter_names: Optional[Mapping[int, str]] = None,
        output_tensors: Optional[List[Tuple[str, torch.Tensor]]] = None,
        graph_source: str = "autograd",
    ):
        """Populate GraphNode and GraphEdge instances from one or more autograd roots."""
        operations: Dict[str, GraphNode] = {}
        edges: Dict[Tuple[str, str], GraphEdge] = {}
        node_ids: Dict[int, int] = {}
        active: Set[str] = set()
        next_id = 0

        def get_node_id(grad_fn: Any, prefix: str = "autograd") -> str:
            nonlocal next_id
            key = id(grad_fn)
            if key not in node_ids:
                node_ids[key] = next_id
                next_id += 1
            return f"{prefix}_{node_ids[key]}"

        def ensure_parameter_node(variable: torch.Tensor, depth: int) -> str:
            node_id = get_node_id(variable, prefix="parameter")
            parameter_name = parameter_names.get(id(variable)) if parameter_names else None

            if node_id not in operations:
                operations[node_id] = GraphNode(
                    id=node_id,
                    name=parameter_name or "Parameter",
                    operation_type=OperationType.GRADIENT_OP,
                    module_name=parameter_name.rsplit(".", 1)[0] if parameter_name and "." in parameter_name else None,
                    input_shapes=[tuple(variable.shape)],
                    output_shapes=[tuple(variable.shape)],
                    parameters={
                        "count": int(variable.numel()),
                    },
                    execution_time=None,
                    memory_usage=None,
                    metadata={
                        "family": "parameter",
                        "depth": depth,
                        "dtype": str(variable.dtype),
                        "requires_grad": bool(variable.requires_grad),
                        "is_parameter": True,
                    },
                    parent_ids=[],
                    child_ids=[],
                    timestamp=time.time() - self.start_time if self.start_time else None,
                )
            else:
                operations[node_id].metadata["depth"] = min(
                    operations[node_id].metadata.get("depth", depth),
                    depth,
                )

            return node_id

        def ensure_node(grad_fn: Any, depth: int) -> str:
            node_id = get_node_id(grad_fn)
            module_meta = module_metadata.get(id(grad_fn), {})
            operation_name = self._normalize_operation_name(grad_fn)
            family = self._classify_operation_family(operation_name)
            tensor_variable = getattr(grad_fn, "variable", None)
            parameter_name = None

            if torch.is_tensor(tensor_variable):
                parameter_name = parameter_names.get(id(tensor_variable)) if parameter_names else None

            if node_id not in operations:
                metadata = {
                    "family": family,
                    "depth": depth,
                    "module_type": module_meta.get("module_type"),
                }
                input_shapes = module_meta.get("input_shapes")
                output_shapes = module_meta.get("output_shapes")
                parameters = module_meta.get("parameters")

                if torch.is_tensor(tensor_variable):
                    tensor_shape = tuple(tensor_variable.shape)
                    metadata.update({
                        "parameter_name": parameter_name,
                        "dtype": str(tensor_variable.dtype),
                        "requires_grad": bool(tensor_variable.requires_grad),
                    })
                    if parameter_name:
                        metadata["module_type"] = metadata.get("module_type") or "Parameter"
                    if input_shapes is None:
                        input_shapes = [tensor_shape]
                    if output_shapes is None:
                        output_shapes = [tensor_shape]
                    if parameters is None:
                        parameters = {"count": int(tensor_variable.numel())}

                operations[node_id] = GraphNode(
                    id=node_id,
                    name=operation_name if not parameter_name else f"{operation_name}: {parameter_name}",
                    operation_type=self._map_family_to_operation_type(family),
                    module_name=module_meta.get("module_name"),
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    parameters=parameters,
                    execution_time=None,
                    memory_usage=None,
                    metadata=metadata,
                    parent_ids=[],
                    child_ids=[],
                    timestamp=time.time() - self.start_time if self.start_time else None,
                )
            else:
                operations[node_id].metadata["depth"] = min(
                    operations[node_id].metadata.get("depth", depth),
                    depth,
                )

            return node_id

        def add_edge(source_id: str, target_id: str, edge_type: str = "autograd_dependency",
                     tensor_shape: Optional[Tuple[int, ...]] = None):
            if (source_id, target_id) in edges:
                return

            target_node = operations[target_id]
            edge = GraphEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                tensor_shape=tensor_shape or (target_node.input_shapes[0] if target_node.input_shapes else None),
            )
            edges[(source_id, target_id)] = edge

            if target_id not in operations[source_id].child_ids:
                operations[source_id].child_ids.append(target_id)
            if source_id not in operations[target_id].parent_ids:
                operations[target_id].parent_ids.append(source_id)

        def traverse(grad_fn: Any, depth: int = 0):
            if grad_fn is None:
                return

            node_id = ensure_node(grad_fn, depth)
            if node_id in active:
                return

            active.add(node_id)
            tensor_variable = getattr(grad_fn, "variable", None)
            if torch.is_tensor(tensor_variable):
                parameter_node_id = ensure_parameter_node(tensor_variable, depth + 1)
                add_edge(
                    parameter_node_id,
                    node_id,
                    edge_type="parameter_dependency",
                    tensor_shape=tuple(tensor_variable.shape),
                )

            for next_fn, _ in getattr(grad_fn, "next_functions", []):
                if next_fn is None:
                    continue

                parent_id = ensure_node(next_fn, depth + 1)
                add_edge(parent_id, node_id)
                traverse(next_fn, depth + 1)
            active.remove(node_id)

        for grad_fn in root_functions:
            traverse(grad_fn)

        if output_tensors:
            for output_index, (output_name, output_tensor) in enumerate(output_tensors):
                output_node_id = f"output_{output_index}"
                output_shape = tuple(output_tensor.shape)
                operations[output_node_id] = GraphNode(
                    id=output_node_id,
                    name=output_name,
                    operation_type=OperationType.TENSOR_OP,
                    module_name=None,
                    input_shapes=[output_shape],
                    output_shapes=[output_shape],
                    parameters=None,
                    execution_time=None,
                    memory_usage=None,
                    metadata={
                        "family": "io",
                        "depth": -1,
                        "dtype": str(output_tensor.dtype),
                        "requires_grad": bool(output_tensor.requires_grad),
                        "is_output": True,
                    },
                    parent_ids=[],
                    child_ids=[],
                    timestamp=time.time() - self.start_time if self.start_time else None,
                )
                if output_tensor.grad_fn is not None:
                    source_id = ensure_node(output_tensor.grad_fn, 0)
                    add_edge(
                        source_id,
                        output_node_id,
                        edge_type="model_output",
                        tensor_shape=output_shape,
                    )

        ordered_nodes = sorted(
            operations.values(),
            key=lambda node: (
                node.metadata.get("depth", 0) if node.metadata else 0,
                node.id,
            ),
        )

        self.nodes = {node.id: node for node in ordered_nodes}
        self.edges = sorted(edges.values(), key=lambda edge: (edge.source_id, edge.target_id))
        self.graph_source = graph_source

    def capture_execution(
        self,
        input_tensor: Any,
        parameter_names: Optional[Mapping[int, str]] = None,
    ) -> "ComputationalGraphTracker":
        """Capture a real autograd graph for a model execution."""
        self._reset_graph_data()
        self.input_tensor = input_tensor
        self.start_time = time.time()

        if self.track_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tracked_input = self._clone_input_structure(input_tensor)

        module_metadata: Dict[int, Dict[str, Any]] = {}
        hooks = []

        def create_hook(module_name: str):
            def hook(module, inputs, output):
                metadata = {
                    "module_name": module_name,
                    "module_type": type(module).__name__,
                    "input_shapes": self._extract_tensor_shapes(inputs),
                    "output_shapes": self._extract_tensor_shapes(output),
                    "parameters": {
                        "count": sum(parameter.numel() for parameter in module.parameters()),
                    },
                }
                for grad_fn in self._collect_grad_fns(output):
                    module_metadata[id(grad_fn)] = metadata
            return hook

        for module_name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(create_hook(module_name)))

        try:
            output = self._forward_model(tracked_input)
            self.execution_time = time.time() - self.start_time if self.track_timing else None

            memory_snapshot = self._capture_memory_snapshot()
            if memory_snapshot is not None:
                self.memory_snapshots.append(memory_snapshot)

            output_tensors = self._collect_named_tensors(output)
            root_functions = [tensor.grad_fn for _, tensor in output_tensors if tensor.grad_fn is not None]
            self._build_graph_from_autograd(
                root_functions,
                module_metadata,
                parameter_names=parameter_names,
                output_tensors=output_tensors,
                graph_source="autograd",
            )
        finally:
            for hook in hooks:
                hook.remove()

        return self

    def capture_output(
        self,
        output: Any,
        params: Optional[Mapping[str, torch.Tensor]] = None,
        output_names: Optional[List[str]] = None,
    ) -> "ComputationalGraphTracker":
        """Build a graph directly from model outputs without rerunning the model."""
        self._reset_graph_data()
        self.start_time = time.time()

        output_tensors = self._collect_named_tensors(output)
        if output_names is not None:
            if len(output_names) != len(output_tensors):
                raise ValueError(
                    f"Expected {len(output_tensors)} output names, received {len(output_names)}."
                )
            output_tensors = [
                (output_name, tensor)
                for output_name, (_, tensor) in zip(output_names, output_tensors)
            ]

        parameter_names = {
            id(tensor): name
            for name, tensor in (params or {}).items()
            if torch.is_tensor(tensor)
        }
        root_functions = [tensor.grad_fn for _, tensor in output_tensors if tensor.grad_fn is not None]

        self._build_graph_from_autograd(
            root_functions,
            {},
            parameter_names=parameter_names,
            output_tensors=output_tensors,
            graph_source="autograd_output",
        )
        return self
    
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
                if hasattr(module, "register_full_backward_hook"):
                    backward_hook = module.register_full_backward_hook(
                        self._create_backward_hook(name)
                    )
                else:
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
        tracker = self
        
        # Override tensor operations
        def tracked_add(tensor, other):
            if tracker.is_tracking:
                tracker._track_tensor_operation('add', tensor, other)
            return tracker.original_methods['tensor_add'](tensor, other)
        
        def tracked_mul(tensor, other):
            if tracker.is_tracking:
                tracker._track_tensor_operation('mul', tensor, other)
            return tracker.original_methods['tensor_mul'](tensor, other)
        
        def tracked_matmul(tensor, other):
            if tracker.is_tracking:
                tracker._track_tensor_operation('matmul', tensor, other)
            return tracker.original_methods['tensor_matmul'](tensor, other)
        
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
                'execution_time': self.execution_time if self.execution_time is not None else (
                    time.time() - self.start_time if self.start_time else None
                ),
                'memory_usage': self.memory_snapshots[-1] if self.memory_snapshots else None,
                'graph_source': self.graph_source,
            }
            
            # Count operation types
            for node in self.nodes.values():
                # Handle both GraphNode objects and dictionary objects
                if hasattr(node, 'operation_type'):
                    # GraphNode object
                    op_type = node.operation_type.value if hasattr(node.operation_type, 'value') else str(node.operation_type)
                    summary['operation_types'][op_type] += 1
                    if hasattr(node, 'module_name') and node.module_name:
                        module_type = node.metadata.get('module_type', 'Unknown') if hasattr(node, 'metadata') and node.metadata else 'Unknown'
                        summary['module_types'][module_type] += 1
                elif isinstance(node, dict):
                    # Dictionary object
                    op_type = node.get('operation_type', 'unknown')
                    summary['operation_types'][op_type] += 1
                    if node.get('module_name'):
                        module_type = node.get('metadata', {}).get('module_type', 'Unknown') if isinstance(node.get('metadata'), dict) else 'Unknown'
                        summary['module_types'][module_type] += 1
            
            return summary

    def _serialize_node(self, node: Any) -> Dict[str, Any]:
        """Serialize GraphNode or dictionary nodes into export-friendly dictionaries."""
        if isinstance(node, GraphNode):
            data = asdict(node)
            if isinstance(node.operation_type, Enum):
                data['operation_type'] = node.operation_type.value
            return data

        data = dict(node)
        op_type = data.get('operation_type')
        if isinstance(op_type, Enum):
            data['operation_type'] = op_type.value
        return data

    def _serialize_edge(self, edge: Any) -> Dict[str, Any]:
        """Serialize GraphEdge or dictionary edges into export-friendly dictionaries."""
        if isinstance(edge, GraphEdge):
            return asdict(edge)
        return dict(edge)
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get the complete graph data for visualization."""
        with self.lock:
            return {
                'nodes': [self._serialize_node(node) for node in self.nodes.values()],
                'edges': [self._serialize_edge(edge) for edge in self.edges],
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
                       node_size: int = 20, font_size: int = 10,
                       submission_type: Optional[str] = None,
                       title: Optional[str] = None) -> str:
        """
        Save the computational graph as a PNG image with a publication-oriented layout.
        
        Args:
            filepath: Output file path
            width: Image width in pixels
            height: Image height in pixels
            dpi: Dots per inch for high resolution
            show_legend: Whether to show legend (positioned outside plot area)
            node_size: Size of nodes in the graph
            font_size: Font size for labels
            submission_type: Publication target profile
            title: Optional chart title override
            
        Returns:
            Path to the saved PNG file
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
            import warnings
            import textwrap
            
            # Suppress PyTorch warnings
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
            
        except ImportError:
            raise ImportError("Matplotlib is required for PNG generation. Install with: pip install matplotlib")
        if (not self.nodes or self.graph_source not in {"autograd", "autograd_output"}) and getattr(self, "input_tensor", None) is not None:
            self.capture_execution(self.input_tensor)

        graph_data = self.get_graph_data()
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        operations = []
        for node in nodes:
            metadata = node.get("metadata") or {}
            name = node.get("name", "Unknown")
            operations.append({
                "node_id": node.get("id"),
                "name": name,
                "depth": metadata.get("depth", node.get("depth", 0)),
                "family": metadata.get("family", self._classify_operation_family(name)),
            })

        graph_edges = [
            (edge.get("source_id"), edge.get("target_id"))
            for edge in edges
            if edge.get("source_id") and edge.get("target_id")
        ]

        if not operations:
            return filepath

        submission_profile = get_submission_style(submission_type)
        profile_label = submission_profile["label"]
        title_color = submission_profile["title_color"]
        subtitle_color = submission_profile["subtitle_color"]
        box_fill = submission_profile["box_fill"]
        box_border = submission_profile["box_border"]
        shadow_color = submission_profile["shadow_color"]
        arrow_color = submission_profile["arrow_color"]
        legend_frame = submission_profile["legend_frame"]
        note_color = submission_profile["note_color"]

        family_palette = {
            "linear": {"accent": submission_profile["family_palette"]["Dense"], "label": "Linear"},
            "convolution": {"accent": submission_profile["family_palette"]["Convolution"], "label": "Convolution"},
            "activation": {"accent": submission_profile["family_palette"]["Activation"], "label": "Activation"},
            "normalization": {"accent": submission_profile["family_palette"]["Normalization"], "label": "Normalization"},
            "pooling": {"accent": submission_profile["family_palette"]["Pooling"], "label": "Pooling"},
            "reduction": {"accent": submission_profile["family_palette"]["Other"], "label": "Tensor Op"},
            "tensor": {"accent": submission_profile["family_palette"]["Tensor Shape"], "label": "Tensor Shape"},
            "backward": {"accent": submission_profile["family_palette"]["Dense"], "label": "Backward"},
            "gradient": {"accent": submission_profile["family_palette"]["Regularization"], "label": "Gradient"},
            "parameter": {"accent": submission_profile["family_palette"]["Dense"], "label": "Parameter"},
            "io": {"accent": submission_profile["family_palette"]["Recurrent"], "label": "Input / Output"},
            "other": {"accent": submission_profile["family_palette"]["Other"], "label": "Other"},
        }

        depth_groups = defaultdict(list)
        for operation in operations:
            depth_groups[operation["depth"]].append(operation)

        compact_depths = sorted(depth_groups.keys())
        depth_mapping = {depth: index for index, depth in enumerate(compact_depths)}
        max_compact_depth = max(depth_mapping.values()) if depth_mapping else 0

        node_width = 2.75
        node_height = 1.0
        y_spacing = 1.55
        positions = {}

        operation_lookup = {
            operation["node_id"]: operation
            for operation in operations
        }
        depth_transition_counts: Dict[int, int] = defaultdict(int)
        for source_id, target_id in graph_edges:
            source_operation = operation_lookup.get(source_id)
            target_operation = operation_lookup.get(target_id)
            if source_operation is None or target_operation is None:
                continue

            source_compact_depth = depth_mapping.get(source_operation["depth"])
            target_compact_depth = depth_mapping.get(target_operation["depth"])
            if source_compact_depth is None or target_compact_depth is None:
                continue

            lower_depth = min(source_compact_depth, target_compact_depth)
            upper_depth = max(source_compact_depth, target_compact_depth)
            for gap_index in range(lower_depth, upper_depth):
                depth_transition_counts[gap_index] += 1

        base_column_gap = 0.95
        extra_gap_per_edge = 0.18
        column_positions: Dict[int, float] = {max_compact_depth: 1.5}
        for compact_depth in range(max_compact_depth - 1, -1, -1):
            edge_count = depth_transition_counts.get(compact_depth, 1)
            column_gap = base_column_gap + max(0, edge_count - 1) * extra_gap_per_edge
            column_positions[compact_depth] = (
                column_positions[compact_depth + 1] + node_width + column_gap
            )

        for original_depth in compact_depths:
            compact_depth = depth_mapping[original_depth]
            x_pos = column_positions[compact_depth]
            group = depth_groups[original_depth]
            total_height = (len(group) - 1) * y_spacing
            start_y = total_height / 2

            for row_index, operation in enumerate(group):
                y_pos = start_y - row_index * y_spacing
                positions[operation["node_id"]] = (x_pos, y_pos)

        present_families = []
        for operation in operations:
            family = operation["family"]
            if family not in present_families:
                present_families.append(family)

        fig_width = max(width / 100, 8.5 + len(compact_depths) * 1.25 + (2.2 if show_legend else 0))
        fig_height = max(height / 100, 4.8 + max(len(group) for group in depth_groups.values()) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        grouped_edges: Dict[Tuple[float, float], List[Dict[str, float]]] = defaultdict(list)
        edge_specs: List[Dict[str, float]] = []

        for source_id, target_id in graph_edges:
            if source_id not in positions or target_id not in positions:
                continue

            start_x, start_y = positions[source_id]
            end_x, end_y = positions[target_id]
            source_edge_x = start_x + node_width / 2
            target_edge_x = end_x - node_width / 2
            edge_spec = {
                "source_id": source_id,
                "target_id": target_id,
                "start_y": start_y,
                "end_y": end_y,
                "source_edge_x": source_edge_x,
                "target_edge_x": target_edge_x,
            }
            edge_specs.append(edge_spec)
            grouped_edges[(source_edge_x, target_edge_x)].append(edge_spec)

        lane_centers: Dict[Tuple[str, str], float] = {}
        for lane_key, lane_edges in grouped_edges.items():
            source_edge_x, target_edge_x = lane_key
            available_width = max(0.0, target_edge_x - source_edge_x)
            if available_width <= 0:
                for edge_spec in lane_edges:
                    lane_centers[(edge_spec["source_id"], edge_spec["target_id"])] = (
                        source_edge_x + target_edge_x
                    ) / 2
                continue

            horizontal_margin = min(0.18, available_width * 0.22)
            left_bound = source_edge_x + horizontal_margin
            right_bound = target_edge_x - horizontal_margin
            if right_bound <= left_bound:
                left_bound = source_edge_x + available_width * 0.35
                right_bound = target_edge_x - available_width * 0.35

            ordered_edges = sorted(
                lane_edges,
                key=lambda edge: ((edge["start_y"] + edge["end_y"]) / 2, edge["start_y"], edge["end_y"]),
                reverse=True,
            )
            if len(ordered_edges) == 1:
                lane_positions = [(left_bound + right_bound) / 2]
            else:
                lane_positions = [
                    left_bound + (right_bound - left_bound) * ((index + 1) / (len(ordered_edges) + 1))
                    for index in range(len(ordered_edges))
                ]

            for edge_spec, lane_center_x in zip(ordered_edges, lane_positions):
                lane_centers[(edge_spec["source_id"], edge_spec["target_id"])] = lane_center_x

        for edge_spec in edge_specs:
            source_id = edge_spec["source_id"]
            target_id = edge_spec["target_id"]
            start_y = edge_spec["start_y"]
            end_y = edge_spec["end_y"]
            source_edge_x = edge_spec["source_edge_x"]
            target_edge_x = edge_spec["target_edge_x"]

            if abs(start_y - end_y) <= 0.08:
                arrow = FancyArrowPatch(
                    (source_edge_x, start_y),
                    (target_edge_x, end_y),
                    arrowstyle="-|>",
                    mutation_scale=max(10, node_size // 2),
                    linewidth=1.1,
                    color=arrow_color,
                    alpha=0.9,
                    shrinkA=4,
                    shrinkB=4,
                    zorder=1,
                )
                ax.add_patch(arrow)
                continue

            lane_center_x = lane_centers[(source_id, target_id)]
            arrow_head_length = min(0.2, max(0.1, (target_edge_x - lane_center_x) * 0.5))
            arrow_start_x = target_edge_x - arrow_head_length

            ax.plot(
                [source_edge_x, lane_center_x],
                [start_y, start_y],
                color=arrow_color,
                linewidth=1.1,
                alpha=0.9,
                zorder=1,
            )
            ax.plot(
                [lane_center_x, lane_center_x],
                [start_y, end_y],
                color=arrow_color,
                linewidth=1.1,
                alpha=0.9,
                zorder=1,
            )
            ax.plot(
                [lane_center_x, arrow_start_x],
                [end_y, end_y],
                color=arrow_color,
                linewidth=1.1,
                alpha=0.9,
                zorder=1,
            )

            arrow = FancyArrowPatch(
                (arrow_start_x, end_y),
                (target_edge_x, end_y),
                arrowstyle="-|>",
                mutation_scale=max(10, node_size // 2),
                linewidth=1.1,
                color=arrow_color,
                alpha=0.9,
                shrinkA=0,
                shrinkB=4,
                zorder=1,
            )
            ax.add_patch(arrow)

        for operation in operations:
            node_id = operation["node_id"]
            if node_id not in positions:
                continue

            x_pos, y_pos = positions[node_id]
            family_info = family_palette.get(operation["family"], family_palette["other"])
            accent = family_info["accent"]

            shadow = FancyBboxPatch(
                (x_pos - node_width / 2 + 0.04, y_pos - node_height / 2 - 0.05),
                node_width,
                node_height,
                boxstyle="round,pad=0.06,rounding_size=0.08",
                facecolor=shadow_color,
                edgecolor="none",
                alpha=0.35,
                zorder=2,
            )
            ax.add_patch(shadow)

            node = FancyBboxPatch(
                (x_pos - node_width / 2, y_pos - node_height / 2),
                node_width,
                node_height,
                boxstyle="round,pad=0.06,rounding_size=0.08",
                facecolor=box_fill,
                edgecolor=box_border,
                linewidth=0.9,
                zorder=3,
            )
            ax.add_patch(node)

            accent_bar = patches.Rectangle(
                (x_pos - node_width / 2, y_pos + node_height / 2 - 0.14),
                node_width,
                0.14,
                facecolor=accent,
                edgecolor="none",
                zorder=4,
            )
            ax.add_patch(accent_bar)

            label = textwrap.fill(operation["name"], width=18)
            ax.text(
                x_pos,
                y_pos + 0.06,
                label,
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color=title_color,
                zorder=5,
            )
            ax.text(
                x_pos,
                y_pos - node_height / 2 + 0.18,
                family_info["label"].upper(),
                ha="center",
                va="center",
                fontsize=max(7, font_size - 2),
                fontweight="bold",
                color=accent,
                zorder=5,
            )

        all_x = [position[0] for position in positions.values()]
        all_y = [position[1] for position in positions.values()]
        legend_padding = 2.8 if show_legend else 1.0
        ax.set_xlim(min(all_x) - node_width, max(all_x) + node_width + legend_padding)
        ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)
        ax.axis("off")

        model_name = title or f"{type(self.model).__name__} Autograd Graph"
        ax.text(
            0.0,
            1.05,
            model_name,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=title_color,
        )
        ax.text(
            0.0,
            1.01,
            f"{len(operations)} nodes | {len(graph_edges)} dependencies | {profile_label} profile",
            transform=ax.transAxes,
            fontsize=9.5,
            ha="left",
            va="bottom",
            color=subtitle_color,
        )

        if show_legend:
            legend_elements = [
                patches.Patch(
                    facecolor="#FCFCFD",
                    edgecolor=family_palette[family]["accent"],
                    linewidth=1.2,
                    label=family_palette[family]["label"],
                )
                for family in present_families
            ]

            if legend_elements:
                legend = ax.legend(
                    handles=legend_elements,
                    title="Operation Family",
                    loc="upper left",
                    bbox_to_anchor=(1.01, 0.98),
                    frameon=True,
                    fancybox=False,
                    fontsize=9,
                    title_fontsize=10,
                    borderpad=0.6,
                    labelspacing=0.5,
                )
                legend.get_frame().set_facecolor("white")
                legend.get_frame().set_edgecolor(legend_frame)
                legend.get_frame().set_alpha(0.98)

        trace_note = "autograd dependencies traced from model outputs."
        if self.graph_source == "autograd_output":
            trace_note = "autograd dependencies traced directly from output tensors."

        fig.text(
            0.01,
            0.01,
            f"{profile_label} styling | {trace_note}",
            fontsize=8,
            color=note_color,
        )

        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
        plt.close()

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
        """Create an enhanced Matplotlib visualization of the computational graph."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import FancyBboxPatch
            import networkx as nx
        except ImportError:
            raise ImportError("Matplotlib and NetworkX are required for visualization. Install with: pip install matplotlib networkx")
        
        # Get graph data
        graph_data = self.get_graph_data()
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # Enhanced color scheme
        colors = {
            'forward': '#2E7D32',      # Dark green
            'backward': '#C62828',     # Dark red
            'tensor_op': '#1565C0',    # Dark blue
            'layer_op': '#AD1457',     # Dark pink
            'gradient_op': '#37474F',  # Dark gray
            'memory_op': '#5D4037',    # Dark brown
            'custom': '#E65100'        # Dark orange
        }
        
        # Create figure with space for legend
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with enhanced attributes
        for node in nodes:
            G.add_node(node['id'], 
                      name=node['name'],
                      operation_type=node['operation_type'],
                      color=colors.get(node['operation_type'], '#424242'))
        
        # Add edges
        for edge in edges:
            G.add_edge(edge['source_id'], edge['target_id'])
        
        # Enhanced layout
        pos = nx.spring_layout(G, k=2, iterations=100)
        
        # Draw nodes with enhanced styling
        for node_id, data in G.nodes(data=True):
            x, y = pos[node_id]
            color = data['color']
            
            # Create enhanced node
            rect = FancyBboxPatch((x-0.1, y-0.05), 0.2, 0.1,
                                boxstyle='round,pad=0.02', 
                                facecolor=color, alpha=0.95, 
                                edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            
            # Add label with full method/object names
            full_name = data['name']
            # Clean up the name for better readability
            if full_name.startswith('<') and full_name.endswith('>'):
                # Remove angle brackets and clean up
                clean_name = full_name[1:-1]
                if 'object at 0x' in clean_name:
                    # Extract just the class name for T0 objects
                    parts = clean_name.split(' ')
                    if len(parts) > 0:
                        clean_name = parts[0]
                full_name = clean_name
            
            # Use full name without truncation
            ax.text(x, y, full_name, ha='center', va='center', 
                   fontsize=8, weight='bold', color='white')
        
        # Draw edges with enhanced styling
        for edge in G.edges():
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', color='#333333', 
                                     alpha=0.7, lw=1.5, shrinkA=3, shrinkB=3))
        
        # Set up the plot
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        plt.title('PyTorch Computational Graph\n(Enhanced Matplotlib Visualization)', 
                 fontsize=16, weight='bold', pad=20)
        
        # Add legend with proper positioning
        legend_elements = []
        for op_type, color in colors.items():
            if any(node['operation_type'] == op_type for node in nodes):
                label = op_type.replace('_', ' ').title()
                legend_elements.append(patches.Patch(color=color, label=label))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='center left', 
                     bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.95)
        
        # Add summary
        summary = graph_data['summary']
        summary_text = f'Operations: {summary["total_nodes"]} | Edges: {summary["total_edges"]}'
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, 
               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return plt.gcf()


def track_computational_graph(model: nn.Module, input_tensor: Any,
                            track_memory: bool = True, track_timing: bool = True,
                            track_tensor_ops: bool = True) -> ComputationalGraphTracker:
    """
    Track the computational graph of a PyTorch model execution.
    Uses a simplified approach to avoid PyTorch hook warnings.
    
    Args:
        model: PyTorch model to track
        input_tensor: Input tensor for the forward pass
        track_memory: Whether to track memory usage
        track_timing: Whether to track execution timing
        track_tensor_ops: Whether to track tensor operations
        
    Returns:
        ComputationalGraphTracker with the execution data
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    
    tracker = ComputationalGraphTracker(
        model, track_memory, track_timing, track_tensor_ops
    )
    
    tracker.input_tensor = input_tensor
    parameter_names = {
        id(parameter): name
        for name, parameter in model.named_parameters()
    }
    
    try:
        tracker.capture_execution(input_tensor, parameter_names=parameter_names)
    except Exception as e:
        print(f"Warning: Could not track computational graph: {e}")
        # Create minimal fallback data
        tracker._reset_graph_data()
        tracker.graph_source = "fallback"
        tracker.nodes = {
            'input': GraphNode(
                id='input',
                name='Input Tensor',
                operation_type=OperationType.TENSOR_OP,
                metadata={'family': 'io', 'depth': 0},
                parent_ids=[],
                child_ids=['output'],
            ),
            'output': GraphNode(
                id='output',
                name='Output Tensor',
                operation_type=OperationType.TENSOR_OP,
                metadata={'family': 'io', 'depth': 1, 'error': str(e)},
                parent_ids=['input'],
                child_ids=[],
            ),
        }
        tracker.edges = [
            GraphEdge(source_id='input', target_id='output', edge_type='fallback_dependency')
        ]
    
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
            # Handle both GraphNode objects and dictionary objects
            module_name = None
            op_type = None
            execution_time = None
            input_shapes = None
            output_shapes = None
            
            if hasattr(node, 'module_name'):
                # GraphNode object
                module_name = node.module_name
                op_type = node.operation_type.value if hasattr(node.operation_type, 'value') else str(node.operation_type)
                execution_time = node.execution_time
                input_shapes = node.input_shapes
                output_shapes = node.output_shapes
            elif isinstance(node, dict):
                # Dictionary object
                module_name = node.get('module_name')
                op_type = node.get('operation_type', 'unknown')
                execution_time = node.get('execution_time')
                input_shapes = node.get('input_shapes')
                output_shapes = node.get('output_shapes')
            
            if module_name:
                layer_analysis[module_name].append({
                    'operation_type': op_type,
                    'execution_time': execution_time,
                    'input_shapes': input_shapes,
                    'output_shapes': output_shapes,
                })
        
        analysis['layer_analysis'] = dict(layer_analysis)
    
    return analysis 

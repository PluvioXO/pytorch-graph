"""
Layer information and metadata handling.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class LayerInfo:
    """
    Represents information about a neural network layer for visualization.
    """
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: int
    trainable_params: int
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: str = '#3498db'
    metadata: Dict[str, Any] = None
    connections: List[str] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.metadata is None:
            self.metadata = {}
        if self.connections is None:
            self.connections = []
    
    @property
    def volume(self) -> float:
        """Calculate the volume of the layer for sizing."""
        return self.size[0] * self.size[1] * self.size[2]
    
    @property
    def param_density(self) -> float:
        """Calculate parameter density (params per output unit)."""
        output_units = np.prod(self.output_shape)
        return self.parameters / max(output_units, 1)
    
    def get_display_name(self) -> str:
        """Get a formatted display name for the layer."""
        if self.layer_type in ['Linear', 'Dense', 'Fully Connected']:
            return f"{self.name}\n{self.layer_type}\n{self.output_shape[-1]} units"
        elif self.layer_type in ['Conv1d', 'Conv2d', 'Conv3d']:
            return f"{self.name}\n{self.layer_type}\n{self.output_shape}"
        elif self.layer_type in ['LSTM', 'GRU', 'RNN']:
            return f"{self.name}\n{self.layer_type}\n{self.output_shape[-1]} units"
        else:
            return f"{self.name}\n{self.layer_type}"
    
    def get_color_by_type(self) -> str:
        """Get color based on layer type."""
        color_map = {
            'Linear': '#e74c3c',      # Red
            'Dense': '#e74c3c',       # Red
            'Conv1d': '#3498db',      # Blue
            'Conv2d': '#3498db',      # Blue
            'Conv3d': '#3498db',      # Blue
            'BatchNorm': '#f39c12',   # Orange
            'Dropout': '#95a5a6',     # Gray
            'ReLU': '#2ecc71',        # Green
            'Sigmoid': '#9b59b6',     # Purple
            'Tanh': '#9b59b6',        # Purple
            'LSTM': '#e67e22',        # Dark Orange
            'GRU': '#e67e22',         # Dark Orange
            'Embedding': '#1abc9c',   # Turquoise
            'MaxPool': '#34495e',     # Dark Blue
            'AvgPool': '#34495e',     # Dark Blue
            'AdaptivePool': '#34495e', # Dark Blue
            'Flatten': '#7f8c8d',     # Gray
            'Reshape': '#7f8c8d',     # Gray
        }
        return color_map.get(self.layer_type, self.color)
    
    def calculate_size(self, scale_factor: float = 1.0) -> Tuple[float, float, float]:
        """
        Calculate layer size based on output shape and parameters.
        
        Args:
            scale_factor: Scaling factor for visualization
            
        Returns:
            Tuple of (width, height, depth) for the layer
        """
        # Base size calculation
        output_volume = max(np.prod(self.output_shape), 1)
        param_factor = max(np.log10(max(self.parameters, 1)), 1)
        
        # Different sizing strategies based on layer type
        if self.layer_type in ['Linear', 'Dense']:
            # For dense layers, height represents number of units
            width = 1.0
            height = min(np.log10(self.output_shape[-1] + 1), 3.0)
            depth = min(np.log10(param_factor), 2.0)
        elif self.layer_type in ['Conv1d', 'Conv2d', 'Conv3d']:
            # For conv layers, size represents feature maps and spatial dimensions
            if len(self.output_shape) >= 3:  # Has channel dimension
                channels = self.output_shape[-3] if len(self.output_shape) >= 3 else 1
                width = min(np.log10(channels + 1), 2.0)
                height = min(np.log10(np.prod(self.output_shape[-2:]) + 1), 3.0)
                depth = 1.0
            else:
                width = height = depth = min(np.log10(output_volume), 2.0)
        elif self.layer_type in ['LSTM', 'GRU', 'RNN']:
            # For RNN layers, emphasize the hidden dimension
            width = 1.5
            height = min(np.log10(self.output_shape[-1] + 1), 3.0)
            depth = 1.0
        else:
            # Default sizing for other layers
            size = min(np.log10(output_volume), 2.0)
            width = height = depth = max(size, 0.5)
        
        return (
            width * scale_factor,
            height * scale_factor,
            depth * scale_factor
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert layer info to dictionary."""
        return {
            'name': self.name,
            'layer_type': self.layer_type,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'parameters': self.parameters,
            'trainable_params': self.trainable_params,
            'position': self.position,
            'size': self.size,
            'color': self.color,
            'metadata': self.metadata,
            'connections': self.connections,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerInfo':
        """Create LayerInfo from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return (f"LayerInfo(name='{self.name}', type='{self.layer_type}', "
                f"input_shape={self.input_shape}, output_shape={self.output_shape}, "
                f"params={self.parameters})")


class LayerInfoExtractor:
    """
    Utility class for extracting layer information from different frameworks.
    """
    
    @staticmethod
    def extract_pytorch_layer_info(layer, name: str, input_shape: Tuple[int, ...], 
                                 output_shape: Tuple[int, ...]) -> LayerInfo:
        """Extract layer information from PyTorch layer."""
        layer_type = layer.__class__.__name__
        
        # Count parameters
        total_params = sum(p.numel() for p in layer.parameters())
        trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        # Extract metadata
        metadata = {}
        if hasattr(layer, 'weight'):
            metadata['has_weight'] = True
            if hasattr(layer.weight, 'shape'):
                metadata['weight_shape'] = tuple(layer.weight.shape)
        if hasattr(layer, 'bias') and layer.bias is not None:
            metadata['has_bias'] = True
            metadata['bias_shape'] = tuple(layer.bias.shape)
        
        # Add layer-specific metadata
        if hasattr(layer, 'kernel_size'):
            metadata['kernel_size'] = layer.kernel_size
        if hasattr(layer, 'stride'):
            metadata['stride'] = layer.stride
        if hasattr(layer, 'padding'):
            metadata['padding'] = layer.padding
        if hasattr(layer, 'dilation'):
            metadata['dilation'] = layer.dilation
        if hasattr(layer, 'groups'):
            metadata['groups'] = layer.groups
        if hasattr(layer, 'dropout'):
            metadata['dropout'] = layer.dropout
        if hasattr(layer, 'hidden_size'):
            metadata['hidden_size'] = layer.hidden_size
        if hasattr(layer, 'num_layers'):
            metadata['num_layers'] = layer.num_layers
        
        return LayerInfo(
            name=name,
            layer_type=layer_type,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters=total_params,
            trainable_params=trainable_params,
            metadata=metadata
        )
    
    @staticmethod
    def extract_keras_layer_info(layer, name: str) -> LayerInfo:
        """Extract layer information from Keras layer."""
        layer_type = layer.__class__.__name__
        
        # Get shapes
        input_shape = tuple(layer.input_shape[1:]) if hasattr(layer, 'input_shape') else ()
        output_shape = tuple(layer.output_shape[1:]) if hasattr(layer, 'output_shape') else ()
        
        # Count parameters
        total_params = layer.count_params()
        trainable_params = sum([np.prod(w.shape) for w in layer.trainable_weights])
        
        # Extract metadata
        metadata = {}
        config = layer.get_config()
        
        # Add relevant config parameters
        relevant_keys = [
            'units', 'filters', 'kernel_size', 'strides', 'padding',
            'activation', 'use_bias', 'dropout', 'recurrent_dropout',
            'return_sequences', 'return_state', 'go_backwards', 'stateful'
        ]
        
        for key in relevant_keys:
            if key in config:
                metadata[key] = config[key]
        
        return LayerInfo(
            name=name,
            layer_type=layer_type,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters=total_params,
            trainable_params=trainable_params,
            metadata=metadata
        ) 
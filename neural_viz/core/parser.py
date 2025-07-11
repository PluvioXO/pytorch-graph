"""
Model parsing utilities for extracting layer information from neural networks.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
from ..utils.layer_info import LayerInfo, LayerInfoExtractor

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. PyTorch model parsing will be disabled.")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Keras model parsing will be disabled.")


class ModelParser:
    """
    Parses neural network models and extracts layer information for visualization.
    """
    
    def __init__(self):
        """Initialize the model parser."""
        self.torch_available = TORCH_AVAILABLE
        self.tf_available = TF_AVAILABLE
    
    def parse_model(self, model, framework: Optional[str] = None, 
                   input_shape: Optional[Tuple[int, ...]] = None) -> Tuple[List[LayerInfo], Dict[str, List[str]]]:
        """
        Parse a neural network model and extract layer information.
        
        Args:
            model: Neural network model (PyTorch or Keras)
            framework: Framework hint ('pytorch' or 'keras')
            input_shape: Input tensor shape (required for PyTorch)
            
        Returns:
            Tuple of (layers, connections) where:
            - layers: List of LayerInfo objects
            - connections: Dictionary mapping layer names to their output connections
        """
        # Auto-detect framework if not provided
        if framework is None:
            framework = self._detect_framework(model)
        
        if framework == 'pytorch':
            return self._parse_pytorch_model(model, input_shape)
        elif framework == 'keras':
            return self._parse_keras_model(model)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _detect_framework(self, model) -> str:
        """Auto-detect the framework of the model."""
        if self.torch_available and isinstance(model, torch.nn.Module):
            return 'pytorch'
        elif self.tf_available and isinstance(model, (tf.keras.Model, keras.Model)):
            return 'keras'
        else:
            # Try to detect by class name
            class_name = model.__class__.__module__
            if 'torch' in class_name:
                return 'pytorch'
            elif 'tensorflow' in class_name or 'keras' in class_name:
                return 'keras'
            else:
                raise ValueError("Cannot auto-detect framework. Please specify 'pytorch' or 'keras'.")
    
    def _parse_pytorch_model(self, model, input_shape: Optional[Tuple[int, ...]] = None) -> Tuple[List[LayerInfo], Dict[str, List[str]]]:
        """
        Parse a PyTorch model.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (batch_size, ...)
            
        Returns:
            Tuple of (layers, connections)
        """
        if not self.torch_available:
            raise RuntimeError("PyTorch is not available")
        
        if input_shape is None:
            raise ValueError("input_shape is required for PyTorch models")
        
        layers = []
        connections = {}
        
        # Create a dummy input to trace the model
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # Hook to capture layer information
        layer_outputs = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                layer_outputs[name] = {
                    'module': module,
                    'input': input,
                    'output': output
                }
            return hook
        
        # Register hooks for all named modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass to collect information
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process collected information
        prev_layer_name = None
        
        for name, info in layer_outputs.items():
            module = info['module']
            input_tensor = info['input'][0] if isinstance(info['input'], tuple) else info['input']
            output_tensor = info['output']
            
            # Handle different tensor shapes
            if hasattr(input_tensor, 'shape'):
                input_shape_layer = tuple(input_tensor.shape[1:])  # Remove batch dimension
            else:
                input_shape_layer = ()
            
            if hasattr(output_tensor, 'shape'):
                output_shape_layer = tuple(output_tensor.shape[1:])  # Remove batch dimension
            elif isinstance(output_tensor, (list, tuple)) and len(output_tensor) > 0:
                output_shape_layer = tuple(output_tensor[0].shape[1:])
            else:
                output_shape_layer = ()
            
            # Create layer info
            layer_info = LayerInfoExtractor.extract_pytorch_layer_info(
                module, name, input_shape_layer, output_shape_layer
            )
            
            # Set color based on type
            layer_info.color = layer_info.get_color_by_type()
            
            # Calculate size
            layer_info.size = layer_info.calculate_size()
            
            layers.append(layer_info)
            
            # Create connections (simple sequential for now)
            if prev_layer_name is not None:
                if prev_layer_name not in connections:
                    connections[prev_layer_name] = []
                connections[prev_layer_name].append(name)
            
            prev_layer_name = name
        
        return layers, connections
    
    def _parse_keras_model(self, model) -> Tuple[List[LayerInfo], Dict[str, List[str]]]:
        """
        Parse a Keras/TensorFlow model.
        
        Args:
            model: Keras model
            
        Returns:
            Tuple of (layers, connections)
        """
        if not self.tf_available:
            raise RuntimeError("TensorFlow is not available")
        
        layers = []
        connections = {}
        
        # Process each layer
        for i, layer in enumerate(model.layers):
            # Create layer info
            layer_info = LayerInfoExtractor.extract_keras_layer_info(layer, layer.name)
            
            # Set color based on type
            layer_info.color = layer_info.get_color_by_type()
            
            # Calculate size
            layer_info.size = layer_info.calculate_size()
            
            layers.append(layer_info)
            
            # Extract connections from layer's input/output relationships
            connections[layer.name] = []
            
            # Get the layers that this layer outputs to
            try:
                # For functional API models
                if hasattr(layer, '_outbound_nodes'):
                    for node in layer._outbound_nodes:
                        if hasattr(node, 'outbound_layer'):
                            target_layer = node.outbound_layer
                            if target_layer and hasattr(target_layer, 'name'):
                                connections[layer.name].append(target_layer.name)
                
                # Alternative method for sequential models
                if not connections[layer.name] and i < len(model.layers) - 1:
                    next_layer = model.layers[i + 1]
                    connections[layer.name].append(next_layer.name)
                    
            except (AttributeError, IndexError):
                # Fallback: assume sequential connections
                if i < len(model.layers) - 1:
                    next_layer = model.layers[i + 1]
                    connections[layer.name].append(next_layer.name)
        
        return layers, connections
    
    def get_model_summary(self, model, framework: Optional[str] = None, 
                         input_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Args:
            model: Neural network model
            framework: Framework hint
            input_shape: Input tensor shape (for PyTorch)
            
        Returns:
            Dictionary containing model summary information
        """
        layers, connections = self.parse_model(model, framework, input_shape)
        
        total_params = sum(layer.parameters for layer in layers)
        trainable_params = sum(layer.trainable_params for layer in layers)
        
        layer_types = {}
        for layer in layers:
            layer_type = layer.layer_type
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += 1
        
        return {
            'total_layers': len(layers),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'layer_types': layer_types,
            'input_shape': layers[0].input_shape if layers else None,
            'output_shape': layers[-1].output_shape if layers else None,
            'connections_count': sum(len(targets) for targets in connections.values()),
        }
    
    def validate_model(self, model, framework: Optional[str] = None) -> List[str]:
        """
        Validate if the model can be parsed and visualized.
        
        Args:
            model: Neural network model
            framework: Framework hint
            
        Returns:
            List of validation warnings/errors
        """
        warnings_list = []
        
        try:
            # Detect framework
            if framework is None:
                framework = self._detect_framework(model)
            
            # Framework-specific validation
            if framework == 'pytorch':
                if not self.torch_available:
                    warnings_list.append("PyTorch is not available")
                elif not isinstance(model, torch.nn.Module):
                    warnings_list.append("Model is not a PyTorch nn.Module")
                
            elif framework == 'keras':
                if not self.tf_available:
                    warnings_list.append("TensorFlow/Keras is not available")
                elif not isinstance(model, (tf.keras.Model, keras.Model)):
                    warnings_list.append("Model is not a Keras Model")
            
            # Try to get layer count
            if framework == 'pytorch':
                layer_count = len(list(model.named_modules()))
                if layer_count == 0:
                    warnings_list.append("No layers found in PyTorch model")
            elif framework == 'keras':
                layer_count = len(model.layers)
                if layer_count == 0:
                    warnings_list.append("No layers found in Keras model")
            
        except Exception as e:
            warnings_list.append(f"Error during model validation: {str(e)}")
        
        return warnings_list 
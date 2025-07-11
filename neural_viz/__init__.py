"""
Neural Viz 3D - A Python package for 3D visualization of neural network architectures.

This package provides tools to visualize neural networks from PyTorch and Keras/TensorFlow
in interactive 3D plots.
"""

__version__ = "0.1.0"
__author__ = "Neural Viz Team"
__email__ = "contact@neuralviz.com"

from .core.visualizer import NeuralNetworkVisualizer
from .core.parser import ModelParser
from .renderers.plotly_renderer import PlotlyRenderer
from .renderers.matplotlib_renderer import MatplotlibRenderer
from .utils.layer_info import LayerInfo
from .utils.position_calculator import PositionCalculator

# Main API functions
def visualize_pytorch(model, input_shape=None, renderer='plotly', **kwargs):
    """
    Visualize a PyTorch model in 3D.
    
    Args:
        model: PyTorch model (torch.nn.Module)
        input_shape: Input tensor shape (tuple)
        renderer: Rendering backend ('plotly' or 'matplotlib')
        **kwargs: Additional visualization parameters
    
    Returns:
        Visualization object
    """
    visualizer = NeuralNetworkVisualizer(renderer=renderer)
    return visualizer.visualize_pytorch(model, input_shape, **kwargs)

def visualize_keras(model, renderer='plotly', **kwargs):
    """
    Visualize a Keras/TensorFlow model in 3D.
    
    Args:
        model: Keras model (tf.keras.Model)
        renderer: Rendering backend ('plotly' or 'matplotlib')
        **kwargs: Additional visualization parameters
    
    Returns:
        Visualization object
    """
    visualizer = NeuralNetworkVisualizer(renderer=renderer)
    return visualizer.visualize_keras(model, **kwargs)

def visualize_model(model, framework=None, input_shape=None, renderer='plotly', **kwargs):
    """
    Auto-detect framework and visualize model in 3D.
    
    Args:
        model: Neural network model (PyTorch or Keras)
        framework: Framework hint ('pytorch' or 'keras'), auto-detected if None
        input_shape: Input tensor shape (tuple), required for PyTorch
        renderer: Rendering backend ('plotly' or 'matplotlib')
        **kwargs: Additional visualization parameters
    
    Returns:
        Visualization object
    """
    visualizer = NeuralNetworkVisualizer(renderer=renderer)
    return visualizer.visualize(model, framework, input_shape, **kwargs)

# Public API
__all__ = [
    'NeuralNetworkVisualizer',
    'ModelParser',
    'PlotlyRenderer',
    'MatplotlibRenderer',
    'LayerInfo',
    'PositionCalculator',
    'visualize_pytorch',
    'visualize_keras',
    'visualize_model',
] 
"""
PyTorch Viz 3D - A PyTorch-specific package for 3D visualization of neural network architectures.

This package provides tools to visualize PyTorch neural networks in interactive 3D plots
with deep integration into the PyTorch ecosystem.
"""

__version__ = "0.1.0"
__author__ = "PyTorch Viz Team"
__email__ = "contact@pytorchviz.com"

from .core.visualizer import PyTorchVisualizer
from .core.parser import PyTorchModelParser
from .renderers.plotly_renderer import PlotlyRenderer
from .renderers.matplotlib_renderer import MatplotlibRenderer
from .utils.layer_info import LayerInfo
from .utils.position_calculator import PositionCalculator
from .utils.pytorch_hooks import HookManager, ActivationExtractor
from .utils.model_analyzer import ModelAnalyzer

# Main API functions
def visualize(model, input_shape=None, renderer='plotly', **kwargs):
    """
    Visualize a PyTorch model in 3D.
    
    Args:
        model: PyTorch model (torch.nn.Module)
        input_shape: Input tensor shape (tuple), if None will try to infer
        renderer: Rendering backend ('plotly' or 'matplotlib')
        **kwargs: Additional visualization parameters
    
    Returns:
        Visualization object
    """
    visualizer = PyTorchVisualizer(renderer=renderer)
    return visualizer.visualize(model, input_shape, **kwargs)

def visualize_model(model, input_shape=None, renderer='plotly', **kwargs):
    """
    Alias for visualize() function for backward compatibility.
    """
    return visualize(model, input_shape, renderer, **kwargs)

def analyze_model(model, input_shape=None, detailed=True):
    """
    Analyze a PyTorch model and return detailed statistics.
    
    Args:
        model: PyTorch model (torch.nn.Module)
        input_shape: Input tensor shape (tuple)
        detailed: Whether to include detailed layer analysis
    
    Returns:
        Dictionary containing model analysis
    """
    analyzer = ModelAnalyzer()
    return analyzer.analyze(model, input_shape, detailed)

def compare_models(models, names=None, input_shapes=None, renderer='plotly', **kwargs):
    """
    Compare multiple PyTorch models in a single visualization.
    
    Args:
        models: List of PyTorch models
        names: Optional list of model names
        input_shapes: Optional list of input shapes for each model
        renderer: Rendering backend ('plotly' or 'matplotlib')
        **kwargs: Additional visualization parameters
    
    Returns:
        Comparison visualization object
    """
    visualizer = PyTorchVisualizer(renderer=renderer)
    return visualizer.compare_models(models, names, input_shapes, **kwargs)

def create_architecture_report(model, input_shape=None, output_path="pytorch_report.html"):
    """
    Create a comprehensive HTML report of the PyTorch model architecture.
    
    Args:
        model: PyTorch model (torch.nn.Module)
        input_shape: Input tensor shape (tuple)
        output_path: Path for the output HTML file
    """
    visualizer = PyTorchVisualizer()
    visualizer.export_architecture_report(model, input_shape, output_path)

# PyTorch-specific utilities
def profile_model(model, input_shape, device='cpu'):
    """
    Profile a PyTorch model for performance analysis.
    
    Args:
        model: PyTorch model (torch.nn.Module)
        input_shape: Input tensor shape (tuple)
        device: Device to run profiling on ('cpu' or 'cuda')
    
    Returns:
        Profiling results dictionary
    """
    analyzer = ModelAnalyzer()
    return analyzer.profile_model(model, input_shape, device)

def extract_activations(model, input_tensor, layer_names=None):
    """
    Extract intermediate activations from a PyTorch model.
    
    Args:
        model: PyTorch model (torch.nn.Module)
        input_tensor: Input tensor for forward pass
        layer_names: Specific layer names to extract (if None, extracts all)
    
    Returns:
        Dictionary of layer names to activation tensors
    """
    extractor = ActivationExtractor(model)
    return extractor.extract(input_tensor, layer_names)

# Public API
__all__ = [
    'PyTorchVisualizer',
    'PyTorchModelParser', 
    'PlotlyRenderer',
    'MatplotlibRenderer',
    'LayerInfo',
    'PositionCalculator',
    'HookManager',
    'ActivationExtractor',
    'ModelAnalyzer',
    'visualize',
    'visualize_model',
    'analyze_model',
    'compare_models',
    'create_architecture_report',
    'profile_model',
    'extract_activations',
] 
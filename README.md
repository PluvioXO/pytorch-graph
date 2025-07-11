# PyTorch Viz 3D

A **PyTorch-specific** package for **3D visualization of neural network architectures**. Transform your PyTorch models into stunning, interactive 3D visualizations with deep integration into the PyTorch ecosystem.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://pytorch-viz-3d.readthedocs.io/)

## Features

- **PyTorch Native**: Built specifically for PyTorch with deep ecosystem integration
- **Interactive 3D Visualization**: Powered by Plotly for smooth, interactive exploration
- **Multiple Layout Algorithms**: Hierarchical, circular, spring, and custom layouts
- **Rich Layer Information**: Detailed hover information with parameters, shapes, and metadata
- **Connection Visualization**: Clear representation of data flow between layers
- **Performance Profiling**: Built-in PyTorch profiling and performance analysis
- **Activation Analysis**: Extract and visualize intermediate activations
- **Memory Analysis**: Comprehensive memory usage tracking
- **Customizable Themes**: Multiple color themes and styling options
- **Export Options**: Save as HTML, PNG, SVG, or PDF
- **Model Comparison**: Compare multiple architectures side-by-side
- **Responsive Design**: Works on desktop and mobile devices

## Installation

### Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 1.8.0 or higher
- **Operating System**: Windows, macOS, or Linux

### Quick Install

```bash
pip install pytorch-viz-3d
```

### Development Install

```bash
git clone https://github.com/yourusername/pytorch-viz-3d.git
cd pytorch-viz-3d
pip install -e .
```

### Optional Dependencies

For enhanced features:

```bash
# For advanced model analysis
pip install torchinfo torchsummary

# For FLOP counting
pip install thop

# For additional export formats
pip install kaleido

# For development
pip install pytest black flake8 mypy
```

### Verification

Test your installation:

```python
import pytorch_viz
print(pytorch_viz.__version__)
```

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
import pytorch_viz

# Define your PyTorch model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN()

# Visualize in 3D!
fig = pytorch_viz.visualize(
    model, 
    input_shape=(3, 32, 32),
    title="My CNN Architecture"
)
fig.show()
```

### One-Line Visualization

```python
# Minimal code for quick visualization
pytorch_viz.visualize(model, input_shape=(3, 224, 224)).show()
```

## Advanced Usage

### Custom Visualization Settings

```python
from pytorch_viz import PyTorchVisualizer

# Create visualizer with custom settings
visualizer = PyTorchVisualizer(
    layout_style='spring',      # 'hierarchical', 'circular', 'spring', 'custom'
    theme='plotly_dark',        # 'plotly', 'plotly_white', 'plotly_dark'
    spacing=2.0,                # Space between layers
    width=1400,                 # Figure width
    height=900                  # Figure height
)

# Advanced visualization with PyTorch-specific features
fig = visualizer.visualize(
    model,
    input_shape=(3, 224, 224),
    title="ResNet Architecture",
    show_connections=True,      # Show layer connections
    show_labels=True,           # Show layer names
    show_parameters=True,       # Show parameter count chart
    show_activations=True,      # Include activation statistics
    optimize_layout=True,       # Optimize layer positions
    device='cuda',              # Use GPU for analysis
    export_path="model.html"    # Auto-export to HTML
)
```

### PyTorch-Specific Features

```python
# Model Analysis
analysis = pytorch_viz.analyze_model(
    model, 
    input_shape=(3, 224, 224),
    detailed=True
)
print(f"Total parameters: {analysis['basic_info']['total_parameters']:,}")
print(f"Memory usage: {analysis['memory']['total_memory_mb']:.2f} MB")

# Performance Profiling
profiling = pytorch_viz.profile_model(
    model,
    input_shape=(3, 224, 224),
    device='cuda',
    num_runs=100
)
print(f"Inference time: {profiling['mean_time_ms']:.2f} ms")
print(f"Throughput: {profiling['fps']:.1f} FPS")

# Activation Extraction
activations = pytorch_viz.extract_activations(
    model,
    torch.randn(1, 3, 224, 224),
    layer_names=['conv1', 'conv2']
)

# Comprehensive Report
pytorch_viz.create_architecture_report(
    model,
    input_shape=(3, 224, 224),
    output_path="pytorch_report.html"
)
```

### Model Comparison

```python
# Compare multiple models
models = [resnet18, resnet34, resnet50]
names = ["ResNet-18", "ResNet-34", "ResNet-50"]

fig = pytorch_viz.compare_models(
    models=models,
    names=names,
    input_shapes=[(3, 224, 224)] * 3,
    device='cuda'
)
fig.show()
```

## Layout Styles

### Hierarchical Layout
```python
# Best for sequential models like CNNs and MLPs
visualizer = PyTorchVisualizer(layout_style='hierarchical')
```
- **Use case**: Sequential models, CNNs, simple MLPs
- **Characteristics**: Top-down flow, clear layer progression
- **Best for**: Understanding data flow in feed-forward networks

### Spring Layout
```python
# Best for complex architectures with skip connections
visualizer = PyTorchVisualizer(layout_style='spring')
```
- **Use case**: ResNets, DenseNets, complex architectures
- **Characteristics**: Physics-based positioning, natural clustering
- **Best for**: Models with skip connections and complex topologies

### Circular Layout
```python
# Best for small models or artistic visualization
visualizer = PyTorchVisualizer(layout_style='circular')
```
- **Use case**: Small models, artistic presentations
- **Characteristics**: Circular arrangement, symmetric appearance
- **Best for**: Presentations and aesthetic visualizations

### Custom Layout
```python
# Groups similar layer types together
visualizer = PyTorchVisualizer(layout_style='custom')
```
- **Use case**: Analysis and debugging
- **Characteristics**: Groups by layer type, organized clustering
- **Best for**: Understanding model composition and layer distribution

## Supported PyTorch Layers

### Convolutional Layers
- `Conv1d`, `Conv2d`, `Conv3d` - Standard convolutions
- `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` - Transposed convolutions
- `LazyConv1d`, `LazyConv2d`, `LazyConv3d` - Lazy convolutions

### Pooling Layers
- `MaxPool1d`, `MaxPool2d`, `MaxPool3d` - Max pooling
- `AvgPool1d`, `AvgPool2d`, `AvgPool3d` - Average pooling
- `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d` - Adaptive max pooling
- `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d` - Adaptive average pooling

### Linear Layers
- `Linear` - Fully connected layers
- `LazyLinear` - Lazy linear layers
- `Bilinear` - Bilinear transformation

### Activation Functions
- `ReLU`, `ReLU6`, `LeakyReLU`, `PReLU`, `ELU`, `SELU`, `CELU` - ReLU variants
- `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax` - Classical activations
- `GELU`, `SiLU`, `Mish`, `Swish` - Modern activations
- `GLU`, `MultiheadAttention` - Gated and attention activations

### Normalization Layers
- `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d` - Batch normalization
- `LayerNorm` - Layer normalization
- `GroupNorm` - Group normalization
- `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d` - Instance normalization
- `LocalResponseNorm` - Local response normalization

### Regularization Layers
- `Dropout`, `Dropout1d`, `Dropout2d`, `Dropout3d` - Standard dropout
- `AlphaDropout` - Alpha dropout for SELU
- `FeatureAlphaDropout` - Feature-wise alpha dropout

### Recurrent Layers
- `RNN`, `LSTM`, `GRU` - Standard RNN cells
- `RNNCell`, `LSTMCell`, `GRUCell` - Single-step RNN cells

### Attention & Transformer Layers
- `MultiheadAttention` - Multi-head attention mechanism
- `TransformerEncoder`, `TransformerDecoder` - Transformer blocks
- `TransformerEncoderLayer`, `TransformerDecoderLayer` - Individual transformer layers

### Embedding Layers
- `Embedding` - Standard embedding
- `EmbeddingBag` - Embedding with reduction

### Loss Functions (when part of model)
- `CrossEntropyLoss`, `MSELoss`, `BCELoss` - Standard losses
- `NLLLoss`, `PoissonNLLLoss`, `KLDivLoss` - Specialized losses

## PyTorch-Specific Features

### Advanced Analysis

#### Memory Profiling
```python
analysis = pytorch_viz.analyze_model(model, input_shape=(3, 224, 224))
memory_info = analysis['memory']

print(f"Parameter memory: {memory_info['parameters_mb']:.2f} MB")
print(f"Activation memory: {memory_info['activations_mb']:.2f} MB")
print(f"Peak memory usage: {memory_info['total_memory_mb']:.2f} MB")
```

#### FLOP Counting
```python
# Requires torchinfo or thop
analysis = pytorch_viz.analyze_model(model, input_shape=(3, 224, 224))
if 'complexity' in analysis:
    flops = analysis['complexity']['flops']
    print(f"Model FLOPs: {flops:,}")
```

#### Parameter Statistics
```python
analysis = pytorch_viz.analyze_model(model, detailed=True)
params = analysis['parameters']

print(f"Total parameters: {params['statistics']['total_params']:,}")
print(f"Mean weight: {params['statistics']['mean']:.4f}")
print(f"Weight std: {params['statistics']['std']:.4f}")
```

### Performance Insights

#### GPU/CPU Profiling
```python
# CPU profiling
cpu_profile = pytorch_viz.profile_model(model, input_shape, device='cpu')

# GPU profiling (if CUDA available)
if torch.cuda.is_available():
    gpu_profile = pytorch_viz.profile_model(model, input_shape, device='cuda')
    print(f"CPU time: {cpu_profile['mean_time_ms']:.2f} ms")
    print(f"GPU time: {gpu_profile['mean_time_ms']:.2f} ms")
    print(f"Speedup: {cpu_profile['mean_time_ms'] / gpu_profile['mean_time_ms']:.1f}x")
```

#### Batch Size Analysis
```python
batch_sizes = [1, 8, 16, 32, 64]
for batch_size in batch_sizes:
    input_shape = (batch_size, 3, 224, 224)
    profile = pytorch_viz.profile_model(model, input_shape)
    throughput = profile['fps'] * batch_size
    print(f"Batch {batch_size}: {throughput:.1f} images/sec")
```

### Activation Visualization

#### Feature Map Extraction
```python
# For CNN models
visualizer = pytorch_viz.PyTorchVisualizer()
feature_maps = visualizer.visualize_feature_maps(
    model,
    torch.randn(1, 3, 224, 224),
    layer_names=['conv1', 'conv2'],
    max_channels=16
)
```

#### Activation Statistics
```python
activations = pytorch_viz.extract_activations(model, input_tensor)
for layer_name, activation in activations.items():
    if isinstance(activation, torch.Tensor):
        sparsity = (activation == 0).float().mean()
        print(f"{layer_name}: {sparsity:.2%} sparsity")
```

#### Gradient Analysis
```python
from pytorch_viz.utils.pytorch_hooks import ActivationExtractor

extractor = ActivationExtractor(model)
result = extractor.extract_with_gradients(input_tensor.requires_grad_(True))

activations = result['activations']
gradients = result['gradients']
```

## Customization

### Color Themes

#### Built-in Themes
```python
themes = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn']

for theme in themes:
    visualizer = PyTorchVisualizer(theme=theme)
    fig = visualizer.visualize(model, input_shape)
    fig.write_html(f"model_{theme}.html")
```

#### Custom Colors
```python
# Custom layer colors
color_map = {
    'Conv2d': '#FF6B6B',      # Red for convolutions
    'Linear': '#4ECDC4',      # Teal for linear layers
    'BatchNorm2d': '#45B7D1', # Blue for normalization
    'ReLU': '#96CEB4',        # Green for activations
    'MaxPool2d': '#FFEAA7'    # Yellow for pooling
}

visualizer = PyTorchVisualizer(theme='plotly_white')
# Apply custom colors through renderer
visualizer.renderer.layer_colors = color_map
```

### Layout Customization

#### Custom Spacing
```python
# Adjust spacing between layers
visualizer = PyTorchVisualizer(spacing=3.0)  # Larger spacing
fig = visualizer.visualize(model, input_shape)
```

#### Position Optimization
```python
# Enable position optimization for better layout
fig = visualizer.visualize(
    model, 
    input_shape,
    optimize_layout=True,
    iterations=100  # More iterations for better positioning
)
```

### Export Options

#### Interactive HTML
```python
# Export interactive visualization
fig = pytorch_viz.visualize(model, input_shape)
fig.write_html("model.html", include_plotlyjs='cdn')
```

#### Static Images
```python
# Requires kaleido: pip install kaleido
fig = pytorch_viz.visualize(model, input_shape)
fig.write_image("model.png", width=1920, height=1080, scale=2)
fig.write_image("model.svg")  # Vector format
fig.write_image("model.pdf")  # PDF format
```

#### Comprehensive Reports
```python
# Generate detailed HTML report
pytorch_viz.create_architecture_report(
    model,
    input_shape=(3, 224, 224),
    output_path="model_report.html",
    include_profiling=True
)
```

## API Reference

### Main Functions

#### `pytorch_viz.visualize()`
```python
pytorch_viz.visualize(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    title: Optional[str] = None,
    show_connections: bool = True,
    show_labels: bool = True,
    show_parameters: bool = False,
    show_activations: bool = False,
    optimize_layout: bool = True,
    device: str = 'auto',
    export_path: Optional[str] = None,
    **kwargs
) -> plotly.graph_objects.Figure
```

**Parameters:**
- `model`: PyTorch model to visualize
- `input_shape`: Input tensor shape (excluding batch dimension)
- `title`: Plot title (auto-generated if None)
- `show_connections`: Display layer connections
- `show_labels`: Display layer names
- `show_parameters`: Include parameter count visualization
- `show_activations`: Include activation statistics
- `optimize_layout`: Apply position optimization
- `device`: Device for analysis ('auto', 'cpu', 'cuda')
- `export_path`: Auto-export path (HTML, PNG, SVG, PDF)

#### `pytorch_viz.analyze_model()`
```python
pytorch_viz.analyze_model(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    detailed: bool = True,
    device: str = 'auto'
) -> Dict[str, Any]
```

**Returns:**
```python
{
    'basic_info': {
        'total_parameters': int,
        'trainable_parameters': int,
        'total_layers': int,
        'device': str,
        'layer_types': Dict[str, int]
    },
    'memory': {
        'parameters_mb': float,
        'activations_mb': float,
        'total_memory_mb': float
    },
    'architecture': {
        'depth': int,
        'conv_layers': int,
        'linear_layers': int,
        'patterns': List[str]
    },
    'parameters': {
        'statistics': Dict[str, float],
        'by_layer': Dict[str, Dict]
    }
}
```

#### `pytorch_viz.profile_model()`
```python
pytorch_viz.profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu',
    num_runs: int = 100
) -> Dict[str, Any]
```

**Returns:**
```python
{
    'mean_time_ms': float,
    'std_time_ms': float,
    'min_time_ms': float,
    'max_time_ms': float,
    'fps': float,
    'peak_memory_mb': float,  # CUDA only
    'memory_reserved_mb': float  # CUDA only
}
```

#### `pytorch_viz.extract_activations()`
```python
pytorch_viz.extract_activations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]
```

#### `pytorch_viz.compare_models()`
```python
pytorch_viz.compare_models(
    models: List[nn.Module],
    names: Optional[List[str]] = None,
    input_shapes: Optional[List[Tuple[int, ...]]] = None,
    device: str = 'auto',
    **kwargs
) -> plotly.graph_objects.Figure
```

### PyTorchVisualizer Class

#### Constructor
```python
PyTorchVisualizer(
    renderer: str = 'plotly',
    layout_style: str = 'hierarchical',
    spacing: float = 2.0,
    theme: str = 'plotly_dark',
    width: int = 1200,
    height: int = 800
)
```

#### Methods
```python
# Visualization
visualizer.visualize(model, input_shape, **kwargs)
visualizer.compare_models(models, names, input_shapes, **kwargs)
visualizer.visualize_feature_maps(model, input_tensor, **kwargs)

# Analysis
visualizer.analyze_model(model, input_shape, **kwargs)
visualizer.profile_model(model, input_shape, device, **kwargs)
visualizer.get_model_summary(model, input_shape, **kwargs)

# Export
visualizer.export_architecture_report(model, input_shape, output_path)

# Configuration
visualizer.set_theme(theme)
visualizer.set_layout_style(layout_style)
visualizer.set_spacing(spacing)
```

## Examples

### Complete Examples

#### 1. ResNet Visualization
```python
import torch
import torchvision.models as models
import pytorch_viz

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)
model.eval()

# Create comprehensive visualization
visualizer = pytorch_viz.PyTorchVisualizer(
    layout_style='spring',
    theme='plotly_dark',
    spacing=2.5
)

fig = visualizer.visualize(
    model,
    input_shape=(3, 224, 224),
    title="ResNet-50 Architecture",
    show_connections=True,
    show_parameters=True,
    optimize_layout=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

fig.show()

# Generate detailed report
pytorch_viz.create_architecture_report(
    model,
    input_shape=(3, 224, 224),
    output_path="resnet50_report.html"
)
```

#### 2. Custom CNN with Analysis
```python
import torch
import torch.nn as nn
import pytorch_viz

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create and analyze model
model = CustomCNN()

# Comprehensive analysis
analysis = pytorch_viz.analyze_model(
    model,
    input_shape=(3, 32, 32),
    detailed=True
)

print("=== Model Analysis ===")
print(f"Parameters: {analysis['basic_info']['total_parameters']:,}")
print(f"Memory: {analysis['memory']['total_memory_mb']:.2f} MB")
print(f"Layers: {analysis['basic_info']['total_layers']}")

# Performance profiling
profile = pytorch_viz.profile_model(
    model,
    input_shape=(3, 32, 32),
    device='cpu',
    num_runs=50
)

print("\n=== Performance ===")
print(f"Inference time: {profile['mean_time_ms']:.2f} ± {profile['std_time_ms']:.2f} ms")
print(f"Throughput: {profile['fps']:.1f} FPS")

# Visualization
fig = pytorch_viz.visualize(
    model,
    input_shape=(3, 32, 32),
    title="Custom CNN - CIFAR-10 Classifier",
    show_activations=True,
    show_parameters=True
)
fig.show()
```

#### 3. Transformer Visualization
```python
import torch
import torch.nn as nn
import pytorch_viz

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(5000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = self.output_projection(x)
        return x

# Create transformer model
model = SimpleTransformer()

# Note: For transformers, we use sequence length instead of image dimensions
# input_shape represents (sequence_length,) for embedding input
fig = pytorch_viz.visualize(
    model,
    input_shape=(512,),  # Sequence length
    title="Transformer Encoder Architecture",
    layout_style='hierarchical',
    show_connections=True
)
fig.show()
```

#### 4. Model Comparison
```python
import torch
import torchvision.models as models
import pytorch_viz

# Load different models
models_to_compare = [
    models.resnet18(pretrained=False),
    models.resnet34(pretrained=False),
    models.resnet50(pretrained=False)
]

model_names = ["ResNet-18", "ResNet-34", "ResNet-50"]

# Compare architectures
comparison_fig = pytorch_viz.compare_models(
    models=models_to_compare,
    names=model_names,
    input_shapes=[(3, 224, 224)] * 3,
    device='cpu'
)
comparison_fig.show()

# Detailed comparison analysis
for name, model in zip(model_names, models_to_compare):
    analysis = pytorch_viz.analyze_model(model, (3, 224, 224))
    profile = pytorch_viz.profile_model(model, (3, 224, 224), num_runs=10)
    
    print(f"\n=== {name} ===")
    print(f"Parameters: {analysis['basic_info']['total_parameters']:,}")
    print(f"Memory: {analysis['memory']['total_memory_mb']:.2f} MB")
    print(f"Inference: {profile['mean_time_ms']:.2f} ms")
    print(f"Throughput: {profile['fps']:.1f} FPS")
```

## Advanced Features

### Custom Hooks and Analysis

#### Custom Hook Functions
```python
from pytorch_viz.utils.pytorch_hooks import HookManager

# Create custom hook manager
hook_manager = HookManager()

def custom_analysis_hook(name):
    def hook_fn(module, input, output):
        # Custom analysis logic
        if isinstance(output, torch.Tensor):
            print(f"{name}: mean={output.mean():.3f}, std={output.std():.3f}")
    return hook_fn

# Apply to model
model = YourModel()
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        hook_manager.register_forward_hook(module, name, custom_analysis_hook(name))

# Run forward pass
with torch.no_grad():
    output = model(input_tensor)

# Clean up
hook_manager.remove_all_hooks()
```

#### Gradient Analysis
```python
from pytorch_viz.utils.pytorch_hooks import ActivationExtractor

# Extract gradients during backward pass
extractor = ActivationExtractor(model)
result = extractor.extract_with_gradients(
    input_tensor.requires_grad_(True),
    layer_names=['conv1', 'fc']
)

activations = result['activations']
gradients = result['gradients']

# Analyze gradient magnitudes
for layer_name, grad in gradients.items():
    if grad is not None:
        grad_norm = grad.norm()
        print(f"{layer_name} gradient norm: {grad_norm:.4f}")
```

### Performance Optimization

#### Memory Optimization
```python
# Analyze memory bottlenecks
analysis = pytorch_viz.analyze_model(model, input_shape, detailed=True)
memory_by_layer = analysis['memory']['activation_sizes_by_layer']

# Find memory-intensive layers
sorted_layers = sorted(memory_by_layer.items(), key=lambda x: x[1], reverse=True)
print("Top 5 memory-consuming layers:")
for layer_name, memory_mb in sorted_layers[:5]:
    print(f"  {layer_name}: {memory_mb:.2f} MB")
```

#### FLOP Analysis
```python
# Detailed FLOP analysis (requires thop)
try:
    from thop import profile, clever_format
    
    input_tensor = torch.randn(1, *input_shape)
    flops, params = profile(model, inputs=(input_tensor,))
    
    flops_formatted = clever_format([flops], "%.3f")
    params_formatted = clever_format([params], "%.3f")
    
    print(f"FLOPs: {flops_formatted[0]}")
    print(f"Params: {params_formatted[0]}")
    
except ImportError:
    print("Install thop for detailed FLOP analysis: pip install thop")
```

### Integration with Training

#### Training Loop Integration
```python
import pytorch_viz
from torch.utils.tensorboard import SummaryWriter

class TrainingVisualizer:
    def __init__(self, model, input_shape, log_dir="logs"):
        self.model = model
        self.input_shape = input_shape
        self.writer = SummaryWriter(log_dir)
        
    def log_architecture(self, epoch=0):
        """Log model architecture to TensorBoard"""
        fig = pytorch_viz.visualize(
            self.model,
            self.input_shape,
            title=f"Model Architecture - Epoch {epoch}"
        )
        
        # Convert to image and log
        img_bytes = fig.to_image(format="png")
        # Log to TensorBoard (implementation depends on your setup)
        
    def log_activations(self, input_batch, epoch):
        """Log activation statistics"""
        activations = pytorch_viz.extract_activations(
            self.model,
            input_batch[0:1],  # Single sample
        )
        
        for layer_name, activation in activations.items():
            if isinstance(activation, torch.Tensor):
                self.writer.add_histogram(
                    f"activations/{layer_name}",
                    activation.flatten(),
                    epoch
                )
    
    def profile_epoch(self, epoch):
        """Profile model performance"""
        profile = pytorch_viz.profile_model(
            self.model,
            self.input_shape,
            num_runs=10
        )
        
        self.writer.add_scalar("performance/inference_time_ms", 
                              profile['mean_time_ms'], epoch)
        self.writer.add_scalar("performance/throughput_fps", 
                              profile['fps'], epoch)

# Usage in training loop
visualizer = TrainingVisualizer(model, input_shape)

for epoch in range(num_epochs):
    # Training code...
    
    if epoch % 10 == 0:  # Log every 10 epochs
        visualizer.log_architecture(epoch)
        visualizer.log_activations(val_batch, epoch)
        visualizer.profile_epoch(epoch)
```

## Troubleshooting

### Common Issues

#### Installation Issues

**Problem**: `pip install pytorch-viz-3d` fails
```bash
# Solution 1: Update pip
pip install --upgrade pip
pip install pytorch-viz-3d

# Solution 2: Use conda
conda install pytorch torchvision
pip install pytorch-viz-3d

# Solution 3: Install from source
git clone https://github.com/yourusername/pytorch-viz-3d.git
cd pytorch-viz-3d
pip install -e .
```

**Problem**: Missing dependencies
```bash
# Install all optional dependencies
pip install pytorch-viz-3d[dev,docs,examples]

# Or install manually
pip install torchinfo torchsummary thop kaleido
```

#### Visualization Issues

**Problem**: Empty or incomplete visualizations
```python
# Solution: Ensure input_shape is provided
fig = pytorch_viz.visualize(
    model, 
    input_shape=(3, 224, 224),  # Required for detailed analysis
    device='cpu'  # Specify device explicitly
)
```

**Problem**: Model parsing fails
```python
# Solution: Validate model first
warnings = pytorch_viz.PyTorchVisualizer().parser.validate_model(model)
for warning in warnings:
    print(f"Warning: {warning}")

# Ensure model is in eval mode
model.eval()
```

#### Performance Issues

**Problem**: Slow visualization for large models
```python
# Solution: Disable expensive features
fig = pytorch_viz.visualize(
    model,
    input_shape=input_shape,
    show_activations=False,  # Disable activation analysis
    optimize_layout=False,   # Disable layout optimization
    device='cpu'             # Use CPU for analysis
)
```

**Problem**: CUDA out of memory during profiling
```python
# Solution: Use smaller batch size or CPU
profile = pytorch_viz.profile_model(
    model,
    input_shape=input_shape,
    device='cpu',  # Use CPU instead of GPU
    num_runs=10    # Reduce number of runs
)
```

#### Export Issues

**Problem**: Cannot export images
```bash
# Solution: Install kaleido
pip install kaleido

# Alternative: Export as HTML
fig.write_html("model.html")
```

**Problem**: Large HTML files
```python
# Solution: Use CDN for Plotly
fig.write_html("model.html", include_plotlyjs='cdn')

# Or use compressed format
fig.write_html("model.html", config={'plotlyServerURL': 'https://cdn.plot.ly'})
```

### Debugging Tips

#### Enable Verbose Output
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed parsing and analysis information
fig = pytorch_viz.visualize(model, input_shape)
```

#### Check Model Compatibility
```python
# Validate model structure
from pytorch_viz.core.parser import PyTorchModelParser

parser = PyTorchModelParser()
warnings = parser.validate_model(model)

if warnings:
    print("Model validation warnings:")
    for warning in warnings:
        print(f"  - {warning}")
else:
    print("Model validation passed!")
```

#### Memory Debugging
```python
import torch
import gc

# Clear GPU memory before analysis
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Run analysis
analysis = pytorch_viz.analyze_model(model, input_shape)

if torch.cuda.is_available():
    print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    torch.cuda.empty_cache()

# Force garbage collection
gc.collect()
```

## Best Practices

### Model Design

1. **Layer Naming**: Use descriptive names for your layers
```python
class WellNamedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, name="input_conv"),
            nn.BatchNorm2d(64, name="input_bn"),
            nn.ReLU(inplace=True, name="input_relu"),
            # ... more layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256, name="classifier_fc1"),
            nn.ReLU(inplace=True, name="classifier_relu"),
            nn.Linear(256, 10, name="output_fc")
        )
```

2. **Modular Design**: Organize complex models into modules
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # This will be clearly visible in visualization
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
```

### Visualization Workflow

1. **Start Simple**: Begin with basic visualization
```python
# Step 1: Basic visualization
fig = pytorch_viz.visualize(model, input_shape)
fig.show()

# Step 2: Add details
fig = pytorch_viz.visualize(
    model, input_shape,
    show_parameters=True,
    show_connections=True
)

# Step 3: Full analysis
analysis = pytorch_viz.analyze_model(model, input_shape, detailed=True)
pytorch_viz.create_architecture_report(model, input_shape, "report.html")
```

2. **Iterative Development**: Use visualization during development
```python
def develop_model():
    """Iterative model development with visualization"""
    
    # Start with simple model
    model_v1 = SimpleCNN()
    fig1 = pytorch_viz.visualize(model_v1, input_shape)
    fig1.show()
    
    # Add complexity
    model_v2 = ImprovedCNN()
    fig2 = pytorch_viz.visualize(model_v2, input_shape)
    
    # Compare versions
    comparison = pytorch_viz.compare_models(
        [model_v1, model_v2],
        ["Simple", "Improved"],
        [input_shape, input_shape]
    )
    comparison.show()
```

3. **Performance Monitoring**: Regular profiling
```python
def monitor_performance(model, input_shape):
    """Monitor model performance across development"""
    
    # Baseline profiling
    baseline = pytorch_viz.profile_model(model, input_shape)
    
    # After optimization
    model_optimized = optimize_model(model)
    optimized = pytorch_viz.profile_model(model_optimized, input_shape)
    
    print(f"Speedup: {baseline['mean_time_ms'] / optimized['mean_time_ms']:.2f}x")
    print(f"Memory reduction: {baseline.get('peak_memory_mb', 0) - optimized.get('peak_memory_mb', 0):.2f} MB")
```

### Export and Sharing

1. **Interactive Reports**: For detailed analysis
```python
pytorch_viz.create_architecture_report(
    model,
    input_shape,
    "detailed_report.html",
    include_profiling=True
)
```

2. **Publication Figures**: High-quality static images
```python
fig = pytorch_viz.visualize(
    model, input_shape,
    title="Model Architecture",
    theme='plotly_white'  # Better for publications
)

fig.write_image(
    "model_architecture.png",
    width=1920, height=1080,
    scale=2  # High DPI
)
```

3. **Presentations**: Simplified views
```python
# For presentations, use simpler layouts
fig = pytorch_viz.visualize(
    model, input_shape,
    layout_style='circular',
    show_connections=False,  # Cleaner look
    show_labels=True,
    theme='plotly_dark'  # Good for dark slides
)
```

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-viz-3d.git
cd pytorch-viz-3d

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,docs,examples]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

We use black for code formatting and flake8 for linting:

```bash
# Format code
black pytorch_viz/

# Check linting
flake8 pytorch_viz/

# Type checking
mypy pytorch_viz/
```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=pytorch_viz tests/

# Run specific test
pytest tests/test_visualizer.py::test_basic_visualization
```

### Adding New Features

1. **New Layer Support**: Add to `layer_info.py`
```python
# In pytorch_viz/utils/layer_info.py
def extract_pytorch_layer_info(module, name, input_shape, output_shape):
    # Add new layer type handling
    if isinstance(module, YourNewLayerType):
        return LayerInfo(
            name=name,
            layer_type="YourNewLayer",
            # ... other properties
        )
```

2. **New Renderers**: Implement renderer interface
```python
# In pytorch_viz/renderers/your_renderer.py
class YourRenderer:
    def render(self, layers, connections, **kwargs):
        # Implementation
        pass
```

3. **New Analysis Features**: Extend `ModelAnalyzer`
```python
# In pytorch_viz/utils/model_analyzer.py
class ModelAnalyzer:
    def your_new_analysis(self, model, input_shape):
        # Implementation
        return analysis_results
```

### Documentation

- Update docstrings for all public functions
- Add examples for new features
- Update README.md if needed
- Add tests for new functionality

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests
5. Update documentation
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyTorch** team for the amazing deep learning framework
- **Plotly** for the incredible 3D visualization capabilities
- **torchinfo** and **torchsummary** for model analysis tools
- **NetworkX** for graph layout algorithms
- **The PyTorch Community** for inspiration and feedback

## Additional Resources

### Documentation
- [API Reference](https://pytorch-viz-3d.readthedocs.io/en/latest/api/)
- [Tutorials](https://pytorch-viz-3d.readthedocs.io/en/latest/tutorials/)
- [Examples Gallery](https://pytorch-viz-3d.readthedocs.io/en/latest/gallery/)

### Related Projects
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Plotly](https://plotly.com/python/) - Interactive plotting
- [torchinfo](https://github.com/TylerYep/torchinfo) - Model summaries
- [Netron](https://github.com/lutzroeder/netron) - Neural network viewer

### Citation

If you use PyTorch Viz 3D in your research, please cite:

```bibtex
@software{pytorch_viz_3d,
  title={PyTorch Viz 3D: Interactive 3D Visualization for PyTorch Neural Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pytorch-viz-3d}
}
```

## Roadmap

### Near Term (v1.0)
- [ ] **TensorBoard integration** for training visualization
- [ ] **ONNX export** support
- [ ] **Mobile model** optimization analysis
- [ ] **Quantization** visualization

### Medium Term (v1.5)
- [ ] **Distributed training** visualization
- [ ] **AutoML** integration
- [ ] **Real-time training** monitoring
- [ ] **Custom layer** definition support

### Long Term (v2.0)
- [ ] **WebGL renderer** for better performance
- [ ] **VR/AR** visualization support
- [ ] **Collaborative** model analysis
- [ ] **Cloud-based** visualization service

## PyTorch Integration

This package is built specifically for PyTorch and leverages:

- **torch.fx** for computational graph analysis
- **torch.profiler** for performance profiling
- **torch.jit** for model tracing
- **torchinfo** for enhanced model summaries
- **PyTorch hooks** for activation extraction
- **CUDA** memory tracking and optimization

---

**Made with love for the PyTorch community** 
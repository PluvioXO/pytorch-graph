# Neural Viz 3D 🧠✨

A powerful Python package for **3D visualization of neural network architectures**. Transform your PyTorch and Keras models into stunning, interactive 3D visualizations that help you understand and present your deep learning architectures.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-supported-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **🎯 Framework Support**: Works with both PyTorch and Keras/TensorFlow models
- **🌟 Interactive 3D Visualization**: Powered by Plotly for smooth, interactive exploration
- **🎨 Multiple Layout Algorithms**: Hierarchical, circular, spring, and custom layouts
- **📊 Rich Layer Information**: Detailed hover information with parameters, shapes, and metadata
- **🔗 Connection Visualization**: Clear representation of data flow between layers
- **🎭 Customizable Themes**: Multiple color themes and styling options
- **📈 Parameter Analysis**: Built-in parameter counting and visualization
- **📤 Export Options**: Save as HTML, PNG, SVG, or PDF
- **🔄 Model Comparison**: Compare multiple architectures side-by-side
- **📱 Responsive Design**: Works on desktop and mobile devices

## 📦 Installation

```bash
pip install neural-viz-3d
```

Or install from source:

```bash
git clone https://github.com/yourusername/neural-viz-3d.git
cd neural-viz-3d
pip install -e .
```

## 🎯 Quick Start

### PyTorch Models

```python
import torch
import torch.nn as nn
import neural_viz

# Define your model
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
fig = neural_viz.visualize_pytorch(
    model, 
    input_shape=(3, 32, 32),
    title="My CNN Architecture"
)
fig.show()
```

### Keras Models

```python
import tensorflow as tf
from tensorflow.keras import layers
import neural_viz

# Define your model
model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Visualize in 3D!
fig = neural_viz.visualize_keras(
    model,
    title="My Keras CNN"
)
fig.show()
```

## 🎨 Advanced Usage

### Custom Visualization Settings

```python
from neural_viz import NeuralNetworkVisualizer

# Create visualizer with custom settings
visualizer = NeuralNetworkVisualizer(
    layout_style='spring',      # 'hierarchical', 'circular', 'spring', 'custom'
    theme='plotly_dark',        # 'plotly', 'plotly_white', 'plotly_dark'
    spacing=2.0,                # Space between layers
    width=1400,                 # Figure width
    height=900                  # Figure height
)

# Visualize with advanced options
fig = visualizer.visualize_pytorch(
    model,
    input_shape=(3, 224, 224),
    title="ResNet Architecture",
    show_connections=True,      # Show layer connections
    show_labels=True,           # Show layer names
    show_parameters=True,       # Show parameter count chart
    optimize_layout=True,       # Optimize layer positions
    export_path="model.html"    # Auto-export
)
```

### Model Comparison

```python
# Compare multiple models
fig = visualizer.compare_models(
    models=[model1, model2, model3],
    names=["ResNet-18", "ResNet-34", "ResNet-50"],
    frameworks=['pytorch', 'pytorch', 'pytorch'],
    input_shapes=[(3, 224, 224)] * 3
)
fig.show()
```

### Export Options

```python
# Export as HTML (interactive)
fig.write_html("model_architecture.html")

# Export as static image
visualizer.export_image("model.png", format='png', scale=2.0)

# Generate comprehensive report
visualizer.export_summary_report(
    model,
    framework='pytorch',
    input_shape=(3, 224, 224),
    output_path="model_report.html"
)
```

## 🎨 Layout Styles

### Hierarchical Layout
```python
# Best for sequential models like CNNs and MLPs
visualizer = NeuralNetworkVisualizer(layout_style='hierarchical')
```

### Spring Layout
```python
# Best for complex architectures with skip connections
visualizer = NeuralNetworkVisualizer(layout_style='spring')
```

### Circular Layout
```python
# Best for small models or artistic visualization
visualizer = NeuralNetworkVisualizer(layout_style='circular')
```

### Custom Layout
```python
# Groups similar layer types together
visualizer = NeuralNetworkVisualizer(layout_style='custom')
```

## 🎯 Supported Layer Types

### PyTorch
- **Convolutional**: `Conv1d`, `Conv2d`, `Conv3d`
- **Pooling**: `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
- **Linear**: `Linear`
- **Activation**: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- **Normalization**: `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`
- **Regularization**: `Dropout`
- **Recurrent**: `LSTM`, `GRU`, `RNN`

### Keras/TensorFlow
- **Convolutional**: `Conv1D`, `Conv2D`, `Conv3D`
- **Pooling**: `MaxPooling2D`, `AveragePooling2D`, `GlobalAveragePooling2D`
- **Dense**: `Dense`
- **Activation**: `ReLU`, `Sigmoid`, `Softmax`
- **Normalization**: `BatchNormalization`, `LayerNormalization`
- **Regularization**: `Dropout`
- **Recurrent**: `LSTM`, `GRU`, `SimpleRNN`
- **Embedding**: `Embedding`

## 📊 Model Information

Each layer visualization includes:
- **Layer name and type**
- **Input/output shapes**
- **Parameter count** (total and trainable)
- **Layer-specific metadata** (kernel size, activation, etc.)
- **Visual encoding** (size represents complexity, color represents type)

## 🎨 Customization

### Color Themes
- `plotly` - Clean white background
- `plotly_white` - Minimal white theme  
- `plotly_dark` - Dark theme for presentations
- `ggplot2` - ggplot2-inspired theme

### Layer Colors
Different layer types are automatically colored:
- 🔴 **Linear/Dense layers**: Red
- 🔵 **Convolutional layers**: Blue  
- 🟠 **Normalization layers**: Orange
- 🟢 **Activation layers**: Green
- 🟣 **Recurrent layers**: Purple
- ⚫ **Pooling layers**: Dark gray

## 🔧 API Reference

### Main Functions
```python
neural_viz.visualize_pytorch(model, input_shape, **kwargs)
neural_viz.visualize_keras(model, **kwargs)
neural_viz.visualize_model(model, framework=None, **kwargs)
```

### NeuralNetworkVisualizer Class
```python
visualizer = NeuralNetworkVisualizer(
    renderer='plotly',
    layout_style='hierarchical',
    spacing=2.0,
    theme='plotly_dark',
    width=1200,
    height=800
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Plotly** for the amazing 3D visualization capabilities
- **NetworkX** for graph layout algorithms
- **PyTorch** and **TensorFlow** communities for the excellent frameworks

## 📚 Examples

Check out the `examples/` directory for more comprehensive examples:

- `basic_usage.py` - Simple PyTorch and Keras examples
- `advanced_features.py` - Custom layouts and styling
- `model_comparison.py` - Comparing different architectures
- `export_examples.py` - Various export formats

## 🐛 Issues and Feature Requests

Please report issues and request features on our [GitHub Issues](https://github.com/yourusername/neural-viz-3d/issues) page.

## 📈 Roadmap

- [ ] Matplotlib renderer backend
- [ ] Animation support for training visualization
- [ ] ONNX model support
- [ ] Interactive layer editing
- [ ] VR/AR visualization support
- [ ] Real-time training visualization
- [ ] Custom layer type support

---

**Made with ❤️ for the deep learning community** 
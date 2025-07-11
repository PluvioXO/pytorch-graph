#!/usr/bin/env python3
"""
Quick Example - Neural Viz 3D

A minimal example showing how the neural network visualization would work.
This creates a mock visualization to demonstrate the concept.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_mock_visualization():
    """Create a simple 3D visualization to demonstrate the concept."""
    
    print("🧠 Neural Viz 3D - Quick Example")
    print("=" * 50)
    print("\n📊 Creating mock neural network visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Mock layer data (simulating what our package would extract)
    layers = [
        {"name": "Input", "pos": (0, 0, 0), "size": (1, 2, 1), "color": "lightblue", "params": 0},
        {"name": "Conv2D-1", "pos": (2, 0, 0), "size": (1.5, 1.5, 1), "color": "blue", "params": 896},
        {"name": "ReLU-1", "pos": (4, 0, 0), "size": (0.8, 0.8, 0.8), "color": "green", "params": 0},
        {"name": "Pool-1", "pos": (6, 0, 0), "size": (1, 1, 1), "color": "gray", "params": 0},
        {"name": "Conv2D-2", "pos": (8, 0, 0), "size": (2, 2, 1.2), "color": "blue", "params": 18496},
        {"name": "ReLU-2", "pos": (10, 0, 0), "size": (0.8, 0.8, 0.8), "color": "green", "params": 0},
        {"name": "Pool-2", "pos": (12, 0, 0), "size": (1, 1, 1), "color": "gray", "params": 0},
        {"name": "Flatten", "pos": (14, 0, 0), "size": (0.6, 0.6, 2), "color": "orange", "params": 0},
        {"name": "Dense-1", "pos": (16, 0, 0), "size": (1.8, 1.8, 1.5), "color": "red", "params": 32896},
        {"name": "ReLU-3", "pos": (18, 0, 0), "size": (0.8, 0.8, 0.8), "color": "green", "params": 0},
        {"name": "Dense-2", "pos": (20, 0, 0), "size": (1.2, 1.2, 1), "color": "red", "params": 650},
        {"name": "Softmax", "pos": (22, 0, 0), "size": (0.8, 0.8, 0.8), "color": "purple", "params": 0},
    ]
    
    # Draw layers as 3D boxes
    for layer in layers:
        x, y, z = layer["pos"]
        w, h, d = layer["size"]
        color = layer["color"]
        
        # Create a simple 3D box representation
        # Bottom face
        xx, yy = np.meshgrid([x-w/2, x+w/2], [y-h/2, y+h/2])
        zz = np.full_like(xx, z-d/2)
        ax.plot_surface(xx, yy, zz, alpha=0.7, color=color)
        
        # Top face
        zz = np.full_like(xx, z+d/2)
        ax.plot_surface(xx, yy, zz, alpha=0.7, color=color)
        
        # Add text label
        ax.text(x, y, z+d/2+0.3, layer["name"], fontsize=8, ha='center')
    
    # Draw connections between layers
    for i in range(len(layers)-1):
        x1, y1, z1 = layers[i]["pos"]
        x2, y2, z2 = layers[i+1]["pos"]
        ax.plot([x1+layers[i]["size"][0]/2, x2-layers[i+1]["size"][0]/2], 
                [y1, y2], [z1, z2], 'k-', alpha=0.6, linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('Layer Progression →')
    ax.set_ylabel('Width')
    ax.set_zlabel('Height')
    ax.set_title('🔥 Neural Network Architecture Visualization\nCNN for Image Classification', 
                 fontsize=14, pad=20)
    
    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    # Color legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Convolution'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Activation'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Pooling'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Dense'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Utility'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=10, label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Show statistics
    total_params = sum(layer["params"] for layer in layers)
    print(f"\n📋 Model Statistics:")
    print(f"   • Total Layers: {len(layers)}")
    print(f"   • Total Parameters: {total_params:,}")
    print(f"   • Trainable Layers: {sum(1 for layer in layers if layer['params'] > 0)}")
    
    print(f"\n🧩 Layer Breakdown:")
    layer_types = {}
    for layer in layers:
        layer_type = layer["name"].split('-')[0]
        if layer_type not in layer_types:
            layer_types[layer_type] = 0
        layer_types[layer_type] += 1
    
    for layer_type, count in layer_types.items():
        print(f"   • {layer_type}: {count}")
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 This demonstrates what the neural-viz-3d package creates!")
    print("   • Interactive 3D visualization with Plotly")
    print("   • Hover information for each layer")
    print("   • Automatic layout optimization")
    print("   • Support for PyTorch and Keras models")
    print("   • Export to HTML, PNG, SVG, PDF")

def show_usage_example():
    """Show how the actual package would be used."""
    
    print("\n" + "="*50)
    print("📚 How to use Neural Viz 3D:")
    print("="*50)
    
    code_example = '''
# 🔥 PyTorch Example
import torch
import torch.nn as nn
import neural_viz

class MyCNN(nn.Module):
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
        return self.fc(x)

model = MyCNN()

# ✨ Visualize in 3D!
fig = neural_viz.visualize_pytorch(
    model, 
    input_shape=(3, 32, 32),
    title="My CNN Architecture",
    export_path="model.html"
)
fig.show()

# 🚀 Keras Example  
import tensorflow as tf
import neural_viz

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

# ✨ Visualize in 3D!
fig = neural_viz.visualize_keras(model)
fig.show()
'''
    
    print(code_example)
    
    print("\n🎯 Key Features:")
    print("• 🎨 Interactive 3D visualization with Plotly")
    print("• 🔄 Support for PyTorch and Keras/TensorFlow")
    print("• 📊 Automatic parameter counting and analysis") 
    print("• 🎭 Multiple layout algorithms (hierarchical, spring, circular)")
    print("• 🌈 Smart color coding by layer type")
    print("• 📱 Responsive design for all devices")
    print("• 💾 Export to HTML, PNG, SVG, PDF")
    print("• 🔀 Model comparison capabilities")

def main():
    """Run the quick example."""
    # Show the mock visualization
    create_mock_visualization()
    
    # Show usage examples
    show_usage_example()
    
    print("\n🚀 Ready to visualize your neural networks in 3D!")
    print("   Install: pip install neural-viz-3d")
    print("   GitHub: https://github.com/yourusername/neural-viz-3d")

if __name__ == "__main__":
    main() 
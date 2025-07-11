#!/usr/bin/env python3
"""
Demo script for Neural Viz 3D package.

This script demonstrates how to visualize neural networks in 3D
using both PyTorch and Keras models.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("🧠 Neural Viz 3D - Demo Script")
print("=" * 50)

# Try PyTorch example
def pytorch_demo():
    """Demonstrate PyTorch model visualization."""
    try:
        import torch
        import torch.nn as nn
        print("\n✅ PyTorch is available!")
        
        # Import our visualization package
        import neural_viz
        
        # Define a simple CNN model
        class DemoCNN(nn.Module):
            def __init__(self, num_classes=10):
                super(DemoCNN, self).__init__()
                
                # Feature extraction layers
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.relu1 = nn.ReLU(inplace=True)
                self.bn1 = nn.BatchNorm2d(32)
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.relu2 = nn.ReLU(inplace=True)
                self.bn2 = nn.BatchNorm2d(64)
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.relu3 = nn.ReLU(inplace=True)
                self.bn3 = nn.BatchNorm2d(128)
                self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
                
                # Classification layers
                self.flatten = nn.Flatten()
                self.dropout1 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(128 * 4 * 4, 256)
                self.relu4 = nn.ReLU(inplace=True)
                self.dropout2 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(256, num_classes)
            
            def forward(self, x):
                # Feature extraction
                x = self.pool1(self.bn1(self.relu1(self.conv1(x))))
                x = self.pool2(self.bn2(self.relu2(self.conv2(x))))
                x = self.adaptive_pool(self.bn3(self.relu3(self.conv3(x))))
                
                # Classification
                x = self.flatten(x)
                x = self.dropout1(x)
                x = self.relu4(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)
                return x
        
        # Create the model
        model = DemoCNN(num_classes=10)
        print(f"📊 Created PyTorch CNN with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Visualize the model
        print("🎨 Creating 3D visualization...")
        
        visualizer = neural_viz.NeuralNetworkVisualizer(
            layout_style='hierarchical',
            theme='plotly_dark',
            spacing=2.5,
            width=1400,
            height=900
        )
        
        fig = visualizer.visualize_pytorch(
            model,
            input_shape=(3, 32, 32),  # RGB image 32x32
            title="🔥 PyTorch CNN Architecture - CIFAR-10 Classifier",
            show_connections=True,
            show_labels=True,
            show_parameters=True,
            optimize_layout=True,
            export_path="pytorch_demo.html"
        )
        
        print("✨ PyTorch visualization created!")
        print("📁 Saved as 'pytorch_demo.html'")
        
        # Get model summary
        summary = visualizer.get_model_summary(
            model, 
            framework='pytorch', 
            input_shape=(3, 32, 32)
        )
        
        print("\n📋 Model Summary:")
        print(f"   • Total Layers: {summary['total_layers']}")
        print(f"   • Total Parameters: {summary['total_parameters']:,}")
        print(f"   • Trainable Parameters: {summary['trainable_parameters']:,}")
        print(f"   • Input Shape: {summary['input_shape']}")
        print(f"   • Output Shape: {summary['output_shape']}")
        
        # Show layer types
        print("\n🧩 Layer Types:")
        for layer_type, count in summary['layer_types'].items():
            print(f"   • {layer_type}: {count}")
        
        # Show the visualization
        fig.show()
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available. Skipping PyTorch demo.")
        print("   Install with: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"❌ Error in PyTorch demo: {e}")
        return False


def keras_demo():
    """Demonstrate Keras model visualization."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        print("\n✅ TensorFlow/Keras is available!")
        
        # Import our visualization package
        import neural_viz
        
        # Define a text classification model with LSTM
        model = keras.Sequential([
            layers.Embedding(
                input_dim=10000,     # Vocabulary size
                output_dim=128,      # Embedding dimension
                input_length=100,    # Sequence length
                name='word_embedding'
            ),
            layers.SpatialDropout1D(0.2, name='spatial_dropout'),
            
            layers.LSTM(
                64, 
                return_sequences=True, 
                dropout=0.3, 
                recurrent_dropout=0.3,
                name='lstm_1'
            ),
            layers.LSTM(
                32, 
                dropout=0.3, 
                recurrent_dropout=0.3,
                name='lstm_2'
            ),
            
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.BatchNormalization(name='batch_norm'),
            layers.Dropout(0.5, name='dropout_1'),
            
            layers.Dense(32, activation='relu', name='dense_2'),
            layers.Dropout(0.3, name='dropout_2'),
            
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"📊 Created Keras LSTM with {model.count_params():,} parameters")
        
        # Visualize the model
        print("🎨 Creating 3D visualization...")
        
        visualizer = neural_viz.NeuralNetworkVisualizer(
            layout_style='hierarchical',
            theme='plotly_white',
            spacing=2.0,
            width=1400,
            height=900
        )
        
        fig = visualizer.visualize_keras(
            model,
            title="🚀 Keras LSTM Architecture - Text Classification",
            show_connections=True,
            show_labels=True,
            show_parameters=True,
            export_path="keras_demo.html"
        )
        
        print("✨ Keras visualization created!")
        print("📁 Saved as 'keras_demo.html'")
        
        # Get model summary
        summary = visualizer.get_model_summary(model, framework='keras')
        
        print("\n📋 Model Summary:")
        print(f"   • Total Layers: {summary['total_layers']}")
        print(f"   • Total Parameters: {summary['total_parameters']:,}")
        print(f"   • Trainable Parameters: {summary['trainable_parameters']:,}")
        print(f"   • Input Shape: {summary['input_shape']}")
        print(f"   • Output Shape: {summary['output_shape']}")
        
        # Show layer types
        print("\n🧩 Layer Types:")
        for layer_type, count in summary['layer_types'].items():
            print(f"   • {layer_type}: {count}")
        
        # Show the visualization
        fig.show()
        
        return True
        
    except ImportError:
        print("❌ TensorFlow/Keras not available. Skipping Keras demo.")
        print("   Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error in Keras demo: {e}")
        return False


def comparison_demo():
    """Demonstrate model comparison."""
    try:
        import torch
        import torch.nn as nn
        print("\n🔄 Creating Model Comparison...")
        
        import neural_viz
        
        # Create different architectures
        models = []
        names = []
        
        # Simple MLP
        mlp = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        models.append(mlp)
        names.append("MLP")
        
        # Simple CNN
        cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        models.append(cnn)
        names.append("CNN")
        
        # Create comparison visualization
        visualizer = neural_viz.NeuralNetworkVisualizer(
            layout_style='custom',
            theme='plotly_dark',
            spacing=3.0
        )
        
        fig = visualizer.compare_models(
            models=models,
            names=names,
            frameworks=['pytorch', 'pytorch'],
            input_shapes=[(784,), (1, 28, 28)],
            title="🥊 Architecture Comparison: MLP vs CNN",
            show_connections=True,
            show_labels=True
        )
        
        print("✨ Model comparison created!")
        fig.show()
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available for comparison demo.")
        return False
    except Exception as e:
        print(f"❌ Error in comparison demo: {e}")
        return False


def main():
    """Run the complete demo."""
    print("\n🎯 Starting Neural Network Visualization Demos...")
    
    success_count = 0
    
    # Run PyTorch demo
    if pytorch_demo():
        success_count += 1
    
    # Run Keras demo  
    if keras_demo():
        success_count += 1
    
    # Run comparison demo
    if comparison_demo():
        success_count += 1
    
    print(f"\n🎉 Demo completed! {success_count} visualizations created.")
    
    if success_count > 0:
        print("\n📂 Generated Files:")
        if success_count >= 1:
            print("   • pytorch_demo.html - Interactive PyTorch CNN visualization")
        if success_count >= 2:
            print("   • keras_demo.html - Interactive Keras LSTM visualization")
        
        print("\n💡 Tips:")
        print("   • Open the HTML files in your browser for interactive 3D visualization")
        print("   • Use mouse to rotate, zoom, and pan the 3D models")
        print("   • Hover over layers to see detailed information")
        print("   • Different colors represent different layer types")
        print("   • Layer size represents parameter count and complexity")
    
    print("\n🚀 Try modifying the models and re-running to see different architectures!")


if __name__ == "__main__":
    main() 
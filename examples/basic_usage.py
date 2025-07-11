"""
Basic usage examples for neural-viz-3d package.

This script demonstrates how to visualize neural networks from PyTorch and Keras.
"""

import numpy as np
import neural_viz

# PyTorch example
def pytorch_example():
    """Example of visualizing a PyTorch model."""
    try:
        import torch
        import torch.nn as nn
        
        # Define a simple CNN model
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(32),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(128),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        # Create model
        model = SimpleCNN(num_classes=10)
        
        # Visualize the model
        print("Creating PyTorch CNN visualization...")
        fig = neural_viz.visualize_pytorch(
            model, 
            input_shape=(3, 32, 32),  # RGB image 32x32
            title="PyTorch CNN Architecture",
            show_parameters=True,
            export_path="pytorch_cnn.html"
        )
        
        # Show the visualization
        fig.show()
        
        # Get model summary
        summary = neural_viz.NeuralNetworkVisualizer().get_model_summary(
            model, framework='pytorch', input_shape=(3, 32, 32)
        )
        print("\nPyTorch Model Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
    except ImportError:
        print("PyTorch not available. Skipping PyTorch example.")


def keras_example():
    """Example of visualizing a Keras model."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Define a simple RNN model
        model = keras.Sequential([
            layers.Embedding(input_dim=10000, output_dim=128, input_length=100),
            layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Visualize the model
        print("\nCreating Keras RNN visualization...")
        fig = neural_viz.visualize_keras(
            model,
            title="Keras LSTM Text Classification Model",
            layout_style='hierarchical',
            show_connections=True,
            show_labels=True,
            export_path="keras_lstm.html"
        )
        
        # Show the visualization
        fig.show()
        
        # Get model summary
        summary = neural_viz.NeuralNetworkVisualizer().get_model_summary(
            model, framework='keras'
        )
        print("\nKeras Model Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
    except ImportError:
        print("TensorFlow/Keras not available. Skipping Keras example.")


def advanced_example():
    """Advanced usage examples."""
    try:
        import torch
        import torch.nn as nn
        
        # Create a more complex model
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                residual = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(residual)
                out = self.relu(out)
                return out
        
        class SimpleResNet(nn.Module):
            def __init__(self, num_classes=1000):
                super(SimpleResNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(3, 2, 1)
                
                self.layer1 = ResidualBlock(64, 64)
                self.layer2 = ResidualBlock(64, 128, 2)
                self.layer3 = ResidualBlock(128, 256, 2)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(256, num_classes)
            
            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        model = SimpleResNet(num_classes=10)
        
        # Create visualizer with custom settings
        visualizer = neural_viz.NeuralNetworkVisualizer(
            layout_style='spring',
            theme='plotly_white',
            spacing=1.5,
            width=1400,
            height=900
        )
        
        print("\nCreating advanced ResNet visualization...")
        fig = visualizer.visualize_pytorch(
            model,
            input_shape=(3, 224, 224),
            title="ResNet Architecture with Spring Layout",
            show_connections=True,
            show_parameters=True,
            optimize_layout=True
        )
        
        fig.show()
        
        # Export comprehensive report
        print("Generating comprehensive model report...")
        visualizer.export_summary_report(
            model,
            framework='pytorch',
            input_shape=(3, 224, 224),
            output_path="resnet_report.html"
        )
        
    except ImportError:
        print("PyTorch not available. Skipping advanced example.")


def comparison_example():
    """Example of comparing multiple models."""
    try:
        import torch
        import torch.nn as nn
        
        # Create different model architectures
        models = []
        names = []
        
        # Simple MLP
        mlp = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        models.append(mlp)
        names.append("MLP")
        
        # Simple CNN
        cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        models.append(cnn)
        names.append("CNN")
        
        # Compare models
        visualizer = neural_viz.NeuralNetworkVisualizer(layout_style='custom')
        
        print("\nCreating model comparison visualization...")
        fig = visualizer.compare_models(
            models=models,
            names=names,
            frameworks=['pytorch', 'pytorch'],
            input_shapes=[(784,), (1, 28, 28)],
            title="MLP vs CNN Architecture Comparison"
        )
        
        fig.show()
        
    except ImportError:
        print("PyTorch not available. Skipping comparison example.")


def main():
    """Run all examples."""
    print("Neural Viz 3D - Basic Usage Examples")
    print("=" * 50)
    
    # Run examples
    pytorch_example()
    keras_example()
    advanced_example()
    comparison_example()
    
    print("\nAll examples completed! Check the generated HTML files for interactive visualizations.")


if __name__ == "__main__":
    main() 
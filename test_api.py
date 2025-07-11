#!/usr/bin/env python3
"""
Simple test of torch_vis API to create a neural network diagram.
"""

import torch
import torch.nn as nn
import torch_vis

def main():
    """Test the torch_vis API with a simple neural network."""
    
    # Create a basic feedforward neural network with Sigmoid activation
    model = nn.Sequential(
        nn.Linear(784, 128),  # Input layer (28x28 flattened)
        nn.Sigmoid(),
        nn.Linear(128, 64),   # Hidden layer
        nn.Sigmoid(), 
        nn.Linear(64, 10)     # Output layer (10 classes)
    )
    
    print("Created simple neural network with Sigmoid activation:")
    print(model)
    print()
    
    # Define input shape (batch_size=1, features=784)
    input_shape = (1, 784)
    
    # Generate enhanced flowchart diagram (now the default!)
    print("Generating enhanced architecture diagram...")
    diagram_path = torch_vis.generate_architecture_diagram(
        model=model,
        input_shape=input_shape,
        output_path="neural_network_architecture.png",
        title="Neural Network Architecture"
    )
    
    print(f"Enhanced diagram saved to: {diagram_path}")
    
    # Analyze the model
    print("\nAnalyzing model...")
    analysis = torch_vis.analyze_model(model, input_shape=input_shape)
    
    if analysis.get('total_params'):
        print(f"Total parameters: {analysis['total_params']:,}")
    else:
        print("Model analysis completed")
    
    print("\n Test completed successfully!")
    print(f"Generated: {diagram_path}")
    print("The enhanced flowchart style is now the default!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Clean PyTorch Graph Demo - Uses Enhanced Library Features

This demo showcases the enhanced pytorch-graph library features:
1. Architecture diagram generation
2. Enhanced computational graph visualization with proper legend positioning
3. Model analysis and comparison
4. All visualization improvements are handled by the library
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Import the local pytorch-graph package
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    import pytorch_graph
    print("Successfully imported local pytorch-graph package")
except Exception as e:
    print(f"Failed to import pytorch-graph: {e}")
    sys.exit(1)


def create_sample_models():
    """Create various sample PyTorch models for demonstration."""
    
    # Simple MLP
    mlp_model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 10)
    )
    
    # CNN model
    cnn_model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)
    )
    
    # Custom model with skip connections
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return self.relu(out)
    
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            
            self.res_block1 = ResidualBlock(64, 64)
            self.res_block2 = ResidualBlock(64, 64)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 1000)
            
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.res_block1(x)
            x = self.res_block2(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    custom_model = CustomModel()
    
    return {
        'MLP': mlp_model,
        'CNN': cnn_model,
        'Custom_ResNet': custom_model
    }


def demo_architecture_diagrams():
    """Demonstrate architecture diagram generation using the library."""
    print("\n" + "="*60)
    print("ARCHITECTURE DIAGRAMS")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nGenerating {name} architecture diagram...")
        
        # Use correct input shapes
        if 'MLP' in name:
            input_shape = (1, 784)  # MNIST-like
        elif 'CNN' in name:
            input_shape = (1, 3, 32, 32)  # CIFAR-like
        elif 'Custom' in name:
            input_shape = (1, 3, 224, 224)  # ImageNet-like
        else:
            input_shape = (1, 3, 32, 32)
        
        try:
            # Generate flowchart diagram
            output_path = f"demo_outputs/{name.lower()}_flowchart.png"
            pytorch_graph.generate_flowchart_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"{name} Architecture"
            )
            print(f"   ✅ Flowchart diagram saved: {output_path}")
            
            # Generate research paper style
            output_path = f"demo_outputs/{name.lower()}_research.png"
            pytorch_graph.generate_research_paper_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"{name} Architecture (Research Paper Style)"
            )
            print(f"   ✅ Research diagram saved: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error generating {name} diagram: {e}")


def demo_enhanced_computational_graphs():
    """Demonstrate enhanced computational graph generation using the library."""
    print("\n" + "="*60)
    print("ENHANCED COMPUTATIONAL GRAPHS")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nGenerating enhanced computational graph for {name}...")
        
        # Create appropriate input tensor
        if 'MLP' in name:
            input_tensor = torch.randn(1, 784, requires_grad=True)
        elif 'CNN' in name:
            input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
        elif 'Custom' in name:
            input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        else:
            input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
        
        try:
            # Use the library's enhanced computational graph visualization
            output_path = f"demo_outputs/{name.lower()}_computational_graph.png"
            pytorch_graph.save_computational_graph_png(
                model=model,
                input_tensor=input_tensor,
                filepath=output_path,
                width=1600,
                height=1000,
                dpi=300,
                show_legend=True,
                node_size=25,
                font_size=10
            )
            print(f"   ✅ Enhanced computational graph saved: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error generating {name} computational graph: {e}")


def demo_model_analysis():
    """Demonstrate model analysis using the library."""
    print("\n" + "="*60)
    print("MODEL ANALYSIS")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        
        try:
            # Use the library's model analysis
            analysis = pytorch_graph.analyze_model(model)
            print(f"   ✅ Model analysis completed for {name}")
            print(f"      Total parameters: {analysis.get('total_parameters', 'N/A')}")
            print(f"      Trainable parameters: {analysis.get('trainable_parameters', 'N/A')}")
            print(f"      Model size: {analysis.get('model_size_mb', 'N/A')} MB")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {name}: {e}")


def main():
    """Run the clean demo using enhanced library features."""
    print("PyTorch Graph Package Demo (CLEAN VERSION)")
    print("=" * 60)
    print("This demo showcases the enhanced pytorch-graph library features:")
    print("• Architecture diagram generation")
    print("• Enhanced computational graph visualization with proper legend positioning")
    print("• Model analysis and comparison")
    print("• All visualization improvements are handled by the library")
    
    # Create output directory
    os.makedirs("demo_outputs", exist_ok=True)
    
    try:
        # Run demos using library features
        demo_architecture_diagrams()
        demo_enhanced_computational_graphs()
        demo_model_analysis()
        
        print("\n" + "="*60)
        print("🎉 CLEAN DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the 'demo_outputs' directory for all generated files:")
        
        # List generated files
        if os.path.exists("demo_outputs"):
            files = os.listdir("demo_outputs")
            for file in sorted(files):
                print(f"   📄 {file}")
        
        print("\nEnhanced Library Features Demonstrated:")
        print("   • Architecture diagrams with multiple styles")
        print("   • Enhanced computational graphs with proper legend positioning")
        print("   • Model analysis and parameter counting")
        print("   • High-quality PNG output with 300 DPI")
        print("   • No overlapping legends or text issues")
        print("   • Professional, research-paper ready visualizations")
        
        print("\nKey Benefits:")
        print("   • Clean, simple demo code")
        print("   • All complex visualization logic in the library")
        print("   • Automatic legend positioning (no overlap)")
        print("   • Enhanced color schemes and styling")
        print("   • Easy to use and maintain")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

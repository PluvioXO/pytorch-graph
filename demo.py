#!/usr/bin/env python3
"""
PyTorch Graph Demo - Comprehensive demonstration of pytorch-graph package features.

This demo showcases:
1. Basic architecture diagram generation
2. Different diagram styles (flowchart, standard, research paper)
3. Computational graph tracking and visualization
4. Model analysis and comparison
5. 3D model visualization
6. Export capabilities

Run this demo to see all the visualization capabilities of the pytorch-graph package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Import the pytorch-graph package
try:
    import torch_vis
    print("✅ Successfully imported torch_vis package")
except ImportError as e:
    print(f"❌ Failed to import torch_vis: {e}")
    print("Please install the package first: pip install -e .")
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


def demo_basic_architecture_diagrams():
    """Demonstrate basic architecture diagram generation."""
    print("\n" + "="*60)
    print("🏗️  BASIC ARCHITECTURE DIAGRAMS")
    print("="*60)
    
    models = create_sample_models()
    
    # Create output directory
    os.makedirs("demo_outputs", exist_ok=True)
    
    # Generate diagrams for each model
    for name, model in models.items():
        print(f"\n📊 Generating {name} architecture diagram...")
        
        # Determine input shape based on model type
        if 'MLP' in name:
            input_shape = (1, 784)  # MNIST-like
        else:
            input_shape = (1, 3, 224, 224)  # ImageNet-like
        
        try:
            # Generate flowchart diagram (default style)
            output_path = f"demo_outputs/{name.lower()}_flowchart.png"
            torch_vis.generate_flowchart_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"{name} Architecture (Flowchart Style)"
            )
            print(f"   ✅ Flowchart diagram saved: {output_path}")
            
            # Generate research paper style
            output_path = f"demo_outputs/{name.lower()}_research.png"
            torch_vis.generate_research_paper_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"{name} Architecture (Research Paper Style)"
            )
            print(f"   ✅ Research paper diagram saved: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error generating {name} diagram: {e}")


def demo_diagram_styles():
    """Demonstrate different diagram styles."""
    print("\n" + "="*60)
    print("🎨 DIAGRAM STYLES DEMONSTRATION")
    print("="*60)
    
    models = create_sample_models()
    model = models['CNN']  # Use CNN for style demo
    input_shape = (1, 3, 224, 224)
    
    styles = ['flowchart', 'standard', 'research_paper']
    
    for style in styles:
        print(f"\n🎨 Generating {style} style diagram...")
        try:
            output_path = f"demo_outputs/cnn_{style}_style.png"
            torch_vis.generate_architecture_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"CNN Architecture ({style.title()} Style)",
                style=style
            )
            print(f"   ✅ {style.title()} style diagram saved: {output_path}")
        except Exception as e:
            print(f"   ❌ Error generating {style} style: {e}")


def demo_model_analysis():
    """Demonstrate model analysis capabilities."""
    print("\n" + "="*60)
    print("📈 MODEL ANALYSIS")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\n🔍 Analyzing {name} model...")
        
        # Determine input shape
        if 'MLP' in name:
            input_shape = (1, 784)
        else:
            input_shape = (1, 3, 224, 224)
        
        try:
            # Analyze model
            analysis = torch_vis.analyze_model(model, input_shape=input_shape, detailed=True)
            
            print(f"   📊 Model Statistics:")
            print(f"      • Total Parameters: {analysis.get('total_params', 'N/A'):,}")
            print(f"      • Trainable Parameters: {analysis.get('trainable_params', 'N/A'):,}")
            print(f"      • Model Size: {analysis.get('model_size', 'N/A')}")
            print(f"      • Memory Usage: {analysis.get('memory_usage', 'N/A')}")
            print(f"      • Layer Count: {analysis.get('layer_count', 'N/A')}")
            
            # Show layer breakdown
            if 'layer_breakdown' in analysis:
                print(f"   🏗️  Layer Breakdown:")
                for layer_type, count in analysis['layer_breakdown'].items():
                    print(f"      • {layer_type}: {count}")
                    
        except Exception as e:
            print(f"   ❌ Error analyzing {name}: {e}")


def demo_computational_graph_tracking():
    """Demonstrate computational graph tracking and visualization."""
    print("\n" + "="*60)
    print("🕸️  COMPUTATIONAL GRAPH TRACKING")
    print("="*60)
    
    # Create a simple model for computational graph demo
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Sigmoid(),
        nn.Linear(10, 1)
    )
    
    # Create input tensor
    input_tensor = torch.randn(1, 10, requires_grad=True)
    
    print("\n🔍 Tracking computational graph execution...")
    try:
        # Track computational graph
        tracker = torch_vis.track_computational_graph_execution(
            model=model,
            input_tensor=input_tensor,
            track_memory=True,
            track_timing=True,
            track_tensor_ops=True
        )
        
        # Get summary
        summary = tracker.get_graph_summary()
        print(f"   📊 Graph Summary:")
        print(f"      • Total Nodes: {summary.get('total_nodes', 'N/A')}")
        print(f"      • Total Edges: {summary.get('total_edges', 'N/A')}")
        print(f"      • Execution Time: {summary.get('execution_time', 'N/A'):.4f}s")
        print(f"      • Memory Usage: {summary.get('memory_usage', 'N/A')}")
        
        # Visualize computational graph
        print(f"   🎨 Generating computational graph visualization...")
        try:
            fig = torch_vis.visualize_computational_graph(model, input_tensor, renderer='plotly')
            # Save as HTML for demo purposes
            fig.write_html("demo_outputs/computational_graph.html")
            print(f"   ✅ Computational graph saved: demo_outputs/computational_graph.html")
        except Exception as e:
            print(f"   ⚠️  Could not generate interactive visualization: {e}")
        
        # Export graph data
        print(f"   💾 Exporting graph data...")
        try:
            torch_vis.export_computational_graph(model, input_tensor, "demo_outputs/graph_data.json")
            print(f"   ✅ Graph data exported: demo_outputs/graph_data.json")
        except Exception as e:
            print(f"   ⚠️  Could not export graph data: {e}")
            
    except Exception as e:
        print(f"   ❌ Error in computational graph tracking: {e}")


def demo_3d_visualization():
    """Demonstrate 3D model visualization."""
    print("\n" + "="*60)
    print("🌐 3D MODEL VISUALIZATION")
    print("="*60)
    
    models = create_sample_models()
    model = models['CNN']  # Use CNN for 3D demo
    input_shape = (1, 3, 224, 224)
    
    print("\n🎨 Generating 3D model visualization...")
    try:
        # Try Plotly renderer
        try:
            fig = torch_vis.visualize(model, input_shape=input_shape, renderer='plotly')
            fig.write_html("demo_outputs/model_3d_plotly.html")
            print(f"   ✅ 3D Plotly visualization saved: demo_outputs/model_3d_plotly.html")
        except Exception as e:
            print(f"   ⚠️  Plotly 3D visualization not available: {e}")
        
        # Try Matplotlib renderer
        try:
            fig = torch_vis.visualize(model, input_shape=input_shape, renderer='matplotlib')
            fig.savefig("demo_outputs/model_3d_matplotlib.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ 3D Matplotlib visualization saved: demo_outputs/model_3d_matplotlib.png")
        except Exception as e:
            print(f"   ⚠️  Matplotlib 3D visualization not available: {e}")
            
    except Exception as e:
        print(f"   ❌ Error in 3D visualization: {e}")


def demo_model_comparison():
    """Demonstrate model comparison capabilities."""
    print("\n" + "="*60)
    print("⚖️  MODEL COMPARISON")
    print("="*60)
    
    models = create_sample_models()
    
    # Compare MLP and CNN
    print("\n🔄 Comparing MLP vs CNN models...")
    try:
        comparison_fig = torch_vis.compare_models(
            models=[models['MLP'], models['CNN']],
            names=['MLP', 'CNN'],
            input_shapes=[(1, 784), (1, 3, 224, 224)],
            renderer='plotly'
        )
        comparison_fig.write_html("demo_outputs/model_comparison.html")
        print(f"   ✅ Model comparison saved: demo_outputs/model_comparison.html")
    except Exception as e:
        print(f"   ❌ Error in model comparison: {e}")


def demo_architecture_report():
    """Demonstrate comprehensive architecture report generation."""
    print("\n" + "="*60)
    print("📋 ARCHITECTURE REPORT GENERATION")
    print("="*60)
    
    models = create_sample_models()
    model = models['Custom_ResNet']
    input_shape = (1, 3, 224, 224)
    
    print("\n📄 Generating comprehensive architecture report...")
    try:
        torch_vis.create_architecture_report(
            model=model,
            input_shape=input_shape,
            output_path="demo_outputs/architecture_report.html"
        )
        print(f"   ✅ Architecture report saved: demo_outputs/architecture_report.html")
    except Exception as e:
        print(f"   ❌ Error generating architecture report: {e}")


def demo_advanced_features():
    """Demonstrate advanced features like profiling and activation extraction."""
    print("\n" + "="*60)
    print("🚀 ADVANCED FEATURES")
    print("="*60)
    
    models = create_sample_models()
    model = models['CNN']
    input_shape = (1, 3, 224, 224)
    
    # Model profiling
    print("\n⏱️  Model profiling...")
    try:
        profile_results = torch_vis.profile_model(model, input_shape, device='cpu')
        print(f"   📊 Profiling Results:")
        print(f"      • Forward Pass Time: {profile_results.get('forward_time', 'N/A'):.4f}s")
        print(f"      • Memory Usage: {profile_results.get('memory_usage', 'N/A')}")
        print(f"      • Parameter Count: {profile_results.get('param_count', 'N/A'):,}")
    except Exception as e:
        print(f"   ❌ Error in model profiling: {e}")
    
    # Activation extraction
    print("\n🧠 Extracting intermediate activations...")
    try:
        input_tensor = torch.randn(input_shape)
        activations = torch_vis.extract_activations(model, input_tensor)
        print(f"   📊 Extracted activations from {len(activations)} layers:")
        for layer_name, activation in activations.items():
            if hasattr(activation, 'shape'):
                print(f"      • {layer_name}: {activation.shape}")
    except Exception as e:
        print(f"   ❌ Error in activation extraction: {e}")


def main():
    """Run the complete demo."""
    print("🎯 PyTorch Graph Package Demo")
    print("=" * 60)
    print("This demo showcases all the visualization capabilities of the pytorch-graph package.")
    print("Output files will be saved to the 'demo_outputs' directory.")
    
    # Create output directory
    os.makedirs("demo_outputs", exist_ok=True)
    
    # Run all demos
    try:
        demo_basic_architecture_diagrams()
        demo_diagram_styles()
        demo_model_analysis()
        demo_computational_graph_tracking()
        demo_3d_visualization()
        demo_model_comparison()
        demo_architecture_report()
        demo_advanced_features()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the 'demo_outputs' directory for all generated files:")
        
        # List generated files
        if os.path.exists("demo_outputs"):
            files = os.listdir("demo_outputs")
            for file in sorted(files):
                print(f"   📄 {file}")
        
        print("\n💡 Tips:")
        print("   • Open .html files in your browser for interactive visualizations")
        print("   • .png files are high-quality static diagrams")
        print("   • .json files contain raw graph data for further analysis")
        print("   • Try different models and input shapes for your own experiments")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

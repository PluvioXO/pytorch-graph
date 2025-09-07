#!/usr/bin/env python3
"""
PyTorch Graph - New Features Demo
=================================

This demo showcases the enhanced functionality of PyTorch Graph:
• Complete computational graph traversal (no artificial limits)
• Full method/object names (no truncation)
• Smart arrow positioning (no crossing over boxes)
• Compact layout (no gaps or breaks)
• Professional visualization quality

Run this script to see all the new features in action!
"""

import torch
import torch.nn as nn
import sys
import os

# Add the local package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pytorch_graph import (
        generate_architecture_diagram,
        track_computational_graph,
        ComputationalGraphTracker,
        analyze_computational_graph
    )
    print("✅ Successfully imported pytorch-graph package")
except ImportError as e:
    print(f"❌ Failed to import pytorch-graph: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def create_sample_models():
    """Create sample models for demonstration."""
    
    # Simple MLP
    mlp = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    
    # CNN
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 10)
    )
    
    # Custom ResNet-like model
    class CustomResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            
            self.layer1 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64)
            )
            
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128)
            )
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, 1000)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            residual = x
            x = self.layer1(x)
            x += residual
            x = self.relu(x)
            
            x = self.layer2(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    return {
        'MLP': mlp,
        'CNN': cnn,
        'CustomResNet': CustomResNet()
    }

def demo_architecture_visualization():
    """Demonstrate enhanced architecture visualization."""
    print("\n" + "="*60)
    print("🏗️  ARCHITECTURE VISUALIZATION DEMO")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nGenerating {name} architecture diagrams...")
        
        # Create appropriate input shapes
        if name == 'MLP':
            input_shape = (1, 784)
        elif name == 'CNN':
            input_shape = (1, 3, 32, 32)
        else:  # CustomResNet
            input_shape = (1, 3, 224, 224)
        
        try:
            # Generate flowchart diagram
            generate_architecture_diagram(
                model=model,
                input_shape=input_shape,
                output_path=f"demo_outputs/{name.lower()}_architecture_flowchart.png",
                title=f"{name} Architecture (Flowchart Style)",
                style="flowchart"
            )
            print(f"   ✅ Flowchart diagram saved: demo_outputs/{name.lower()}_architecture_flowchart.png")
            
            # Generate research paper diagram
            generate_architecture_diagram(
                model=model,
                input_shape=input_shape,
                output_path=f"demo_outputs/{name.lower()}_architecture_research.png",
                title=f"{name} Architecture (Research Style)",
                style="research_paper"
            )
            print(f"   ✅ Research diagram saved: demo_outputs/{name.lower()}_architecture_research.png")
            
        except Exception as e:
            print(f"   ❌ Error generating {name} diagrams: {e}")

def demo_computational_graph_analysis():
    """Demonstrate complete computational graph analysis."""
    print("\n" + "="*60)
    print("🔍 COMPUTATIONAL GRAPH ANALYSIS DEMO")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nAnalyzing {name} computational graph...")
        
        # Create appropriate input tensors
        if name == 'MLP':
            input_tensor = torch.randn(1, 784, requires_grad=True)
        elif name == 'CNN':
            input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
        else:  # CustomResNet
            input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        try:
            # Track complete computational graph
            tracker = track_computational_graph(
                model=model,
                input_tensor=input_tensor,
                track_memory=True,
                track_timing=True,
                track_tensor_ops=True
            )
            
            # Save high-quality computational graph
            tracker.save_graph_png(
                filepath=f"demo_outputs/{name.lower()}_complete_computational_graph.png",
                width=1600,
                height=1200,
                dpi=300,
                show_legend=True,
                node_size=25,
                font_size=12
            )
            print(f"   ✅ Complete computational graph saved: demo_outputs/{name.lower()}_complete_computational_graph.png")
            
            # Get comprehensive analysis
            analysis = analyze_computational_graph(model, input_tensor, detailed=True)
            
            # Print analysis results
            summary = analysis['summary']
            print(f"   📊 Analysis Results:")
            print(f"      • Total Operations: {summary['total_nodes']:,}")
            print(f"      • Total Edges: {summary['total_edges']:,}")
            print(f"      • Execution Time: {summary['execution_time']:.4f}s")
            
            if 'performance' in analysis:
                perf = analysis['performance']
                print(f"      • Operations/Second: {perf['operations_per_second']:.2f}")
                print(f"      • Memory Usage: {perf['memory_usage']}")
            
            # Show operation type breakdown
            if 'operation_types' in summary:
                print(f"      • Operation Types:")
                for op_type, count in summary['operation_types'].items():
                    print(f"        - {op_type}: {count}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {name}: {e}")

def demo_advanced_features():
    """Demonstrate advanced features."""
    print("\n" + "="*60)
    print("🚀 ADVANCED FEATURES DEMO")
    print("="*60)
    
    # Create a complex model for demonstration
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )
            
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    model = ComplexModel()
    input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
    
    print("\nDemonstrating advanced computational graph features...")
    
    try:
        # Create tracker with custom settings
        tracker = ComputationalGraphTracker(
            model=model,
            track_memory=True,
            track_timing=True,
            track_tensor_ops=True
        )
        
        # Start tracking
        tracker.start_tracking()
        
        # Run model with multiple operations
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Stop tracking
        tracker.stop_tracking()
        
        # Save with custom parameters
        tracker.save_graph_png(
            filepath="demo_outputs/advanced_features_demo.png",
            width=2000,
            height=1500,
            dpi=300,
            show_legend=True,
            node_size=30,
            font_size=14
        )
        print("   ✅ Advanced features demo saved: demo_outputs/advanced_features_demo.png")
        
        # Export graph data
        tracker.export_graph("demo_outputs/graph_data.json")
        print("   ✅ Graph data exported: demo_outputs/graph_data.json")
        
        # Get detailed summary
        summary = tracker.get_graph_summary()
        print(f"   📊 Advanced Analysis:")
        print(f"      • Total Operations: {summary['total_nodes']:,}")
        print(f"      • Execution Time: {summary['execution_time']:.4f}s")
        print(f"      • Memory Usage: {summary['memory_usage']}")
        
    except Exception as e:
        print(f"   ❌ Error in advanced features demo: {e}")

def main():
    """Main demo function."""
    print("PyTorch Graph - New Features Demo")
    print("="*60)
    print("This demo showcases the enhanced functionality:")
    print("• Complete computational graph traversal (no artificial limits)")
    print("• Full method/object names (no truncation)")
    print("• Smart arrow positioning (no crossing over boxes)")
    print("• Compact layout (no gaps or breaks)")
    print("• Professional visualization quality")
    print("="*60)
    
    # Create output directory
    os.makedirs("demo_outputs", exist_ok=True)
    
    # Run demos
    demo_architecture_visualization()
    demo_computational_graph_analysis()
    demo_advanced_features()
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("📁 Check the 'demo_outputs' directory for all generated files:")
    
    # List generated files
    if os.path.exists("demo_outputs"):
        files = [f for f in os.listdir("demo_outputs") if f.endswith('.png') or f.endswith('.json')]
        for file in sorted(files):
            print(f"   📄 {file}")
    
    print("\n💡 New Features Demonstrated:")
    print("   • Architecture diagrams with multiple styles")
    print("   • Complete computational graphs with full operation names")
    print("   • Smart arrow positioning and compact layout")
    print("   • High-quality PNG output with 300 DPI")
    print("   • Comprehensive analysis and performance metrics")
    print("   • Professional, publication-ready visualizations")
    
    print("\n🔍 Key Benefits:")
    print("   • No artificial limits - shows complete computational graph")
    print("   • Full method names - no truncation of operation names")
    print("   • Proper arrow connections - no crossing over boxes")
    print("   • Compact layout - no gaps or breaks in the graph")
    print("   • Professional quality - publication-ready output")

if __name__ == "__main__":
    main()

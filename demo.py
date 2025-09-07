#!/usr/bin/env python3
"""
PyTorch Graph - Complete Functionality Demo

This demo showcases the FULL functionality of the pytorch-graph library:
1. Architecture diagram generation (multiple styles)
2. Complete computational graph tracking and visualization
3. ComputationalGraphTracker class usage
4. Model analysis and performance metrics
5. Advanced features and customization options
6. Professional visualization quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time

# Import the local pytorch-graph package
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from pytorch_graph import (
        generate_architecture_diagram,
        track_computational_graph,
        ComputationalGraphTracker,
        analyze_computational_graph,
        analyze_model
    )
    print("✅ Successfully imported pytorch-graph package with all features")
except Exception as e:
    print(f"❌ Failed to import pytorch-graph: {e}")
    sys.exit(1)


def create_sample_models():
    """Create various sample PyTorch models for demonstration."""
    
    # Simple MLP
    mlp_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
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
    
    class CustomResNet(nn.Module):
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
    
    custom_model = CustomResNet()
    
    return {
        'MLP': mlp_model,
        'CNN': cnn_model,
        'Custom_ResNet': custom_model
    }


def demo_architecture_diagrams():
    """Demonstrate architecture diagram generation with multiple styles."""
    print("\n" + "="*60)
    print("🏗️  ARCHITECTURE DIAGRAMS")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nGenerating {name} architecture diagrams...")
        
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
            # Generate flowchart diagram (default style)
            output_path = f"demo_outputs/{name.lower()}_flowchart.png"
            generate_architecture_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"{name} Architecture (Flowchart Style)",
                style="flowchart"
            )
            print(f"   ✅ Flowchart diagram saved: {output_path}")
            
            # Generate research paper style
            output_path = f"demo_outputs/{name.lower()}_research.png"
            generate_architecture_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"{name} Architecture (Research Paper Style)",
                style="research_paper"
            )
            print(f"   ✅ Research diagram saved: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error generating {name} diagrams: {e}")


def demo_computational_graph_tracker():
    """Demonstrate ComputationalGraphTracker class usage."""
    print("\n" + "="*60)
    print("🔍 COMPUTATIONAL GRAPH TRACKER DEMO")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nTracking {name} with ComputationalGraphTracker...")
        
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
            # Create tracker with custom settings
            tracker = ComputationalGraphTracker(
                model=model,
                track_memory=True,
                track_timing=True,
                track_tensor_ops=True
            )
            
            # Start tracking
            tracker.start_tracking()
            
            # Run model with forward and backward pass
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
            
            # Stop tracking
            tracker.stop_tracking()
            
            # Save high-quality computational graph
            output_path = f"demo_outputs/{name.lower()}_enhanced_computational_graph.png"
            tracker.save_graph_png(
                filepath=output_path,
                width=1600,
                height=1200,
                dpi=300,
                show_legend=True,
                node_size=25,
                font_size=12
            )
            print(f"   ✅ Enhanced computational graph saved: {output_path}")
            
            # Get comprehensive analysis
            summary = tracker.get_graph_summary()
            print(f"   📊 Analysis Results:")
            print(f"      • Total Operations: {summary['total_nodes']:,}")
            print(f"      • Total Edges: {summary['total_edges']:,}")
            print(f"      • Execution Time: {summary['execution_time']:.4f}s")
            
            if summary['memory_usage']:
                print(f"      • Memory Usage: {summary['memory_usage']}")
            
            # Show operation type breakdown
            if 'operation_types' in summary:
                print(f"      • Operation Types:")
                for op_type, count in summary['operation_types'].items():
                    print(f"        - {op_type}: {count}")
            
            # Export graph data
            json_path = f"demo_outputs/{name.lower()}_graph_data.json"
            tracker.export_graph(json_path)
            print(f"   ✅ Graph data exported: {json_path}")
            
        except Exception as e:
            print(f"   ❌ Error tracking {name}: {e}")


def demo_track_computational_graph():
    """Demonstrate the track_computational_graph convenience function."""
    print("\n" + "="*60)
    print("⚡ TRACK COMPUTATIONAL GRAPH DEMO")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nUsing track_computational_graph for {name}...")
        
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
            # Use the convenience function
            tracker = track_computational_graph(
                model=model,
                input_tensor=input_tensor,
                track_memory=True,
                track_timing=True,
                track_tensor_ops=True
            )
            
            # Save computational graph
            output_path = f"demo_outputs/{name.lower()}_tracked_graph.png"
            tracker.save_graph_png(
                filepath=output_path,
                width=1400,
                height=1000,
                dpi=300,
                show_legend=True
            )
            print(f"   ✅ Tracked computational graph saved: {output_path}")
            
            # Get analysis
            analysis = analyze_computational_graph(model, input_tensor, detailed=True)
            summary = analysis['summary']
            print(f"   📊 Quick Analysis:")
            print(f"      • Operations: {summary['total_nodes']:,}")
            print(f"      • Execution Time: {summary['execution_time']:.4f}s")
            
        except Exception as e:
            print(f"   ❌ Error with track_computational_graph for {name}: {e}")


def demo_model_analysis():
    """Demonstrate comprehensive model analysis."""
    print("\n" + "="*60)
    print("📊 MODEL ANALYSIS DEMO")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        
        try:
            # Use the library's model analysis
            analysis = analyze_model(model)
            print(f"   ✅ Model analysis completed for {name}")
            print(f"      Total parameters: {analysis.get('total_parameters', 'N/A'):,}")
            print(f"      Trainable parameters: {analysis.get('trainable_parameters', 'N/A'):,}")
            print(f"      Model size: {analysis.get('model_size_mb', 'N/A'):.2f} MB")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {name}: {e}")


def demo_advanced_features():
    """Demonstrate advanced features and customization."""
    print("\n" + "="*60)
    print("🚀 ADVANCED FEATURES DEMO")
    print("="*60)
    
    # Create a complex model for advanced demonstration
    class AdvancedModel(nn.Module):
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
    
    model = AdvancedModel()
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
        tracker.export_graph("demo_outputs/advanced_graph_data.json")
        print("   ✅ Advanced graph data exported: demo_outputs/advanced_graph_data.json")
        
        # Get detailed summary
        summary = tracker.get_graph_summary()
        print(f"   📊 Advanced Analysis:")
        print(f"      • Total Operations: {summary['total_nodes']:,}")
        print(f"      • Execution Time: {summary['execution_time']:.4f}s")
        print(f"      • Memory Usage: {summary['memory_usage']}")
        
        # Get complete graph data
        graph_data = tracker.get_graph_data()
        print(f"      • Graph Data: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
        
    except Exception as e:
        print(f"   ❌ Error in advanced features demo: {e}")


def demo_performance_comparison():
    """Demonstrate performance comparison between models."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE COMPARISON DEMO")
    print("="*60)
    
    models = create_sample_models()
    results = {}
    
    for name, model in models.items():
        print(f"\nAnalyzing performance of {name}...")
        
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
            # Track computational graph
            tracker = track_computational_graph(model, input_tensor)
            
            # Get analysis
            analysis = analyze_computational_graph(model, input_tensor, detailed=True)
            summary = analysis['summary']
            
            results[name] = {
                'operations': summary['total_nodes'],
                'execution_time': summary['execution_time'],
                'memory_usage': summary['memory_usage']
            }
            
            print(f"   ✅ {name} analysis completed")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {name}: {e}")
    
    # Print comparison
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Operations: {metrics['operations']:,}")
        print(f"  Time: {metrics['execution_time']:.4f}s")
        print(f"  Memory: {metrics['memory_usage']}")
        print()


def main():
    """Run the complete demo showcasing all library features."""
    print("PyTorch Graph - Complete Functionality Demo")
    print("=" * 60)
    print("This demo showcases the FULL functionality of pytorch-graph:")
    print("• Architecture diagram generation (multiple styles)")
    print("• Complete computational graph tracking and visualization")
    print("• ComputationalGraphTracker class usage")
    print("• Model analysis and performance metrics")
    print("• Advanced features and customization options")
    print("• Professional visualization quality")
    
    # Create output directory
    os.makedirs("demo_outputs", exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Run all demos
        demo_architecture_diagrams()
        demo_computational_graph_tracker()
        demo_track_computational_graph()
        demo_model_analysis()
        demo_advanced_features()
        demo_performance_comparison()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*60)
        print("🎉 COMPLETE DEMO FINISHED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print("📁 Check the 'demo_outputs' directory for all generated files:")
        
        # List generated files
        if os.path.exists("demo_outputs"):
            files = [f for f in os.listdir("demo_outputs") if f.endswith(('.png', '.json'))]
            for file in sorted(files):
                print(f"   📄 {file}")
        
        print("\n🚀 Complete Library Features Demonstrated:")
        print("   • Architecture diagrams with multiple styles (flowchart, research)")
        print("   • ComputationalGraphTracker class with full control")
        print("   • track_computational_graph convenience function")
        print("   • Complete computational graph visualization")
        print("   • Model analysis and parameter counting")
        print("   • Performance comparison and metrics")
        print("   • Advanced features and customization")
        print("   • High-quality PNG output with 300 DPI")
        print("   • JSON data export for further analysis")
        print("   • Professional, publication-ready visualizations")
        
        print("\n🔍 Key Benefits:")
        print("   • Complete computational graph traversal (no artificial limits)")
        print("   • Full method names (no truncation)")
        print("   • Smart arrow positioning (no crossing over boxes)")
        print("   • Compact layout (no gaps or breaks)")
        print("   • Professional quality output")
        print("   • Comprehensive analysis and metrics")
        print("   • Easy integration with existing workflows")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
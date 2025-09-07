#!/usr/bin/env python3
"""
PyTorch Graph - Simplified Demo

This demo showcases the key functionality of the pytorch-graph library:
1. Architecture diagram generation (one of each style)
2. Computational graph tracking and visualization
3. Model analysis
"""

import torch
import torch.nn as nn
import os
import sys

# Import the local pytorch-graph package
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from pytorch_graph import (
        generate_architecture_diagram,
        track_computational_graph,
        analyze_model
    )
    print("✅ Successfully imported pytorch-graph package")
except Exception as e:
    print(f"❌ Failed to import pytorch-graph: {e}")
    sys.exit(1)


def create_sample_model():
    """Create a sample PyTorch model for demonstration."""
    
    # Simple CNN model
    model = nn.Sequential(
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
    
    return model


def demo_architecture_diagrams():
    """Generate one architecture diagram of each style."""
    print("\n" + "="*50)
    print("🏗️  ARCHITECTURE DIAGRAMS")
    print("="*50)
    
    model = create_sample_model()
    input_shape = (3, 32, 32)  # Remove batch dimension - parser adds it automatically
    
    # Create output directory
    os.makedirs("demo_outputs", exist_ok=True)
    
    try:
        # Generate flowchart diagram (default style)
        output_path = "demo_outputs/cnn_flowchart.png"
        generate_architecture_diagram(
            model=model,
            input_shape=input_shape,
            output_path=output_path,
            title="CNN Architecture (Flowchart Style)",
            style="flowchart"
        )
        print(f"✅ Flowchart diagram saved: {output_path}")
        
        # Generate research paper style
        output_path = "demo_outputs/cnn_research.png"
        generate_architecture_diagram(
            model=model,
            input_shape=input_shape,
            output_path=output_path,
            title="CNN Architecture (Research Paper Style)",
            style="research_paper"
        )
        print(f"✅ Research diagram saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating diagrams: {e}")


def demo_computational_graph():
    """Generate one computational graph visualization."""
    print("\n" + "="*50)
    print("🔍 COMPUTATIONAL GRAPH")
    print("="*50)
    
    model = create_sample_model()
    input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
    
    try:
        # Track computational graph
        tracker = track_computational_graph(
            model=model,
            input_tensor=input_tensor,
            track_memory=True,
            track_timing=True,
            track_tensor_ops=True
        )
        
        # Save computational graph
        output_path = "demo_outputs/cnn_computational_graph.png"
        tracker.save_graph_png(
            filepath=output_path,
            width=1600,
            height=1200,
            dpi=300,
            show_legend=True,
            node_size=25,
            font_size=12
        )
        print(f"✅ Computational graph saved: {output_path}")
        
        # Get analysis with error handling
        try:
            summary = tracker.get_graph_summary()
            print(f"📊 Analysis Results:")
            print(f"   • Total Operations: {summary.get('total_nodes', 0):,}")
            print(f"   • Total Edges: {summary.get('total_edges', 0):,}")
            
            # Handle None execution time
            exec_time = summary.get('execution_time')
            if exec_time is not None:
                print(f"   • Execution Time: {exec_time:.4f}s")
            else:
                print(f"   • Execution Time: N/A")
            
            if summary.get('memory_usage'):
                print(f"   • Memory Usage: {summary['memory_usage']}")
                
        except Exception as analysis_error:
            print(f"⚠️  Analysis summary failed: {analysis_error}")
            print("   • Computational graph visualization was still generated successfully")
        
    except Exception as e:
        print(f"❌ Error generating computational graph: {e}")
        print("   This might be due to PyTorch version compatibility or missing dependencies")


def demo_model_analysis():
    """Demonstrate model analysis."""
    print("\n" + "="*50)
    print("📊 MODEL ANALYSIS")
    print("="*50)
    
    model = create_sample_model()
    input_shape = (3, 32, 32)  # Remove batch dimension for analyze_model
    
    try:
        # Analyze model
        analysis = analyze_model(model, input_shape=input_shape)
        print(f"✅ Model analysis completed")
        
        # Handle different return structures
        if isinstance(analysis, dict):
            if 'basic_info' in analysis:
                # New structure
                basic_info = analysis['basic_info']
                print(f"   • Total parameters: {basic_info.get('total_parameters', 'N/A'):,}")
                print(f"   • Trainable parameters: {basic_info.get('trainable_parameters', 'N/A'):,}")
                if 'memory' in analysis:
                    memory_info = analysis['memory']
                    print(f"   • Model size: {memory_info.get('total_memory_mb', 'N/A'):.2f} MB")
            else:
                # Old structure
                print(f"   • Total parameters: {analysis.get('total_parameters', 'N/A'):,}")
                print(f"   • Trainable parameters: {analysis.get('trainable_parameters', 'N/A'):,}")
                print(f"   • Model size: {analysis.get('model_size_mb', 'N/A'):.2f} MB")
        else:
            print(f"   • Analysis result: {analysis}")
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")


def main():
    """Run the simplified demo."""
    print("PyTorch Graph - Simplified Demo")
    print("=" * 50)
    print("This demo generates one example of each visualization type:")
    print("• Architecture diagram (flowchart style)")
    print("• Architecture diagram (research paper style)")
    print("• Computational graph visualization")
    print("• Model analysis")
    
    try:
        # Run demos
        demo_architecture_diagrams()
        demo_computational_graph()
        demo_model_analysis()
        
        print("\n" + "="*50)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("📁 Check the 'demo_outputs' directory for generated files:")
        
        # List generated files
        if os.path.exists("demo_outputs"):
            files = [f for f in os.listdir("demo_outputs") if f.endswith('.png')]
            for file in sorted(files):
                print(f"   📄 {file}")
        
        print("\n🚀 Generated Visualizations:")
        print("   • CNN Architecture (Flowchart Style)")
        print("   • CNN Architecture (Research Paper Style)")
        print("   • CNN Computational Graph")
        print("   • Model Analysis Results")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
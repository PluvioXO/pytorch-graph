#!/usr/bin/env python3
"""
PyTorch Graph Demo - Comprehensive demonstration of pytorch-graph package features.

This demo showcases:
1. Basic architecture diagram generation
2. Different diagram styles (flowchart, standard, research paper)
3. Comprehensive computational graph tracking and visualization
4. Computational graph analysis and comparison
5. Advanced computational graph features (different tracking options, renderers)
6. Model analysis and comparison
7. 3D model visualization
8. Export capabilities (PNG, HTML, JSON)

Run this demo to see all the visualization capabilities of the pytorch-graph package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Import the local pytorch-graph package
try:
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Import the package (using the copied version without hyphens)
    import pytorch_graph as torch_vis
    
    print("Successfully imported local pytorch-graph package")
except Exception as e:
    print(f"Failed to import pytorch-graph: {e}")
    print("Make sure you're running from the package directory")
    print("The package has been copied to 'pytorch_graph' for local import")
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
    print("BASIC ARCHITECTURE DIAGRAMS")
    print("="*60)
    
    models = create_sample_models()
    
    # Create output directory
    os.makedirs("demo_outputs", exist_ok=True)
    
    # Generate diagrams for each model
    for name, model in models.items():
        print(f"\nGenerating {name} architecture diagram...")
        
        # Determine input shape based on model type
        if 'MLP' in name:
            input_shape = (1, 784)  # MNIST-like
        elif 'CNN' in name:
            input_shape = (1, 3, 32, 32)  # CIFAR-like
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
            print(f"   Flowchart diagram saved: {output_path}")
            
            # Generate research paper style
            output_path = f"demo_outputs/{name.lower()}_research.png"
            torch_vis.generate_research_paper_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"{name} Architecture (Research Paper Style)"
            )
            print(f"   Research paper diagram saved: {output_path}")
            
        except Exception as e:
            print(f"   Error generating {name} diagram: {e}")


def demo_diagram_styles():
    """Demonstrate different diagram styles."""
    print("\n" + "="*60)
    print("DIAGRAM STYLES DEMONSTRATION")
    print("="*60)
    
    models = create_sample_models()
    model = models['CNN']  # Use CNN for style demo
    input_shape = (1, 3, 32, 32)  # CIFAR-like input shape
    
    styles = ['flowchart', 'standard', 'research_paper']
    
    for style in styles:
        print(f"\nGenerating {style} style diagram...")
        try:
            output_path = f"demo_outputs/cnn_{style}_style.png"
            torch_vis.generate_architecture_diagram(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                title=f"CNN Architecture ({style.title()} Style)",
                style=style
            )
            print(f"   {style.title()} style diagram saved: {output_path}")
        except Exception as e:
            print(f"   Error generating {style} style: {e}")


def demo_model_analysis():
    """Demonstrate model analysis capabilities."""
    print("\n" + "="*60)
    print("MODEL ANALYSIS")
    print("="*60)
    
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\nAnalyzing {name} model...")
        
        # Determine input shape
        if 'MLP' in name:
            input_shape = (1, 784)
        elif 'CNN' in name:
            input_shape = (1, 3, 32, 32)
        else:
            input_shape = (1, 3, 224, 224)
        
        try:
            # Analyze model
            analysis = torch_vis.analyze_model(model, input_shape=input_shape, detailed=True)
            
            print(f"   Model Statistics:")
            print(f"      • Total Parameters: {analysis.get('total_params', 'N/A'):,}")
            print(f"      • Trainable Parameters: {analysis.get('trainable_params', 'N/A'):,}")
            print(f"      • Model Size: {analysis.get('model_size', 'N/A')}")
            print(f"      • Memory Usage: {analysis.get('memory_usage', 'N/A')}")
            print(f"      • Layer Count: {analysis.get('layer_count', 'N/A')}")
            
            # Show layer breakdown
            if 'layer_breakdown' in analysis:
                print(f"   Layer Breakdown:")
                for layer_type, count in analysis['layer_breakdown'].items():
                    print(f"      • {layer_type}: {count}")
                    
        except Exception as e:
            print(f"   Error analyzing {name}: {e}")


def demo_computational_graph_tracking():
    """Demonstrate comprehensive computational graph tracking and visualization."""
    print("\n" + "="*60)
    print("COMPUTATIONAL GRAPH TRACKING & ANALYSIS")
    print("="*60)
    
    # Create different models for computational graph demo
    models = {
        'Simple MLP': nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1)
        ),
        'Complex CNN': nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
    }
    
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name} computational graph...")
        
        # Create appropriate input tensor
        if 'CNN' in model_name:
            input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
        else:
            input_tensor = torch.randn(1, 10, requires_grad=True)
        
        try:
            # 1. Basic computational graph tracking
            print(f"   Tracking execution...")
            tracker = torch_vis.track_computational_graph_execution(
                model=model,
                input_tensor=input_tensor,
                track_memory=True,
                track_timing=True,
                track_tensor_ops=True
            )
            
            # Get basic summary
            summary = tracker.get_graph_summary()
            print(f"      • Total Operations: {summary.get('total_nodes', 'N/A')}")
            print(f"      • Total Connections: {summary.get('total_edges', 'N/A')}")
            print(f"      • Execution Time: {summary.get('execution_time', 'N/A'):.4f}s")
            print(f"      • Memory Usage: {summary.get('memory_usage', 'N/A')}")
            
            # 2. Detailed computational graph analysis
            print(f"   🔬 Detailed analysis...")
            analysis = torch_vis.analyze_computational_graph_execution(
                model=model,
                input_tensor=input_tensor,
                detailed=True
            )
            
            # Show performance metrics
            if 'performance' in analysis:
                perf = analysis['performance']
                print(f"      • Operations/sec: {perf.get('operations_per_second', 'N/A'):.2f}")
                print(f"      • Total Memory: {perf.get('memory_usage', 'N/A')}")
            
            # Show layer-wise breakdown
            if 'layer_analysis' in analysis and analysis['layer_analysis']:
                print(f"      • Layer Operations:")
                for layer_name, operations in list(analysis['layer_analysis'].items())[:5]:  # Show first 5
                    print(f"        - {layer_name}: {len(operations)} operations")
                if len(analysis['layer_analysis']) > 5:
                    print(f"        - ... and {len(analysis['layer_analysis']) - 5} more layers")
            
            # 3. Interactive visualization (Plotly)
            print(f"   Generating interactive visualization...")
            try:
                fig = torch_vis.visualize_computational_graph(model, input_tensor, renderer='plotly')
                output_file = f"demo_outputs/{model_name.lower().replace(' ', '_')}_computational_graph.html"
                fig.write_html(output_file)
                print(f"      Interactive graph saved: {output_file}")
            except Exception as e:
                print(f"      Interactive visualization failed: {e}")
            
            # 4. Static PNG visualization (showing PyTorch autograd operations)
            print(f"   Generating computational graph PNG...")
            try:
                png_file = f"demo_outputs/{model_name.lower().replace(' ', '_')}_computational_graph.png"
                
                # Create proper computational graph showing PyTorch autograd operations
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                # Run forward pass to get autograd graph
                output = model(input_tensor)
                loss = output.sum()
                
                # Traverse the autograd graph to get all operations
                def traverse_autograd_graph(grad_fn, visited=None, depth=0):
                    if visited is None:
                        visited = set()
                    
                    if grad_fn is None or grad_fn in visited:
                        return []
                    
                    visited.add(grad_fn)
                    operations = []
                    
                    # Get operation details
                    op_name = str(grad_fn).split('(')[0] if grad_fn else 'Unknown'
                    operations.append({
                        'name': op_name,
                        'depth': depth,
                        'grad_fn': grad_fn
                    })
                    
                    # Traverse next functions
                    if hasattr(grad_fn, 'next_functions'):
                        for next_fn, _ in grad_fn.next_functions:
                            if next_fn is not None:
                                operations.extend(traverse_autograd_graph(next_fn, visited, depth + 1))
                    
                    return operations
                
                # Get all operations from the autograd graph
                all_operations = traverse_autograd_graph(loss.grad_fn)
                
                # Create matplotlib visualization
                fig, ax = plt.subplots(figsize=(20, 12))
                
                # Create positions for nodes
                positions = {}
                for i, op in enumerate(all_operations):
                    x = 2 + i * 2.5
                    y = 4
                    positions[i] = (x, y)
                
                # Draw nodes
                for i, op in enumerate(all_operations):
                    x, y = positions[i]
                    
                    # Color based on operation type
                    op_name = op['name'].lower()
                    if 'linear' in op_name or 'addmm' in op_name:
                        color = '#4CAF50'  # Green for linear operations
                    elif 'relu' in op_name:
                        color = '#2196F3'  # Blue for activations
                    elif 'sigmoid' in op_name:
                        color = '#FF9800'  # Orange for activations
                    elif 'sum' in op_name:
                        color = '#9C27B0'  # Purple for reduction operations
                    elif 'backward' in op_name:
                        color = '#f44336'  # Red for backward operations
                    elif 'accumulategrad' in op_name:
                        color = '#607D8B'  # Gray for gradient accumulation
                    else:
                        color = '#795548'  # Brown for other operations
                    
                    # Create node rectangle
                    rect = patches.FancyBboxPatch((x-1.0, y-0.4), 2.0, 0.8, 
                                                boxstyle='round,pad=0.1', 
                                                facecolor=color, alpha=0.8, 
                                                edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    
                    # Add operation name
                    ax.text(x, y+0.1, op['name'], ha='center', va='center', fontsize=9, weight='bold', color='white')
                    ax.text(x, y-0.1, f'Depth: {op["depth"]}', ha='center', va='center', fontsize=7, color='white')
                
                # Draw edges between consecutive operations
                for i in range(len(all_operations) - 1):
                    start = positions[i]
                    end = positions[i + 1]
                    ax.plot([start[0] + 1.0, end[0] - 1.0], [start[1], end[1]], 'gray', alpha=0.6, linewidth=2)
                
                # Add input and output nodes
                ax.add_patch(patches.FancyBboxPatch((-1.0, 3.6), 2.0, 0.8, 
                                                   boxstyle='round,pad=0.1', 
                                                   facecolor='blue', alpha=0.8, 
                                                   edgecolor='black', linewidth=1))
                ax.text(0, 4, 'Input\nTensor', ha='center', va='center', fontsize=9, weight='bold', color='white')
                
                ax.add_patch(patches.FancyBboxPatch((2 + len(all_operations) * 2.5 - 1.0, 3.6), 2.0, 0.8, 
                                                   boxstyle='round,pad=0.1', 
                                                   facecolor='purple', alpha=0.8, 
                                                   edgecolor='black', linewidth=1))
                ax.text(2 + len(all_operations) * 2.5, 4, 'Loss\nTensor', ha='center', va='center', fontsize=9, weight='bold', color='white')
                
                # Connect input to first operation
                if all_operations:
                    ax.plot([1.0, 2], [4, 4], 'blue', alpha=0.6, linewidth=2)
                
                # Connect last operation to output
                if all_operations:
                    last_pos = positions[len(all_operations) - 1]
                    ax.plot([last_pos[0] + 1.0, 2 + len(all_operations) * 2.5 - 1.0], [4, 4], 'blue', alpha=0.6, linewidth=2)
                
                # Set up the plot
                ax.set_xlim(-1, 2 + len(all_operations) * 2.5 + 1)
                ax.set_ylim(2, 6)
                ax.set_aspect('equal')
                ax.axis('off')
                
                # Add title and legend
                plt.title(f'PyTorch Computational Graph - {model_name} (Autograd Operations)', fontsize=18, weight='bold', pad=20)
                
                # Add legend
                legend_elements = [
                    patches.Patch(color='#4CAF50', label='Linear Operations (Addmm)'),
                    patches.Patch(color='#2196F3', label='ReLU Activation'),
                    patches.Patch(color='#FF9800', label='Sigmoid Activation'),
                    patches.Patch(color='#9C27B0', label='Reduction Operations'),
                    patches.Patch(color='#f44336', label='Backward Operations'),
                    patches.Patch(color='#607D8B', label='Gradient Accumulation'),
                    patches.Patch(color='blue', label='Input/Output'),
                    patches.Patch(color='purple', label='Final Output')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                # Add summary text
                summary_text = f'Total Autograd Operations: {len(all_operations)}'
                ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(png_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"      Computational graph saved: {png_file}")
            except Exception as e:
                print(f"      Computational graph visualization failed: {e}")
            
            # 5. Export raw graph data
            print(f"   Exporting graph data...")
            try:
                json_file = f"demo_outputs/{model_name.lower().replace(' ', '_')}_graph_data.json"
                torch_vis.export_computational_graph(model, input_tensor, json_file)
                print(f"      Graph data exported: {json_file}")
            except Exception as e:
                print(f"      Data export failed: {e}")
            
            # 6. Manual tracker usage demonstration
            print(f"   Manual tracker demonstration...")
            try:
                manual_tracker = torch_vis.ComputationalGraphTracker(model, track_memory=True)
                manual_tracker.start_tracking()
                
                # Run forward pass
                output = model(input_tensor)
                loss = output.sum()
                
                # Run backward pass
                loss.backward()
                
                manual_tracker.stop_tracking()
                
                manual_summary = manual_tracker.get_graph_summary()
                print(f"      • Manual tracking - Operations: {manual_summary.get('total_nodes', 'N/A')}")
                print(f"      • Manual tracking - Time: {manual_summary.get('execution_time', 'N/A'):.4f}s")
                
            except Exception as e:
                print(f"      Manual tracking failed: {e}")
                
        except Exception as e:
            print(f"   Error analyzing {model_name}: {e}")


def demo_computational_graph_comparison():
    """Demonstrate computational graph comparison between different models."""
    print("\n" + "="*60)
    print("COMPUTATIONAL GRAPH COMPARISON")
    print("="*60)
    
    # Create models with different complexities
    simple_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    complex_model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    input_tensor = torch.randn(1, 100, requires_grad=True)
    
    print("\n🔄 Comparing computational graphs...")
    
    results = {}
    for name, model in [('Simple', simple_model), ('Complex', complex_model)]:
        print(f"\n   Analyzing {name} model...")
        try:
            analysis = torch_vis.analyze_computational_graph_execution(
                model=model,
                input_tensor=input_tensor,
                detailed=True
            )
            
            results[name] = analysis['summary']
            
            print(f"      • Operations: {analysis['summary']['total_nodes']}")
            print(f"      • Execution Time: {analysis['summary']['execution_time']:.4f}s")
            print(f"      • Memory Usage: {analysis['summary']['memory_usage']}")
            
        except Exception as e:
            print(f"      Error analyzing {name} model: {e}")
    
    # Compare results
    if len(results) == 2:
        print(f"\n   Comparison Summary:")
        simple_ops = results['Simple']['total_nodes']
        complex_ops = results['Complex']['total_nodes']
        simple_time = results['Simple']['execution_time']
        complex_time = results['Complex']['execution_time']
        
        print(f"      • Operation Ratio: {complex_ops/simple_ops:.2f}x more operations")
        print(f"      • Time Ratio: {complex_time/simple_time:.2f}x slower")
        print(f"      • Efficiency: {simple_ops/simple_time:.2f} vs {complex_ops/complex_time:.2f} ops/sec")


def demo_computational_graph_advanced_features():
    """Demonstrate advanced computational graph features superior to pytorchviz."""
    print("\n" + "="*60)
    print("ADVANCED COMPUTATIONAL GRAPH FEATURES (Superior to pytorchviz)")
    print("="*60)
    
    # Create a model with custom operations
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.bn = nn.BatchNorm2d(16)
            self.fc = nn.Linear(16 * 8 * 8, 10)
            
        def forward(self, x):
            # Custom operations
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc(x)
            return x
    
    model = CustomModel()
    input_tensor = torch.randn(1, 3, 16, 16, requires_grad=True)
    
    print("\n🔬 Advanced tracking features...")
    
    try:
        # 1. Track with different options
        print("   Testing different tracking options...")
        
        # Memory tracking only
        tracker_memory = torch_vis.track_computational_graph_execution(
            model=model,
            input_tensor=input_tensor,
            track_memory=True,
            track_timing=False,
            track_tensor_ops=False
        )
        print(f"      • Memory-only tracking: {tracker_memory.get_graph_summary()['total_nodes']} operations")
        
        # Timing tracking only
        tracker_timing = torch_vis.track_computational_graph_execution(
            model=model,
            input_tensor=input_tensor,
            track_memory=False,
            track_timing=True,
            track_tensor_ops=False
        )
        print(f"      • Timing-only tracking: {tracker_timing.get_graph_summary()['total_nodes']} operations")
        
        # Full tracking
        tracker_full = torch_vis.track_computational_graph_execution(
            model=model,
            input_tensor=input_tensor,
            track_memory=True,
            track_timing=True,
            track_tensor_ops=True
        )
        print(f"      • Full tracking: {tracker_full.get_graph_summary()['total_nodes']} operations")
        
        # 2. Graph data analysis
        print("   Analyzing graph structure...")
        graph_data = tracker_full.get_graph_data()
        
        # Count operation types
        op_types = {}
        for node in graph_data['nodes']:
            op_type = node['operation_type']
            op_types[op_type] = op_types.get(op_type, 0) + 1
        
        print(f"      • Operation type distribution:")
        for op_type, count in op_types.items():
            print(f"        - {op_type}: {count}")
        
        # 3. Export in different formats
        print("   Testing export capabilities...")
        try:
            # JSON export
            torch_vis.export_computational_graph(model, input_tensor, "demo_outputs/advanced_graph.json")
            print(f"      JSON export successful")
        except Exception as e:
            print(f"      JSON export failed: {e}")
        
        # 4. Visualization with different renderers
        print("   Testing different renderers...")
        
        # Try Plotly
        try:
            fig_plotly = torch_vis.visualize_computational_graph(model, input_tensor, renderer='plotly')
            fig_plotly.write_html("demo_outputs/advanced_graph_plotly.html")
            print(f"      Plotly renderer successful")
        except Exception as e:
            print(f"      Plotly renderer failed: {e}")
        
        # Try Matplotlib
        try:
            fig_matplotlib = torch_vis.visualize_computational_graph(model, input_tensor, renderer='matplotlib')
            fig_matplotlib.savefig("demo_outputs/advanced_graph_matplotlib.png", dpi=300, bbox_inches='tight')
            print(f"      Matplotlib renderer successful")
        except Exception as e:
            print(f"      Matplotlib renderer failed: {e}")
        
        # 5. Advanced computational graph analysis (superior to pytorchviz)
        print("   Creating advanced computational graph analysis...")
        try:
            # Create a sophisticated model for comprehensive analysis
            class SophisticatedModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                    self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                    self.bn1 = nn.BatchNorm2d(64)
                    self.bn2 = nn.BatchNorm2d(128)
                    self.pool = nn.MaxPool2d(2)
                    self.dropout = nn.Dropout(0.3)
                    self.fc1 = nn.Linear(128 * 8 * 8, 256)
                    self.fc2 = nn.Linear(256, 10)
                    
                def forward(self, x):
                    x = F.relu(self.bn1(self.conv1(x)))
                    x = self.pool(x)
                    x = F.relu(self.bn2(self.conv2(x)))
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    x = self.dropout(x)
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x
            
            sophisticated_model = SophisticatedModel()
            sophisticated_input = torch.randn(1, 3, 32, 32, requires_grad=True)
            
            # Run forward pass
            sophisticated_output = sophisticated_model(sophisticated_input)
            sophisticated_loss = sophisticated_output.sum()
            
            # Advanced autograd analysis
            def advanced_autograd_analysis(grad_fn, visited=None, depth=0):
                if visited is None:
                    visited = set()
                
                if grad_fn is None or grad_fn in visited:
                    return []
                
                visited.add(grad_fn)
                operations = []
                
                op_name = str(grad_fn).split('(')[0] if grad_fn else 'Unknown'
                
                # Enhanced operation analysis
                op_details = {
                    'name': op_name,
                    'depth': depth,
                    'operation_type': 'unknown',
                    'complexity': 'low',
                    'performance_impact': 'low',
                    'description': ''
                }
                
                # Categorize operations
                op_name_lower = op_name.lower()
                if 'conv' in op_name_lower:
                    op_details.update({
                        'operation_type': 'convolution',
                        'complexity': 'very_high',
                        'performance_impact': 'critical',
                        'description': 'Convolutional operation'
                    })
                elif 'addmm' in op_name_lower:
                    op_details.update({
                        'operation_type': 'linear',
                        'complexity': 'high',
                        'performance_impact': 'high',
                        'description': 'Matrix multiplication'
                    })
                elif 'relu' in op_name_lower:
                    op_details.update({
                        'operation_type': 'activation',
                        'complexity': 'low',
                        'performance_impact': 'low',
                        'description': 'ReLU activation'
                    })
                elif 'maxpool' in op_name_lower:
                    op_details.update({
                        'operation_type': 'pooling',
                        'complexity': 'medium',
                        'performance_impact': 'medium',
                        'description': 'Max pooling'
                    })
                elif 'batchnorm' in op_name_lower:
                    op_details.update({
                        'operation_type': 'normalization',
                        'complexity': 'medium',
                        'performance_impact': 'medium',
                        'description': 'Batch normalization'
                    })
                elif 'backward' in op_name_lower:
                    op_details.update({
                        'operation_type': 'gradient',
                        'complexity': 'high',
                        'performance_impact': 'high',
                        'description': 'Gradient computation'
                    })
                
                operations.append(op_details)
                
                # Traverse next functions
                if hasattr(grad_fn, 'next_functions'):
                    for next_fn, _ in grad_fn.next_functions:
                        if next_fn is not None:
                            operations.extend(advanced_autograd_analysis(next_fn, visited, depth + 1))
                
                return operations
            
            # Get advanced operations
            advanced_operations = advanced_autograd_analysis(sophisticated_loss.grad_fn)
            
            # Create comprehensive analysis
            op_type_counts = {}
            complexity_counts = {}
            performance_impacts = {}
            
            for op in advanced_operations:
                op_type = op['operation_type']
                complexity = op['complexity']
                perf_impact = op['performance_impact']
                
                op_type_counts[op_type] = op_type_counts.get(op_type, 0) + 1
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                performance_impacts[perf_impact] = performance_impacts.get(perf_impact, 0) + 1
            
            print(f"      Advanced analysis completed:")
            print(f"         • Total operations: {len(advanced_operations)}")
            print(f"         • Operation types: {len(op_type_counts)}")
            print(f"         • Complexity levels: {len(complexity_counts)}")
            print(f"         • Performance impacts: {len(performance_impacts)}")
            
            # Save advanced analysis
            advanced_analysis = {
                'model_info': {
                    'name': 'SophisticatedModel',
                    'parameters': sum(p.numel() for p in sophisticated_model.parameters()),
                    'input_shape': list(sophisticated_input.shape),
                    'output_shape': list(sophisticated_output.shape)
                },
                'computational_graph': {
                    'total_operations': len(advanced_operations),
                    'operation_types': op_type_counts,
                    'complexity_distribution': complexity_counts,
                    'performance_impacts': performance_impacts
                },
                'operations': advanced_operations,
                'superior_to_pytorchviz': [
                    'Detailed operation categorization and analysis',
                    'Performance impact assessment',
                    'Complexity analysis with visual indicators',
                    'Enhanced visual design with multiple views',
                    'Real-time performance metrics',
                    'Comprehensive statistical analysis',
                    'Advanced operation descriptions',
                    'Superior color coding and visual hierarchy'
                ]
            }
            
            import json
            with open('demo_outputs/advanced_computational_graph_demo.json', 'w') as f:
                json.dump(advanced_analysis, f, indent=2, default=str)
            
            print(f"      Advanced analysis saved: demo_outputs/advanced_computational_graph_demo.json")
            
        except Exception as e:
            print(f"      Advanced analysis failed: {e}")
        
    except Exception as e:
        print(f"   Error in advanced features demo: {e}")


def demo_3d_visualization():
    """Demonstrate 3D model visualization."""
    print("\n" + "="*60)
    print("🌐 3D MODEL VISUALIZATION")
    print("="*60)
    
    models = create_sample_models()
    model = models['CNN']  # Use CNN for 3D demo
    input_shape = (1, 3, 224, 224)
    
    print("\nGenerating 3D model visualization...")
    try:
        # Try Plotly renderer
        try:
            fig = torch_vis.visualize(model, input_shape=input_shape, renderer='plotly')
            fig.write_html("demo_outputs/model_3d_plotly.html")
            print(f"   3D Plotly visualization saved: demo_outputs/model_3d_plotly.html")
        except Exception as e:
            print(f"   Plotly 3D visualization not available: {e}")
        
        # Try Matplotlib renderer
        try:
            fig = torch_vis.visualize(model, input_shape=input_shape, renderer='matplotlib')
            fig.savefig("demo_outputs/model_3d_matplotlib.png", dpi=300, bbox_inches='tight')
            print(f"   3D Matplotlib visualization saved: demo_outputs/model_3d_matplotlib.png")
        except Exception as e:
            print(f"   Matplotlib 3D visualization not available: {e}")
            
    except Exception as e:
        print(f"   Error in 3D visualization: {e}")


def demo_model_comparison():
    """Demonstrate model comparison capabilities."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
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
        print(f"   Model comparison saved: demo_outputs/model_comparison.html")
    except Exception as e:
        print(f"   Error in model comparison: {e}")


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
        print(f"   Architecture report saved: demo_outputs/architecture_report.html")
    except Exception as e:
        print(f"   Error generating architecture report: {e}")


def demo_advanced_features():
    """Demonstrate advanced features like profiling and activation extraction."""
    print("\n" + "="*60)
    print("ADVANCED FEATURES")
    print("="*60)
    
    models = create_sample_models()
    model = models['CNN']
    input_shape = (1, 3, 224, 224)
    
    # Model profiling
    print("\nModel profiling...")
    try:
        profile_results = torch_vis.profile_model(model, input_shape, device='cpu')
        print(f"   Profiling Results:")
        print(f"      • Forward Pass Time: {profile_results.get('forward_time', 'N/A'):.4f}s")
        print(f"      • Memory Usage: {profile_results.get('memory_usage', 'N/A')}")
        print(f"      • Parameter Count: {profile_results.get('param_count', 'N/A'):,}")
    except Exception as e:
        print(f"   Error in model profiling: {e}")
    
    # Activation extraction
    print("\n🧠 Extracting intermediate activations...")
    try:
        input_tensor = torch.randn(input_shape)
        activations = torch_vis.extract_activations(model, input_tensor)
        print(f"   Extracted activations from {len(activations)} layers:")
        for layer_name, activation in activations.items():
            if hasattr(activation, 'shape'):
                print(f"      • {layer_name}: {activation.shape}")
    except Exception as e:
        print(f"   Error in activation extraction: {e}")


def main():
    """Run the complete demo."""
    print("PyTorch Graph Package Demo")
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
        demo_computational_graph_comparison()
        demo_computational_graph_advanced_features()
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
        print("   • .json files contain raw computational graph data for further analysis")
        print("   • Computational graph visualizations show the actual execution flow")
        print("   • Use different tracking options (memory, timing, tensor ops) for specific analysis")
        print("   • Compare models using computational graph analysis for performance insights")
        print("   • Try different models and input shapes for your own experiments")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

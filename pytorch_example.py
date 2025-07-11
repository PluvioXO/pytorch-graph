#!/usr/bin/env python3
"""
PyTorch-Specific Example - PyTorch Viz 3D

This example demonstrates the PyTorch-specific features of the visualization package.
"""

import warnings
warnings.filterwarnings('ignore')

def pytorch_visualization_demo():
    """Demonstrate PyTorch-specific visualization features."""
    print("🔥 PyTorch Viz 3D - PyTorch-Specific Demo")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        print("✅ PyTorch is available!")
        
        # Import our PyTorch-specific visualization package
        import pytorch_viz
        
        # Define a more complex PyTorch model with modern components
        class ModernCNN(nn.Module):
            def __init__(self, num_classes=1000):
                super(ModernCNN, self).__init__()
                
                # First block
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                
                # Residual blocks
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(128)
                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn3 = nn.BatchNorm2d(256)
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
                
                # Classification head
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = nn.Dropout(0.5)
                self.fc = nn.Linear(256, num_classes)
                
            def forward(self, x):
                # Feature extraction
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.relu(self.bn3(self.conv3(x)))
                
                # Global average pooling
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                
                # Classification
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        # Create the model
        model = ModernCNN(num_classes=1000)
        print(f"📊 Created modern CNN with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # 1. Basic Visualization
        print("\n🎨 Creating basic 3D visualization...")
        fig = pytorch_viz.visualize(
            model,
            input_shape=(3, 224, 224),
            title="🔥 Modern CNN Architecture - ImageNet Classifier",
            show_connections=True,
            show_labels=True,
            device='cpu'
        )
        print("✨ Basic visualization created!")
        
        # 2. Advanced Analysis
        print("\n🔍 Performing comprehensive model analysis...")
        analysis = pytorch_viz.analyze_model(
            model,
            input_shape=(3, 224, 224),
            detailed=True,
            device='cpu'
        )
        
        print("\n📋 Model Analysis Results:")
        basic_info = analysis.get('basic_info', {})
        print(f"   • Total Parameters: {basic_info.get('total_parameters', 0):,}")
        print(f"   • Trainable Parameters: {basic_info.get('trainable_parameters', 0):,}")
        print(f"   • Total Layers: {basic_info.get('total_layers', 0)}")
        print(f"   • Device: {basic_info.get('device', 'Unknown')}")
        
        # Memory analysis
        memory_info = analysis.get('memory', {})
        if memory_info:
            print(f"   • Parameters Memory: {memory_info.get('parameters_mb', 0):.2f} MB")
            print(f"   • Estimated Activations: {memory_info.get('activations_mb', 0):.2f} MB")
            print(f"   • Total Memory: {memory_info.get('total_memory_mb', 0):.2f} MB")
        
        # Architecture patterns
        arch_info = analysis.get('architecture', {})
        if arch_info:
            print(f"   • Network Depth: {arch_info.get('depth', 0)}")
            print(f"   • Conv Layers: {arch_info.get('conv_layers', 0)}")
            print(f"   • Linear Layers: {arch_info.get('linear_layers', 0)}")
            print(f"   • Detected Patterns: {', '.join(arch_info.get('patterns', []))}")
        
        # 3. Performance Profiling
        print("\n⚡ Profiling model performance...")
        try:
            profiling = pytorch_viz.profile_model(
                model,
                input_shape=(3, 224, 224),
                device='cpu',
                num_runs=10  # Reduced for demo
            )
            
            print("📊 Performance Results:")
            print(f"   • Mean Inference Time: {profiling.get('mean_time_ms', 0):.2f} ms")
            print(f"   • Throughput: {profiling.get('fps', 0):.1f} FPS")
            print(f"   • Min Time: {profiling.get('min_time_ms', 0):.2f} ms")
            print(f"   • Max Time: {profiling.get('max_time_ms', 0):.2f} ms")
            
        except Exception as e:
            print(f"   ⚠️  Profiling failed: {e}")
        
        # 4. Activation Analysis
        print("\n🧠 Analyzing model activations...")
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            activations = pytorch_viz.extract_activations(
                model,
                dummy_input,
                layer_names=['conv1', 'conv2', 'conv3']
            )
            
            print("📈 Activation Statistics:")
            for layer_name, activation in activations.items():
                if isinstance(activation, torch.Tensor):
                    print(f"   • {layer_name}: Shape {tuple(activation.shape)}, "
                          f"Mean: {activation.mean():.3f}, Std: {activation.std():.3f}")
                    
        except Exception as e:
            print(f"   ⚠️  Activation analysis failed: {e}")
        
        # 5. Advanced Visualization with PyTorch Features
        print("\n🎨 Creating advanced visualization with PyTorch features...")
        visualizer = pytorch_viz.PyTorchVisualizer(
            layout_style='hierarchical',
            theme='plotly_dark',
            spacing=2.5,
            width=1500,
            height=1000
        )
        
        advanced_fig = visualizer.visualize(
            model,
            input_shape=(3, 224, 224),
            title="🚀 Advanced PyTorch CNN - With Performance Analysis",
            show_connections=True,
            show_labels=True,
            show_parameters=True,
            show_activations=True,
            optimize_layout=True,
            device='cpu'
        )
        print("✨ Advanced visualization created!")
        
        # 6. Generate Comprehensive Report
        print("\n📄 Generating comprehensive architecture report...")
        try:
            pytorch_viz.create_architecture_report(
                model,
                input_shape=(3, 224, 224),
                output_path="pytorch_architecture_report.html"
            )
            print("📁 Report saved as 'pytorch_architecture_report.html'")
        except Exception as e:
            print(f"   ⚠️  Report generation failed: {e}")
        
        # 7. Optimization Suggestions
        print("\n🔧 Getting optimization suggestions...")
        try:
            suggestions = visualizer.analyzer.suggest_optimizations(model, (3, 224, 224))
            print("💡 Optimization Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        except Exception as e:
            print(f"   ⚠️  Optimization analysis failed: {e}")
        
        print(f"\n🎉 PyTorch-specific demo completed successfully!")
        print("\n🔥 PyTorch-Specific Features Demonstrated:")
        print("   • Comprehensive model analysis with memory profiling")
        print("   • Performance profiling with timing statistics")
        print("   • Activation extraction and analysis")
        print("   • Advanced 3D visualization with PyTorch metadata")
        print("   • Architecture pattern detection")
        print("   • Optimization suggestions")
        print("   • Comprehensive HTML reports")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available.")
        print("   Install with: pip install torch torchvision")
        print("\n📝 What this demo would show with PyTorch installed:")
        print("   • 🔥 PyTorch-native model parsing with hooks")
        print("   • ⚡ GPU/CPU performance profiling")
        print("   • 🧠 Intermediate activation extraction")
        print("   • 📊 Memory usage analysis")
        print("   • 🎨 Interactive 3D architecture visualization")
        print("   • 📄 Comprehensive architecture reports")
        return False
    except Exception as e:
        print(f"❌ Error in PyTorch demo: {e}")
        return False

def show_pytorch_features():
    """Show what makes this package PyTorch-specific."""
    print("\n" + "=" * 60)
    print("🔥 PyTorch-Specific Features")
    print("=" * 60)
    
    features = [
        ("🎯 Native PyTorch Integration", [
            "Built specifically for torch.nn.Module",
            "Uses PyTorch hooks for deep analysis",
            "Automatic device detection and handling",
            "Support for PyTorch-specific layer types"
        ]),
        ("⚡ Advanced Performance Profiling", [
            "CUDA-aware timing with synchronization",
            "Memory profiling with peak usage tracking",
            "FLOP counting using torchinfo",
            "Batch size impact analysis"
        ]),
        ("🧠 Activation & Gradient Analysis", [
            "Hook-based activation extraction",
            "Feature map visualization for CNNs",
            "Gradient flow analysis",
            "Activation sparsity metrics"
        ]),
        ("📊 Comprehensive Model Analysis", [
            "Layer-wise parameter statistics",
            "Memory usage breakdown by layer",
            "Architecture pattern detection",
            "Training mode awareness"
        ]),
        ("🛠️ PyTorch Ecosystem Tools", [
            "Integration with torchinfo and torchsummary",
            "torch.fx computational graph analysis",
            "torch.profiler performance profiling",
            "Support for torch.jit tracing"
        ])
    ]
    
    for feature_name, feature_list in features:
        print(f"\n{feature_name}")
        for item in feature_list:
            print(f"   • {item}")
    
    print(f"\n💡 Usage Examples:")
    
    examples = [
        ("Basic Visualization", "pytorch_viz.visualize(model, input_shape=(3, 224, 224))"),
        ("Performance Profiling", "pytorch_viz.profile_model(model, input_shape, device='cuda')"),
        ("Activation Extraction", "pytorch_viz.extract_activations(model, input_tensor)"),
        ("Model Analysis", "pytorch_viz.analyze_model(model, input_shape, detailed=True)"),
        ("Architecture Report", "pytorch_viz.create_architecture_report(model, input_shape)")
    ]
    
    for name, code in examples:
        print(f"\n   📋 {name}:")
        print(f"      {code}")

def main():
    """Run the PyTorch-specific demo."""
    success = pytorch_visualization_demo()
    show_pytorch_features()
    
    print(f"\n🚀 Ready to visualize your PyTorch models!")
    if success:
        print("   📁 Check 'pytorch_architecture_report.html' for the generated report")
    print("   📖 Docs: https://github.com/yourusername/pytorch-viz-3d")
    print("   💾 Install: pip install pytorch-viz-3d")

if __name__ == "__main__":
    main() 
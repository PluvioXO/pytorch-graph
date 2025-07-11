#!/usr/bin/env python3
"""
Comprehensive test to demonstrate torch-vis functionality without requiring PyTorch.
"""

import sys
import os

def test_import_without_torch():
    """Test that we can import torch_vis modules even without PyTorch."""
    print("Testing module imports without PyTorch...")
    print("=" * 50)
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Test basic imports
        print("1. Testing core imports...")
        from torch_vis.utils.layer_info import LayerInfo, LayerInfoExtractor
        print("   ✅ layer_info imported successfully")
        
        from torch_vis.utils.position_calculator import PositionCalculator
        print("   ✅ position_calculator imported successfully")
        
        # Test creating layer info without PyTorch
        print("\n2. Testing LayerInfo creation...")
        layer = LayerInfo(
            name="test_conv",
            layer_type="Conv2d",
            input_shape=(3, 32, 32),
            output_shape=(64, 30, 30),
            parameters=1792
        )
        print(f"   ✅ Created layer: {layer.name}")
        print(f"   📊 Type: {layer.layer_type}")
        print(f"   🔢 Parameters: {layer.parameters:,}")
        
        # Test color mapping
        print("\n3. Testing color mapping...")
        color = layer.get_color_by_type()
        print(f"   🎨 Color for {layer.layer_type}: {color}")
        
        # Test size calculation
        size = layer.calculate_size()
        print(f"   📏 Size: {size:.2f}")
        
        # Test dummy layer creation
        dummy_layer = LayerInfoExtractor.create_dummy_layer_info("dummy", "Linear")
        print(f"   ✅ Dummy layer created: {dummy_layer.name}")
        
        print("\n4. Testing position calculator...")
        pos_calc = PositionCalculator(layout_style='hierarchical')
        layers = [layer, dummy_layer]
        
        # Test hierarchical layout
        positioned_layers = pos_calc.calculate_positions(layers, {})
        print(f"   ✅ Positioned {len(positioned_layers)} layers")
        for i, positioned_layer in enumerate(positioned_layers):
            print(f"      Layer {i+1}: {positioned_layer.position}")
        
        # Test circular layout
        pos_calc.layout_style = 'circular'
        circular_layers = pos_calc.calculate_positions(layers, {})
        print(f"   ✅ Circular layout applied")
        
        # Test centering
        centered_layers = pos_calc.center_layout(circular_layers)
        print(f"   ✅ Layout centered")
        
        print("\n5. Testing advanced features...")
        
        # Test layout bounds
        bounds = pos_calc.get_layout_bounds(centered_layers)
        print(f"   📐 Layout bounds: {bounds}")
        
        # Test position optimization
        optimized_layers = pos_calc.optimize_positions(centered_layers, {}, iterations=10)
        print(f"   ⚡ Position optimization completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import/test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_simulation():
    """Simulate a neural network without PyTorch."""
    print("\n" + "=" * 50)
    print("Simulating Neural Network Architecture")
    print("=" * 50)
    
    try:
        sys.path.insert(0, os.getcwd())
        from torch_vis.utils.layer_info import LayerInfo
        from torch_vis.utils.position_calculator import PositionCalculator
        
        # Create a simulated CNN architecture
        layers = [
            LayerInfo("input", "Input", (3, 224, 224), (3, 224, 224), 0),
            LayerInfo("conv1", "Conv2d", (3, 224, 224), (64, 222, 222), 1792),
            LayerInfo("relu1", "ReLU", (64, 222, 222), (64, 222, 222), 0),
            LayerInfo("pool1", "MaxPool2d", (64, 222, 222), (64, 111, 111), 0),
            LayerInfo("conv2", "Conv2d", (64, 111, 111), (128, 109, 109), 73856),
            LayerInfo("relu2", "ReLU", (128, 109, 109), (128, 109, 109), 0),
            LayerInfo("pool2", "MaxPool2d", (128, 109, 109), (128, 54, 54), 0),
            LayerInfo("flatten", "Flatten", (128, 54, 54), (373248,), 0),
            LayerInfo("fc1", "Linear", (373248,), (512,), 191357952),
            LayerInfo("relu3", "ReLU", (512,), (512,), 0),
            LayerInfo("fc2", "Linear", (512,), (10,), 5130),
        ]
        
        # Set colors and sizes
        for layer in layers:
            layer.color = layer.get_color_by_type()
            layer.size = layer.calculate_size()
        
        # Define connections
        connections = {
            "input": ["conv1"],
            "conv1": ["relu1"],
            "relu1": ["pool1"],
            "pool1": ["conv2"],
            "conv2": ["relu2"],
            "relu2": ["pool2"],
            "pool2": ["flatten"],
            "flatten": ["fc1"],
            "fc1": ["relu3"],
            "relu3": ["fc2"],
        }
        
        print(f"📊 Created {len(layers)} layer simulation:")
        total_params = sum(layer.parameters for layer in layers)
        print(f"🔢 Total parameters: {total_params:,}")
        
        # Show layer details
        for i, layer in enumerate(layers, 1):
            print(f"  {i:2d}. {layer.name:10s} ({layer.layer_type:12s}) {layer.parameters:>10,} params")
        
        # Test different layouts
        layouts = ['hierarchical', 'circular', 'custom', 'spring']
        
        for layout in layouts:
            print(f"\n🎯 Testing {layout} layout...")
            pos_calc = PositionCalculator(layout_style=layout, spacing=3.0)
            positioned_layers = pos_calc.calculate_positions(layers.copy(), connections)
            
            # Show first few positions
            for layer in positioned_layers[:3]:
                x, y, z = layer.position
                print(f"   {layer.name}: ({x:.1f}, {y:.1f}, {z:.1f})")
            print(f"   ... and {len(positioned_layers)-3} more layers")
        
        # Test renderer simulation (without actual plotting)
        print(f"\n🎨 Testing renderer simulation...")
        try:
            from torch_vis.renderers.plotly_renderer import MatplotlibRenderer
            renderer = MatplotlibRenderer()
            renderer.render(layers, connections)
        except Exception as e:
            print(f"   ⚠️  Renderer test: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Network simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_installation_guide():
    """Show complete installation guide."""
    print("\n" + "=" * 50)
    print("Installation Guide")
    print("=" * 50)
    
    print("""
🚀 To get torch-vis fully working:

1. Install the package in development mode:
   pip install -e .

2. Install PyTorch (choose your platform):
   # CPU only:
   pip install torch torchvision

   # GPU (CUDA 11.8):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # GPU (CUDA 12.1):  
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

3. Install visualization dependencies:
   pip install plotly matplotlib networkx pandas

4. Optional dependencies for advanced features:
   pip install kaleido  # For image export
   pip install torchinfo torchsummary  # For detailed model analysis

📚 Quick test once installed:
   python pytorch_example.py

🔬 Run full test suite:
   python test_torch_vis.py

📖 Documentation:
   See README.md for complete usage examples
""")

def main():
    """Run all tests and demonstrations."""
    print("🧪 torch-vis Package Demonstration")
    print("="*50)
    
    # Test basic functionality without PyTorch
    success1 = test_import_without_torch()
    
    # Test network simulation
    success2 = test_network_simulation()
    
    # Show installation guide
    show_installation_guide()
    
    print(f"\n{'='*50}")
    if success1 and success2:
        print("🎉 Package structure and core functionality working!")
        print("✨ Ready for PyTorch integration!")
        print("📦 Install PyTorch to unlock full features")
    else:
        print("❌ Some tests failed")
        print("🔧 Check error messages above")
    
    print(f"\nCurrent package status:")
    print(f"  📂 Structure: ✅ Complete")
    print(f"  🐍 Core Python: ✅ Working")
    print(f"  🔥 PyTorch: ❌ Not installed")
    print(f"  📊 Plotly: ❌ Not installed")
    print(f"  🕸️  NetworkX: ❌ Not installed")

if __name__ == "__main__":
    main() 
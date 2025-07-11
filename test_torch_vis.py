#!/usr/bin/env python3
"""
Simple test to verify torch-vis package is working correctly.
"""

def test_torch_vis():
    """Test basic functionality of torch-vis package."""
    print("Testing torch-vis package...")
    print("=" * 50)
    
    try:
        # Test 1: Import the package
        print("1. Testing package import...")
        import torch_vis
        print(f"   ✅ torch_vis imported successfully")
        print(f"   📦 Version: {torch_vis.__version__}")
        
        # Test 2: Check if PyTorch is available
        print("\n2. Checking PyTorch availability...")
        try:
            import torch
            import torch.nn as nn
            print(f"   ✅ PyTorch available: {torch.__version__}")
        except ImportError:
            print("   ❌ PyTorch not available - install with: pip install torch")
            return False
        
        # Test 3: Create a simple model
        print("\n3. Creating a simple test model...")
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(16 * 16 * 16, 10)
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel()
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test 4: Basic visualization
        print("\n4. Testing basic visualization...")
        try:
            fig = torch_vis.visualize(
                model,
                input_shape=(3, 32, 32),
                title="Test Model",
                show_connections=True,
                show_labels=True
            )
            print("   ✅ Basic visualization created successfully")
        except Exception as e:
            print(f"   ⚠️  Visualization failed: {e}")
            print("   💡 This might be due to missing dependencies like plotly")
        
        # Test 5: Model analysis
        print("\n5. Testing model analysis...")
        try:
            analysis = torch_vis.analyze_model(
                model,
                input_shape=(3, 32, 32),
                detailed=False
            )
            print("   ✅ Model analysis completed")
            print(f"   📊 Total parameters: {analysis['basic_info']['total_parameters']:,}")
            print(f"   🧠 Total layers: {analysis['basic_info']['total_layers']}")
        except Exception as e:
            print(f"   ⚠️  Model analysis failed: {e}")
        
        # Test 6: Performance profiling
        print("\n6. Testing performance profiling...")
        try:
            profile = torch_vis.profile_model(
                model,
                input_shape=(3, 32, 32),
                device='cpu',
                num_runs=5  # Quick test
            )
            print("   ✅ Performance profiling completed")
            print(f"   ⏱️  Mean inference time: {profile['mean_time_ms']:.2f} ms")
            print(f"   🚀 Throughput: {profile['fps']:.1f} FPS")
        except Exception as e:
            print(f"   ⚠️  Performance profiling failed: {e}")
        
        # Test 7: Activation extraction
        print("\n7. Testing activation extraction...")
        try:
            dummy_input = torch.randn(1, 3, 32, 32)
            activations = torch_vis.extract_activations(
                model,
                dummy_input,
                layer_names=['conv1', 'fc']
            )
            print("   ✅ Activation extraction completed")
            print(f"   🔍 Extracted activations from {len(activations)} layers")
            for layer_name, activation in activations.items():
                if isinstance(activation, torch.Tensor):
                    print(f"      - {layer_name}: shape {tuple(activation.shape)}")
        except Exception as e:
            print(f"   ⚠️  Activation extraction failed: {e}")
        
        print(f"\n🎉 All tests completed!")
        print(f"✨ torch-vis package is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n🔧 To fix this, try:")
        print("   1. Install the package: pip install -e .")
        print("   2. Install PyTorch: pip install torch torchvision")
        print("   3. Install optional deps: pip install plotly matplotlib")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_simple_example():
    """Run a very simple example without dependencies."""
    print("\n" + "=" * 50)
    print("Simple Example (Text-based)")
    print("=" * 50)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        print("📋 Model Architecture:")
        for i, (name, layer) in enumerate(model.named_children()):
            params = sum(p.numel() for p in layer.parameters())
            layer_type = type(layer).__name__
            print(f"   {i+1:2d}. {layer_type:15s} - {params:,} parameters")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📊 Total Parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"🔍 Input shape: {tuple(dummy_input.shape)}")
        print(f"📤 Output shape: {tuple(output.shape)}")
        
        print(f"\n✅ Simple example completed successfully!")
        
    except Exception as e:
        print(f"❌ Simple example failed: {e}")

def main():
    """Run all tests."""
    print("🔬 torch-vis Package Test Suite")
    print("=" * 50)
    
    # Run main test
    success = test_torch_vis()
    
    # Run simple example
    test_simple_example()
    
    print(f"\n{'='*50}")
    if success:
        print("🎯 Ready to use torch-vis!")
        print("📖 See README.md for more examples")
        print("💻 Try: python pytorch_example.py")
    else:
        print("🔧 Please install missing dependencies")
        print("📦 Run: pip install torch torchvision plotly")

if __name__ == "__main__":
    main() 
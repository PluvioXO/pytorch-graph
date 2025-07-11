#!/usr/bin/env python3
"""
Test package structure without requiring PyTorch installation.
"""

import sys
import os

def test_package_structure():
    """Test that the package structure is correct."""
    print("Testing torch-vis package structure...")
    print("=" * 50)
    
    # Test 1: Check if torch_vis directory exists
    print("1. Checking package directory...")
    if os.path.exists("torch_vis"):
        print("   ✅ torch_vis/ directory found")
    else:
        print("   ❌ torch_vis/ directory not found")
        return False
    
    # Test 2: Check core files
    print("\n2. Checking core files...")
    required_files = [
        "torch_vis/__init__.py",
        "torch_vis/core/parser.py", 
        "torch_vis/core/visualizer.py",
        "torch_vis/utils/layer_info.py",
        "torch_vis/utils/position_calculator.py",
        "torch_vis/utils/pytorch_hooks.py",
        "torch_vis/utils/model_analyzer.py",
        "torch_vis/renderers/plotly_renderer.py"
    ]
    
    all_found = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_found = False
    
    # Test 3: Try to import the package
    print("\n3. Testing package import...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        import torch_vis
        print("   ✅ torch_vis imported successfully")
        print(f"   📦 Version: {torch_vis.__version__}")
        print(f"   👥 Author: {torch_vis.__author__}")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        all_found = False
    except Exception as e:
        print(f"   ⚠️  Import warning: {e}")
    
    # Test 4: Check if main functions are available
    print("\n4. Checking available functions...")
    try:
        functions_to_check = [
            'visualize',
            'analyze_model', 
            'profile_model',
            'extract_activations',
            'compare_models',
            'create_architecture_report'
        ]
        
        for func_name in functions_to_check:
            if hasattr(torch_vis, func_name):
                print(f"   ✅ {func_name}()")
            else:
                print(f"   ❌ {func_name}() - MISSING")
                all_found = False
                
    except Exception as e:
        print(f"   ⚠️  Function check failed: {e}")
    
    # Test 5: Check setup.py
    print("\n5. Checking setup.py...")
    if os.path.exists("setup.py"):
        print("   ✅ setup.py found")
        try:
            with open("setup.py", "r") as f:
                content = f.read()
                if "torch-vis" in content:
                    print("   ✅ Package name 'torch-vis' found in setup.py")
                else:
                    print("   ❌ Package name 'torch-vis' not found in setup.py")
                    all_found = False
        except Exception as e:
            print(f"   ⚠️  Could not read setup.py: {e}")
    else:
        print("   ❌ setup.py not found")
        all_found = False
    
    return all_found

def test_installation():
    """Test if package can be installed."""
    print("\nTesting installation...")
    print("=" * 30)
    
    print("To install the package, run:")
    print("   pip install -e .")
    print("\nTo install with PyTorch:")
    print("   pip install torch torchvision")
    print("\nTo install optional dependencies:")
    print("   pip install plotly matplotlib networkx pandas")

def show_usage_example():
    """Show usage example."""
    print("\nUsage Example (once PyTorch is installed):")
    print("=" * 45)
    print("""
import torch
import torch.nn as nn
import torch_vis

# Create a simple model
model = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 15 * 15, 10)
)

# Visualize the model
fig = torch_vis.visualize(
    model, 
    input_shape=(3, 32, 32),
    title="My Model"
)
fig.show()

# Analyze the model
analysis = torch_vis.analyze_model(model, input_shape=(3, 32, 32))
print(f"Parameters: {analysis['basic_info']['total_parameters']:,}")
""")

def main():
    """Run structure tests."""
    print("🔬 torch-vis Package Structure Test")
    print("=" * 50)
    
    success = test_package_structure()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ Package structure looks good!")
        print("🎯 Ready for installation and use")
    else:
        print("❌ Package structure has issues")
        print("🔧 Please check missing files")
    
    test_installation()
    show_usage_example()
    
    print(f"\n📁 Current directory contents:")
    for item in sorted(os.listdir(".")):
        if os.path.isdir(item):
            print(f"   📂 {item}/")
        else:
            print(f"   📄 {item}")

if __name__ == "__main__":
    main() 
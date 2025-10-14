#!/usr/bin/env python
"""
Test script to verify image-as-brush installation
Run this after installing with: pip install -e .
"""

def test_import():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        import image_as_brush
        print(f"✓ image_as_brush imported successfully")
        print(f"  Version: {image_as_brush.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import image_as_brush: {e}")
        return False


def test_core_modules():
    """Test core module imports"""
    print("\nTesting core module imports...")
    success = True
    
    modules = [
        ('SimpleTileRenderer', 'from image_as_brush import SimpleTileRenderer'),
        ('VectorRenderer', 'from image_as_brush import VectorRenderer'),
        ('StructureAwareInitializer', 'from image_as_brush import StructureAwareInitializer'),
        ('RandomInitializer', 'from image_as_brush import RandomInitializer'),
        ('PrimitiveLoader', 'from image_as_brush import PrimitiveLoader'),
        ('SVGLoader', 'from image_as_brush import SVGLoader'),
        ('PSDExporter', 'from image_as_brush import PSDExporter'),
        ('Preprocessor', 'from image_as_brush import Preprocessor'),
    ]
    
    for name, import_stmt in modules:
        try:
            exec(import_stmt)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            success = False
    
    return success


def test_cuda_extensions():
    """Test if CUDA extensions are available"""
    print("\nTesting CUDA extensions...")
    try:
        import image_as_brush
        if image_as_brush.check_cuda_available():
            print("  ✓ CUDA extensions compiled and available")
            return True
        else:
            print("  ⚠ CUDA extensions not available")
            print("    Build them with: cd cuda_tile_rasterizer && python setup.py build_ext --inplace")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    print("\nTesting PyTorch CUDA...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def test_simple_usage():
    """Test basic usage with minimal example"""
    print("\nTesting basic usage...")
    try:
        import torch
        from image_as_brush import SimpleTileRenderer, RandomInitializer
        
        # Create dummy primitive
        S = torch.ones(1, 64, 64)  # Dummy primitive template
        
        # Initialize renderer
        renderer = SimpleTileRenderer(
            canvas_size=(128, 128),
            S=S,
            device='cpu',  # Use CPU for testing
            tile_size=32,
        )
        
        # Initialize parameters with random initializer
        init_conf = {'N': 10, 'initializer': 'random'}
        initializer = RandomInitializer(init_conf)
        
        target = torch.rand(128, 128, 3)
        x, y, r, v, theta, c = renderer.initialize_parameters(initializer, target, None)
        
        # Try rendering
        output = renderer.render_from_params(x, y, r, theta, v, c)
        
        print(f"  ✓ Successfully created renderer and rendered {x.shape[0]} primitives")
        print(f"    Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"  ✗ Basic usage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("image-as-brush Installation Test")
    print("="*60)
    
    results = {
        'Basic Import': test_import(),
        'Core Modules': test_core_modules(),
        'CUDA Extensions': test_cuda_extensions(),
        'PyTorch CUDA': test_torch_cuda(),
        'Basic Usage': test_simple_usage(),
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Installation successful.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)

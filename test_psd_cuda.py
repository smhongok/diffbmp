#!/usr/bin/env python3
"""
Test script for CUDA PSD export implementation
"""

import torch
import numpy as np
import time
import os
import sys

# Add the project root to path
sys.path.insert(0, '/home/sonic/ICL_SMH/Research_compass_aftersubmit/circle_art')

def test_cuda_psd_export():
    """Test the CUDA PSD export implementation with sample data"""
    
    print("🧪 Testing CUDA PSD Export Implementation")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping test")
        return False
    
    device = torch.device('cuda')
    print(f"✅ Using device: {device}")
    
    # Create sample data
    N = 10  # Number of primitives
    H, W = 512, 512  # Canvas size
    template_size = 64  # Template size
    
    print(f"📊 Test parameters: {N} primitives, {H}x{W} canvas, {template_size}x{template_size} templates")
    
    # Generate random primitive parameters
    torch.manual_seed(42)  # For reproducible results
    
    x = torch.rand(N, device=device) * W * 0.8 + W * 0.1  # Keep away from edges
    y = torch.rand(N, device=device) * H * 0.8 + H * 0.1
    r = torch.rand(N, device=device) * 50 + 20  # Radius 20-70
    theta = torch.rand(N, device=device) * 2 * np.pi  # Full rotation
    v = torch.rand(N, device=device) * 2 - 1  # Visibility logits
    c = torch.rand(N, 3, device=device) * 2 - 1  # Color logits
    
    # Create sample templates (simple circular gradients)
    templates = []
    for i in range(3):  # 3 different templates
        template = torch.zeros(template_size, template_size, device=device)
        center = template_size // 2
        y_grid, x_grid = torch.meshgrid(
            torch.arange(template_size, device=device),
            torch.arange(template_size, device=device),
            indexing='ij'
        )
        dist = torch.sqrt((x_grid - center)**2 + (y_grid - center)**2)
        template = torch.clamp(1.0 - dist / (template_size // 2), 0, 1)
        templates.append(template)
    
    primitive_templates = torch.stack(templates)
    global_bmp_sel = torch.arange(N, device=device) % 3  # Cycle through templates
    
    print("✅ Sample data generated")
    
    # Test CUDA implementation
    try:
        # Add cuda_tile_rasterizer to path for import
        import sys
        import os
        cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda_tile_rasterizer')
        if cuda_dir not in sys.path:
            sys.path.insert(0, cuda_dir)
        
        from psd_export_uint8 import export_psd_layers_cuda_uint8
        
        print("🚀 Testing CUDA implementation...")
        start_time = time.time()
        
        pil_images, bounds_list = export_psd_layers_cuda_uint8(
            primitive_templates, x, y, r, theta,
            torch.sigmoid(v), torch.sigmoid(c), global_bmp_sel.int(),
            H, W, scale_factor=1.0, alpha_upper_bound=1.0
        )
        
        cuda_time = time.time() - start_time
        print(f"✅ CUDA implementation completed in {cuda_time:.3f}s")
        print(f"   📊 {len(pil_images)} PIL images generated")
        print(f"   📊 Average time per primitive: {cuda_time/N*1000:.2f}ms")
        
        # Validate results
        non_empty_layers = 0
        total_pixels = 0
        for i, (img, bounds) in enumerate(zip(pil_images, bounds_list)):
            left, top, right, bottom = bounds
            w, h = img.size
            if w > 1 and h > 1:
                non_empty_layers += 1
                total_pixels += w * h
            print(f"   Layer {i}: {w}x{h}, bounds=({left},{top},{right},{bottom})")
        
        print(f"✅ Validation: {non_empty_layers}/{N} non-empty layers, {total_pixels} total pixels")
        
        return True
        
    except ImportError as e:
        print(f"❌ CUDA extension not available: {e}")
        return False
    except Exception as e:
        print(f"❌ CUDA implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch_fallback():
    """Test the PyTorch fallback implementation"""
    
    print("\n🧪 Testing PyTorch Fallback Implementation")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    # Create sample data (same as CUDA test)
    N = 5  # Smaller for fallback test
    H, W = 256, 256
    template_size = 32
    
    torch.manual_seed(42)
    x = torch.rand(N, device=device) * W * 0.8 + W * 0.1
    y = torch.rand(N, device=device) * H * 0.8 + H * 0.1
    r = torch.rand(N, device=device) * 30 + 10
    theta = torch.rand(N, device=device) * 2 * np.pi
    v = torch.rand(N, device=device) * 2 - 1
    c = torch.rand(N, 3, device=device) * 2 - 1
    
    # Single template for simplicity
    template = torch.zeros(template_size, template_size, device=device)
    center = template_size // 2
    y_grid, x_grid = torch.meshgrid(
        torch.arange(template_size, device=device),
        torch.arange(template_size, device=device),
        indexing='ij'
    )
    dist = torch.sqrt((x_grid - center)**2 + (y_grid - center)**2)
    template = torch.clamp(1.0 - dist / (template_size // 2), 0, 1)
    
    global_bmp_sel = torch.zeros(N, dtype=torch.int32, device=device)
    
    try:
        # Add cuda_tile_rasterizer to path for import
        import sys
        import os
        cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda_tile_rasterizer')
        if cuda_dir not in sys.path:
            sys.path.insert(0, cuda_dir)
        
        from psd_export_uint8 import export_psd_layers_pytorch_fallback
        
        print("🔄 Testing PyTorch fallback...")
        start_time = time.time()
        
        pil_images, bounds_list = export_psd_layers_pytorch_fallback(
            template, x, y, r, theta, v, c, global_bmp_sel,
            H, W, scale_factor=1.0, alpha_upper_bound=1.0
        )
        
        pytorch_time = time.time() - start_time
        print(f"✅ PyTorch fallback completed in {pytorch_time:.3f}s")
        print(f"   📊 {len(pil_images)} PIL images generated")
        print(f"   📊 Average time per primitive: {pytorch_time/N*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch fallback failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_cuda_extension():
    """Check if CUDA extension is available (assumes user built it manually)"""
    
    print("\n🔧 Checking CUDA Extension Availability")
    print("=" * 50)
    
    try:
        # Add cuda_tile_rasterizer to path for import
        import sys
        import os
        cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda_tile_rasterizer')
        if cuda_dir not in sys.path:
            sys.path.insert(0, cuda_dir)
        
        import psd_export_cuda
        print("✅ CUDA extension is available")
        return True
    except ImportError as e:
        print(f"❌ CUDA extension not found: {e}")
        print("💡 Please build the extension manually with:")
        print("   cd cuda_tile_rasterizer")
        print("   TORCH_CUDA_ARCH_LIST=\"8.6\" CUDA_HOME=/usr/local/cuda-12.1 python setup_psd_export.py build_ext --inplace")
        return False

if __name__ == "__main__":
    print("🚀 CUDA PSD Export Test Suite")
    print("=" * 60)
    
    # Check if CUDA extension is available (user builds manually)
    extension_available = check_cuda_extension()
    
    if extension_available:
        # Test CUDA implementation
        cuda_success = test_cuda_psd_export()
    else:
        cuda_success = False
    
    # Test PyTorch fallback
    pytorch_success = test_pytorch_fallback()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"🔧 Extension Available: {'✅ PASS' if extension_available else '❌ FAIL'}")
    print(f"🚀 CUDA Implementation: {'✅ PASS' if cuda_success else '❌ FAIL'}")
    print(f"🔄 PyTorch Fallback: {'✅ PASS' if pytorch_success else '❌ FAIL'}")
    
    if cuda_success:
        print("\n🎉 CUDA PSD export is ready for use!")
    elif pytorch_success:
        print("\n⚠️  CUDA not available, but PyTorch fallback works")
    else:
        print("\n❌ Both implementations failed - check setup")

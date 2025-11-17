#!/usr/bin/env python3
"""
Test script to validate CUDA forward kernel correctness by comparing
CUDA and PyTorch tile renderer outputs pixel by pixel.

Usage:
1. Test with synthetic data:
   python test_cuda_forward.py

2. Test with trained parameters from main.py:
   python main.py --config configs/default.json  # First, train and save parameters
   python test_cuda_forward.py --use-trained     # Then, test with trained parameters
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pydiffbmp.core.renderer.simple_tile_renderer import SimpleTileRenderer
from pydiffbmp.core.renderer.vector_renderer import VectorRenderer
import json
import os

def load_test_config():
    """Load a simple test configuration"""
    config_path = "configs/default_tile.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        # Fallback simple config
        return {
            "canvas_size": [169, 256],
            "tile_size": 32,
            "num_primitives": 100
        }

def load_trained_parameters(device='cuda'):
    """Load trained parameters from main.py training output"""
    # Path to saved parameters (adjust as needed)
    param_path = "trained_parameters.pt"
    
    if not os.path.exists(param_path):
        print(f"❌ Trained parameters file not found: {param_path}")
        print("💡 Please run 'python main.py --config configs/default.json' first to generate parameters")
        return None
    
    try:
        # Load trained parameters
        params = torch.load(param_path, map_location=device)
        print(f"✅ Loaded trained parameters from {param_path}")
        
        # Extract parameters
        x = params['x'].to(device)
        y = params['y'].to(device)
        r = params['r'].to(device)
        theta = params['theta'].to(device)
        v = params['v'].to(device)
        c = params['c'].to(device)
        S = params['S'].to(device)
        
        # Get canvas size from parameters
        H, W = params.get('canvas_size', [256, 256])
        
        print(f"📊 Loaded parameters:")
        print(f"  - Number of primitives: {len(x)}")
        print(f"  - Canvas size: {H}x{W}")
        print(f"  - Template shape: {S.shape}")
        print(f"  - Parameter ranges:")
        print(f"    x: [{x.min():.4f}, {x.max():.4f}]")
        print(f"    y: [{y.min():.4f}, {y.max():.4f}]")
        print(f"    r: [{r.min():.4f}, {r.max():.4f}]")
        print(f"    v: [{v.min():.4f}, {v.max():.4f}]")
        print(f"    c: [{c.min():.4f}, {c.max():.4f}]")
        
        return x, y, r, theta, v, c, S, H, W
        
    except Exception as e:
        print(f"❌ Failed to load trained parameters: {e}")
        return None

def save_trained_parameters(x, y, r, theta, v, c, S, canvas_size):
    """Save trained parameters for later testing"""
    params = {
        'x': x,
        'y': y,
        'r': r,
        'theta': theta,
        'v': v,
        'c': c,
        'S': S,
        'canvas_size': canvas_size
    }
    
    torch.save(params, "trained_parameters.pt")
    print(f"✅ Saved trained parameters to trained_parameters.pt")

def create_simple_test_data(device='cuda'):
    """Create simple test data for comparison"""
    N = 2000  # Larger number for better CUDA performance
    H, W = 256, 256  # Larger image size
    
    # Create simple primitive parameters
    x = torch.rand(N, device=device) * W * 0.8 + W * 0.1  # Center in image
    y = torch.rand(N, device=device) * H * 0.8 + H * 0.1
    r = torch.rand(N, device=device) * 45 + 5  # Radius 5-50
    theta = torch.rand(N, device=device) * 2 * np.pi  # Random rotation
    v = torch.randn(N, device=device) * 2  # Opacity logits
    c = torch.randn(N, 3, device=device) * 2  # Color logits
    
    # Create simple primitive templates (circles)
    template_size = 32
    S = torch.zeros(N, template_size, template_size, device=device)
    center = template_size // 2
    for i in range(N):
        y_grid, x_grid = torch.meshgrid(
            torch.arange(template_size, device=device),
            torch.arange(template_size, device=device),
            indexing='ij'
        )
        dist = torch.sqrt((x_grid - center)**2 + (y_grid - center)**2)
        S[i] = torch.exp(-(dist**2) / (2 * (template_size/6)**2))  # Gaussian circle
    
    return x, y, r, theta, v, c, S, H, W

def test_cuda_vs_pytorch(use_trained_params=False):
    """Compare CUDA SimpleTileRenderer vs PyTorch fallback SimpleTileRenderer"""
    print("🔍 Testing CUDA Forward Kernel Correctness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    if use_trained_params:
        print("📂 Using trained parameters from main.py...")
        result = load_trained_parameters(device)
        if result is None:
            print("❌ Failed to load trained parameters, falling back to synthetic data")
            x, y, r, theta, v, c, S, H, W = create_simple_test_data(device)
        else:
            x, y, r, theta, v, c, S, H, W = result
    else:
        print("🧪 Using synthetic test data...")
        x, y, r, theta, v, c, S, H, W = create_simple_test_data(device)
    
    print(f"Test data: {len(x)} primitives, {H}x{W} image")
    
    # Create two SimpleTileRenderer instances
    cuda_renderer = SimpleTileRenderer(canvas_size=(H, W), S=S, tile_size=32, alpha_upper_bound=1.0, device=device)
    pytorch_renderer = SimpleTileRenderer(canvas_size=(H, W), S=S, tile_size=32, alpha_upper_bound=1.0, device=device)

    lr_conf = {
        'default': 0.1,
        'gain_x': 1.0,
        'gain_y': 1.0,
        'gain_r': 1.0,
        'gain_v': 1.0,
        'gain_theta': 1.0,
        'gain_c': 1.0
    }
    
    print("\n🚀 Rendering with CUDA SimpleTileRenderer...")
    try:
        # Force CUDA usage by ensuring CUDA_AVAILABLE is True
        import core.renderer.simple_tile_renderer as str_module
        original_cuda_available = str_module.CUDA_AVAILABLE
        str_module.CUDA_AVAILABLE = True

        white_bg = torch.ones((H, W, 3), device=device)
        
        # Warm up CUDA
        print("🔥 Warming up CUDA...")
        for _ in range(3):
            _ = cuda_renderer.render_from_params(x, y, r, theta, v, c, sigma=0.0, I_bg=white_bg, lr_conf=lr_conf)
        
        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time CUDA rendering
        print("⏱️ Timing CUDA rendering...")
        
        # Check if CUDA kernel is actually being called
        print("🔍 Checking CUDA kernel usage...")
        
        # Add debug info to see what's happening
        import core.renderer.simple_tile_renderer as str_module
        print(f"  CUDA_AVAILABLE: {str_module.CUDA_AVAILABLE}")
        print(f"  Device: {device}")
        print(f"  Data size: {len(x)} primitives, {H}x{W} image")
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        cuda_result = cuda_renderer.render_from_params(x, y, r, theta, v, c, sigma=0.0, I_bg=white_bg, lr_conf=lr_conf)
        end_time.record()
        
        # Wait for GPU to finish
        torch.cuda.synchronize()
        cuda_time = start_time.elapsed_time(end_time)  # milliseconds
        
        print(f"CUDA result shape: {cuda_result.shape}, dtype: {cuda_result.dtype}")
        print(f"CUDA result range: [{cuda_result.min():.4f}, {cuda_result.max():.4f}]")
        print(f"⏱️ CUDA rendering time: {cuda_time:.2f} ms")
        
    except Exception as e:
        print(f"❌ CUDA rendering failed: {e}")
        return False
    
    print("\n🐍 Rendering with PyTorch fallback SimpleTileRenderer...")
    try:
        # Force PyTorch fallback by temporarily disabling CUDA
        str_module.CUDA_AVAILABLE = False
        
        # Warm up PyTorch
        print("🔥 Warming up PyTorch...")
        for _ in range(3):
            _ = pytorch_renderer.render_from_params(x, y, r, theta, v, c, sigma=0.0, I_bg=white_bg, lr_conf=lr_conf)
        
        # Synchronize if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time PyTorch rendering
        print("⏱️ Timing PyTorch rendering...")
        import time
        
        start_time = time.time()
        pytorch_result = pytorch_renderer.render_from_params(x, y, r, theta, v, c, sigma=0.0, I_bg=white_bg, lr_conf=lr_conf)
        
        # Synchronize if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        pytorch_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        print(f"PyTorch result shape: {pytorch_result.shape}, dtype: {pytorch_result.dtype}")
        print(f"PyTorch result range: [{pytorch_result.min():.4f}, {pytorch_result.max():.4f}]")
        print(f"⏱️ PyTorch rendering time: {pytorch_time:.2f} ms")
        
        # Restore original CUDA availability
        str_module.CUDA_AVAILABLE = original_cuda_available
        
    except Exception as e:
        print(f"❌ PyTorch rendering failed: {e}")
        return False
    
    # Compare results
    print("\n📊 Comparing results...")
    
    # Ensure same shape and dtype
    if cuda_result.shape != pytorch_result.shape:
        print(f"❌ Shape mismatch: CUDA {cuda_result.shape} vs PyTorch {pytorch_result.shape}")
        return False
    
    # Convert to same dtype for comparison
    cuda_result = cuda_result.float()
    pytorch_result = pytorch_result.float()
    
    # Calculate differences
    diff = torch.abs(cuda_result - pytorch_result)
    mse = torch.mean(diff**2).item()
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"📈 Comparison metrics:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Max absolute difference: {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")
    
    # Performance comparison
    print(f"\n🚀 Performance comparison:")
    print(f"  CUDA time: {cuda_time:.2f} ms")
    print(f"  PyTorch time: {pytorch_time:.2f} ms")
    if pytorch_time > 0:
        speedup = pytorch_time / cuda_time
        print(f"  Speedup: {speedup:.2f}x faster with CUDA")
        print(f"  Time savings: {pytorch_time - cuda_time:.2f} ms ({((pytorch_time - cuda_time) / pytorch_time * 100):.1f}% faster)")
        
        # Additional performance metrics
        print(f"\n📊 Detailed performance metrics:")
        total_pixels = H * W * 3
        cuda_throughput = total_pixels / (cuda_time / 1000)  # pixels per second
        pytorch_throughput = total_pixels / (pytorch_time / 1000)  # pixels per second
        print(f"  CUDA throughput: {cuda_throughput/1e6:.1f} M pixels/sec")
        print(f"  PyTorch throughput: {pytorch_throughput/1e6:.1f} M pixels/sec")
    
    # Determine if results are similar enough
    tolerance = 1e-4  # Reasonable tolerance for floating point
    is_similar = mse < tolerance and max_diff < tolerance * 10
    
    if is_similar:
        print(f"✅ CUDA and PyTorch results are similar (within tolerance {tolerance})")
    else:
        print(f"❌ CUDA and PyTorch results differ significantly!")
        
        # Show some sample differences
        print(f"\nSample pixel differences:")
        flat_diff = diff.view(-1)
        top_diff_indices = torch.topk(flat_diff, 5).indices
        for i, idx in enumerate(top_diff_indices):
            h_idx = idx // (W * 3)
            w_idx = (idx % (W * 3)) // 3
            c_idx = idx % 3
            print(f"  Pixel ({h_idx}, {w_idx}, {c_idx}): CUDA={cuda_result.view(-1)[idx]:.6f}, PyTorch={pytorch_result.view(-1)[idx]:.6f}, Diff={flat_diff[idx]:.6f}")
    
    # Save comparison images
    print(f"\n💾 Saving comparison images...")
    try:
        # Convert to numpy for visualization
        cuda_np = cuda_result.detach().cpu().numpy()
        pytorch_np = pytorch_result.detach().cpu().numpy()
        diff_np = diff.detach().cpu().numpy()
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # CUDA result
        axes[0, 0].imshow(cuda_np)
        axes[0, 0].set_title('CUDA Result')
        axes[0, 0].axis('off')
        
        # PyTorch result
        axes[0, 1].imshow(pytorch_np)
        axes[0, 1].set_title('PyTorch Result')
        axes[0, 1].axis('off')
        
        # Difference
        axes[0, 2].imshow(diff_np, cmap='hot')
        axes[0, 2].set_title(f'Absolute Difference (Max: {max_diff:.6f})')
        axes[0, 2].axis('off')
        
        # Histograms
        axes[1, 0].hist(cuda_np.flatten(), bins=50, alpha=0.7, label='CUDA')
        axes[1, 0].set_title('CUDA Pixel Value Distribution')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Count')
        
        axes[1, 1].hist(pytorch_np.flatten(), bins=50, alpha=0.7, label='PyTorch', color='orange')
        axes[1, 1].set_title('PyTorch Pixel Value Distribution')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Count')
        
        axes[1, 2].hist(diff_np.flatten(), bins=50, alpha=0.7, label='Difference', color='red')
        axes[1, 2].set_title('Difference Distribution')
        axes[1, 2].set_xlabel('Absolute Difference')
        axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('cuda_pytorch_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✅ Comparison saved to: cuda_pytorch_comparison.png")
        
    except Exception as e:
        print(f"⚠️ Failed to save comparison images: {e}")
    
    return is_similar

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CUDA forward kernel with trained parameters")
    parser.add_argument('--use-trained', action='store_true', 
                       help='Use trained parameters from main.py instead of synthetic data')
    args = parser.parse_args()
    
    print("🧪 CUDA Forward Kernel Validation Test")
    print("=" * 50)
    
    if args.use_trained:
        print("🎯 Using trained parameters from main.py training output")
        print("💡 Make sure to run 'python main.py --config configs/default.json' first")
    else:
        print("🧪 Using synthetic test data")
    
    success = test_cuda_vs_pytorch(use_trained_params=args.use_trained)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 CUDA forward kernel validation PASSED!")
        print("✅ CUDA and PyTorch produce similar results")
    else:
        print("❌ CUDA forward kernel validation FAILED!")
        print("🚨 CUDA and PyTorch produce different results")
    
    print("Test completed.")

#!/usr/bin/env python3
"""
Test script for gradient visualization functionality.
This demonstrates how to use the new gradient visualization methods.
"""

import torch
import numpy as np
import os
from core.renderer.sequential_renderer import SequentialFrameRenderer

def create_test_data():
    """Create sample primitive parameters and target image for testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample primitives
    N = 100  # Number of primitives
    H, W = 256, 256  # Image dimensions
    
    # Sample primitive parameters
    x = torch.rand(N, device=device) * W
    y = torch.rand(N, device=device) * H
    r = torch.rand(N, device=device) * 10 + 5  # radius 5-15
    v = torch.randn(N, device=device)  # visibility (logits)
    theta = torch.rand(N, device=device) * 2 * np.pi
    c = torch.rand(N, 3, device=device)  # RGB colors
    
    # Create a simple target image (gradient pattern)
    target = torch.zeros(H, W, 3, device=device)
    for i in range(H):
        for j in range(W):
            target[i, j, 0] = i / H  # Red gradient vertically
            target[i, j, 1] = j / W  # Green gradient horizontally
            target[i, j, 2] = 0.5   # Blue constant
    
    return x, y, r, v, theta, c, target

def test_gradient_visualization():
    """Test the gradient visualization functionality."""
    print("Testing gradient visualization functionality...")
    
    # Create test data
    x, y, r, v, theta, c, target = create_test_data()
    H, W = target.shape[:2]
    
    # Create renderer with correct parameters
    canvas_size = (H, W)
    S = torch.eye(3, device=x.device)  # Identity transformation matrix
    renderer = SequentialFrameRenderer(
        canvas_size=canvas_size, 
        S=S, 
        alpha_upper_bound=0.5, 
        device=x.device, 
        use_fp16=False
    )
    
    # Simulate some selected indices (primitives that would be identified as problematic)
    selected_indices = torch.tensor([5, 15, 25, 35, 45], device=x.device)
    
    print(f"Testing with {len(selected_indices)} selected primitives: {selected_indices.tolist()}")
    
    # Test basic gradient visualization
    print("\n1. Testing basic gradient visualization...")
    vis_result = renderer.visualize_gradient_directions_for_selected_primitives(
        x, y, r, v, theta, c, target, selected_indices,
        save_path="./test_gradient_basic.png"
    )
    print(f"Basic visualization result shape: {vis_result.shape}")
    
    # Test debug adaptive control with visualization
    print("\n2. Testing debug adaptive control with visualization...")
    
    # Create adaptive control config
    adaptive_config = {
        'enabled': True,
        'tile_rows': 4,
        'tile_cols': 4,
        'scale_threshold': 8.0,
        'opacity_threshold': 0.7,
        'opacity_reduction_factor': 0.5,
        'max_primitives_per_tile': 3,
        'gradient_criterion': {
            'enabled': True,
            'threshold_percentile': 0.7
        },
        'gradient_ranking': {
            'pixels_per_tile': 16
        }
    }
    
    # Run debug adaptive control
    adapted_params, tile_info = renderer.debug_adaptive_control_with_visualization(
        x, y, r, v, theta, c, target, adaptive_config,
        save_dir="./debug_gradients_test"
    )
    
    print(f"Adaptive control completed. Processed {len(tile_info)} tiles with selected primitives.")
    for tile in tile_info:
        print(f"  Tile ({tile['row']},{tile['col']}): {len(tile['selected_indices'])} selected from {tile['total_primitives']} total")
    
    print("\nTest completed successfully!")
    print("Check the following files for visualizations:")
    print("- test_gradient_basic.png")
    print("- debug_gradients_test/ directory")

if __name__ == "__main__":
    test_gradient_visualization()

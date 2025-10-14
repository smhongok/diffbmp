import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import time
from datetime import timedelta
from .base_initializer import BaseInitializer
from imbrush.core.renderer.vector_renderer import VectorRenderer
from typing import Dict, Any
from imbrush.util.constants import OPACITY_INIT_VALUE, STD_C_INIT, VARIANCE_WINDOW_SIZE, VARIANCE_BASE_PROB
# Record the start time

class StructureAwareInitializer(BaseInitializer):
    def __init__(self, init_opt:Dict[str, Any]):
        super().__init__(init_opt)
        # Get initialization parameters from config (with constants as defaults)
        self.std_c_init = init_opt.get("std_c_init", STD_C_INIT)
        self.variance_window_size = init_opt.get("variance_window_size", VARIANCE_WINDOW_SIZE)
        self.variance_base_prob = init_opt.get("variance_base_prob", VARIANCE_BASE_PROB)

    def initialize(self, I_target, target_binary_mask = None, I_bg=None, renderer:VectorRenderer=None, opt_conf:Dict[str, Any]=None, return_pts: bool = False):
        """
        Specialization for SVG input to match the API expected in main_svg.py
        """
        start_time = time.time()
        device = I_target.device
        # Extract image dimensions
        if I_target.ndim == 3:
            H, W, _ = I_target.shape
            I_color = I_target.detach().cpu().numpy()            # (H,W,3), 0~1
        else:
            H, W = I_target.shape
            gray = np.expand_dims(I_np / 255.0, axis=-1)
            I_color = np.repeat(gray, 3, axis=-1)
            
        N = self.num_init

        # For SVG initialization, we'll use the image structure to guide placement
        if isinstance(I_target, torch.Tensor):
            I_np = I_target.detach().cpu().numpy()
            # If it's a color image, convert to grayscale for structure analysis
            if I_np.ndim == 3:
                I_np = np.mean(I_np, axis=2)
        else:
            I_np = I_target
            if I_np.ndim == 3:
                I_np = cv2.cvtColor(I_np, cv2.COLOR_RGB2GRAY)
                
        # Ensure image is in correct format for ORB
        if I_np.dtype != np.uint8:
            I_np = (I_np * 255).astype(np.uint8)        

        # -------------------- RGB variance-based importance map -------------------- #
        # Compute local variance for each RGB channel
        window_size = self.variance_window_size
        half_window = window_size // 2
        
        # Ensure I_color is in correct format (H, W, 3) with values 0-1
        if I_color.max() > 1.0:
            I_color_normalized = I_color / 255.0
        else:
            I_color_normalized = I_color
        
        # Compute local variance for each channel
        variance_map = np.zeros((H, W), dtype=np.float32)
        
        for channel in range(3):  # R, G, B
            channel_img = I_color_normalized[:, :, channel]
            
            # Use cv2.boxFilter for efficient local mean computation
            local_mean = cv2.boxFilter(channel_img, -1, (window_size, window_size), normalize=True)
            local_mean_sq = cv2.boxFilter(channel_img**2, -1, (window_size, window_size), normalize=True)
            
            # Variance = E[X^2] - E[X]^2
            channel_variance = local_mean_sq - local_mean**2
            variance_map += channel_variance
        
        # Normalize variance map to [0, 1]
        if variance_map.max() > 0:
            variance_map = variance_map / variance_map.max()
        
        print(f"Variance map stats: min={variance_map.min():.4f}, max={variance_map.max():.4f}, mean={variance_map.mean():.4f}")
        
        # -------------------- Stratified sampling based on variance -------------------- #
        # High variance (complex) → more points, small radius, high index
        # Low variance (flat) → fewer points, large radius, low index
        
        # Apply background mask if provided
        valid_mask = np.ones((H, W), dtype=np.float32)
        if target_binary_mask is not None:
            if hasattr(target_binary_mask, 'detach'):
                mask_np = target_binary_mask.detach().cpu().numpy()
            else:
                mask_np = target_binary_mask
            valid_mask = 1 - mask_np  # 1 = foreground, 0 = background
        
        # Create sampling probability map: higher variance = higher probability
        # But also ensure some coverage in low-variance areas
        sampling_prob = variance_map * valid_mask
        
        # Add base probability to ensure low-variance areas get some points
        base_prob = self.variance_base_prob
        sampling_prob = base_prob + (1 - base_prob) * sampling_prob
        sampling_prob = sampling_prob * valid_mask
        
        # Flatten and normalize
        sampling_prob_flat = sampling_prob.flatten()
        if sampling_prob_flat.sum() > 0:
            sampling_prob_flat = sampling_prob_flat / sampling_prob_flat.sum()
        else:
            print("Warning: No valid sampling probabilities. Using uniform distribution.")
            sampling_prob_flat = valid_mask.flatten()
            sampling_prob_flat = sampling_prob_flat / sampling_prob_flat.sum()
        
        # Sample N points based on variance probability
        all_coords = np.array([(y, x) for y in range(H) for x in range(W)])
        sampled_indices = np.random.choice(len(all_coords), size=N, replace=False, p=sampling_prob_flat)
        sampled_coords = all_coords[sampled_indices]
        
        # Get variance values at sampled points for sorting
        sampled_variances = variance_map[sampled_coords[:, 0], sampled_coords[:, 1]]
        
        # Sort by variance based on detail_first setting
        if self.detail_first:
            # Detail-first (East Asian style): low variance first (background), high variance last (foreground)
            sort_indices = np.argsort(sampled_variances)
        else:
            # Background-first (Western style): high variance first (foreground), low variance last (background)
            sort_indices = np.argsort(sampled_variances)[::-1]
        
        sampled_coords = sampled_coords[sort_indices]
        sampled_variances = sampled_variances[sort_indices]
        
        # Convert from (y, x) to (x, y) format
        adjusted_pts = sampled_coords[:, [1, 0]].astype(np.float32)
        
        print(f"Initialized {len(adjusted_pts)} points (variance-based stratified sampling)")
        print(f"Variance distribution: low={sampled_variances.min():.4f}, high={sampled_variances.max():.4f}")
        print(f"Rendering order: {'Detail-first (East Asian style)' if self.detail_first else 'Background-first (Western style)'}")
        
        # Visualize initialization
        self.visualize_initialization_points(I_np, variance_map, adjusted_pts)

        # Color initialization
        # Sample pixel colors at splat coordinates
        idx_x = np.clip(np.round(adjusted_pts[:, 0]).astype(int), 0, W - 1)
        idx_y = np.clip(np.round(adjusted_pts[:, 1]).astype(int), 0, H - 1)
        c_init = I_color[idx_y, idx_x]                  # (N,3) float32

        # Add slight noise to diversify parameters
        c_init += np.random.normal(0.0, self.std_c_init, c_init.shape)
        c_init = np.clip(c_init, 0.0, 1.0)              # Safely clip values
        
        # -------------------- Variance-based radius initialization -------------------- #
        num_points = adjusted_pts.shape[0]
        
        min_radius = self.radii_min
        if self.radii_max is not None:
            max_radius = self.radii_max
        else:
            max_radius = max(min(H, W) / 4, min_radius + 0.1)
        
        # Radius based on variance at each point
        # Low variance (flat, index 0) → large radius
        # High variance (complex, index N-1) → small radius
        # sampled_variances is already sorted from low to high
        
        # Map variance [0, 1] to radius [max, min]
        # variance 0 → max_radius, variance 1 → min_radius
        r_np = max_radius - sampled_variances * (max_radius - min_radius)
        
        # Add noise for variety
        noise_scale = 0.1 * (max_radius - min_radius)
        noise = np.random.normal(0, noise_scale, num_points)
        r_np = np.clip(r_np + noise, min_radius, max_radius)
        
        print(f"Radius distribution: min={r_np.min():.2f}, max={r_np.max():.2f}, mean={r_np.mean():.2f}")

        # -------------------- Convert to tensors -------------------- #
        # Leave FP32 even if renderer is in FP16 mode. Parse in cuda
        #dtype = torch.float16 if renderer and hasattr(renderer, 'use_fp16') and renderer.use_fp16 else torch.float32
        dtype = torch.float32
        
        x = torch.tensor(adjusted_pts[:, 0], dtype=dtype, device=device, requires_grad=True)
        y = torch.tensor(adjusted_pts[:, 1], dtype=dtype, device=device, requires_grad=True)
        r = torch.tensor(r_np, dtype=dtype, device=device, requires_grad=True)

        # -------------------- Initialize opacity v_i = v_0 -------- #
        v = torch.full((num_points,), OPACITY_INIT_VALUE, dtype=dtype, device=device).requires_grad_(True)
        
        # -------------------- Initialize opacity (layer consistent) -------- #
        #rank = torch.linspace(0.0, 1.0, steps=num_points, device=device)     # 0(bottom)→1(top)
        #v = (self.v_init_bias - 0.5 + self.v_init_slope * rank).clone().detach()
        #v += torch.empty_like(v).normal_(mean=0.0, std=0.05)

        theta = torch.rand(num_points, dtype=dtype, device=device, requires_grad=True) * 2 * np.pi
        c = torch.tensor(c_init, dtype=dtype, device=device, requires_grad=True)
        print("len(x): ", len(x))
        
        end_time = time.time()
        formatted_time = str(timedelta(seconds=int(end_time - start_time)))
        print(f"[initialize]total_cost_time: {formatted_time}")

        if not return_pts:
            return x, y, r, v, theta, c
        
        return x, y, r, v, theta, c, adjusted_pts
    
    def visualize_initialization_points(self, I_np, variance_map, adjusted_pts):
        """
        Visualize variance map and initialized points
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(I_np, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Variance map
        im = axes[1].imshow(variance_map, cmap='hot')
        axes[1].set_title('RGB Variance Map\n(Red=Complex, Blue=Flat)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Variance map with initialized points
        axes[2].imshow(variance_map, cmap='hot', alpha=0.7)
        if len(adjusted_pts) > 0:
            # Color points by their index (drawing order)
            colors = plt.cm.viridis(np.linspace(0, 1, len(adjusted_pts)))
            axes[2].scatter(adjusted_pts[:, 0], adjusted_pts[:, 1], 
                          c=colors, s=20, alpha=0.8, edgecolors='white', linewidths=0.5)
            # Add colorbar for drawing order
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(adjusted_pts)-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label('Drawing Order\n(0=First/Large, N=Last/Small)', rotation=270, labelpad=20)
        axes[2].set_title(f'Variance-Based Initialization\n({len(adjusted_pts)} points)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs('visualization_output', exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'visualization_output/initialization_points_{timestamp}.png', 
                   dpi=150, bbox_inches='tight')
        
        print(f"Visualization saved to: visualization_output/initialization_points_{timestamp}.png")

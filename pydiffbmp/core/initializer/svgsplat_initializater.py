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
from pydiffbmp.core.renderer.vector_renderer import VectorRenderer
from typing import Dict, Any
from pydiffbmp.util.constants import OPACITY_INIT_VALUE, STD_C_INIT, VARIANCE_WINDOW_SIZE, VARIANCE_BASE_PROB
# Record the start time

class StructureAwareInitializer(BaseInitializer):
    def __init__(self, init_opt:Dict[str, Any]):
        super().__init__(init_opt)
        # Get initialization parameters from config (with constants as defaults)
        self.std_c_init = init_opt.get("std_c_init", STD_C_INIT)
        self.variance_window_size = init_opt.get("variance_window_size", VARIANCE_WINDOW_SIZE)
        self.variance_base_prob = init_opt.get("variance_base_prob", VARIANCE_BASE_PROB)
        self.adjusted_pts = None
        self.sampled_variances = None
        self.decrease_max_radius = 0

    def _preprocess_image(self, I_target):
        """
        Specialization for SVG input to match the API expected in main_svg.py
        This function preprocesses the target image to extract necessary information
        for structure-aware initialization.
        args:
            I_target: target image tensor (H, W, 3) or (H, W)
        
        Returns:
            H, W: image height and width
            I_color: RGB color image (H, W, 3), 0~1
            I_np: grayscale image (H, W), uint8
            variance_map: RGB variance map (H, W), 0~1
        """
        # Extract image dimensions
        if I_target.ndim == 3:
            H, W, _ = I_target.shape
            I_color = I_target.detach().cpu().numpy()            # (H,W,3), 0~1
        else:
            H, W = I_target.shape
            I_np = I_target.detach().cpu().numpy() if isinstance(I_target, torch.Tensor) else I_target
            gray = np.expand_dims(I_np / 255.0, axis=-1)
            I_color = np.repeat(gray, 3, axis=-1)

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

        return H, W, I_color, I_np, variance_map

    def _compute_sampling_probability(self, variance_map, target_binary_mask, H, W, N):
        """
        This function computes sampling probabilities based on the variance map
        to guide point initialization.
        args:
            variance_map: RGB variance map (H, W), 0~1
            target_binary_mask: binary mask tensor (H, W), 1 or 0
        
        Returns:
            adjusted_pts: sampled point coordinates (N, 2), (x, y) format
            sampled_variances: variances of the sampled points (N,)
        """
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
        return adjusted_pts, sampled_variances

    def _initialize_parameters(self, adjusted_pts, sampled_variances, I_color, H, W, device, 
                              renderer=None, requires_grad=True):
        """
        This function initializes the splat parameters based on the sampled points
        and their associated variances.
        args:
            adjusted_pts: sampled point coordinates (N, 2), (x, y) format
            sampled_variances: variances of the sampled points (N,)
            I_color: RGB color image (H, W, 3), 0~1
            H, W: image height and width
            device: torch device for tensors
            renderer: VectorRenderer instance (not used here but kept for API consistency)
            requires_grad: Whether to set requires_grad=True for tensors (default: True)
        Returns:
            x, y, r, v, theta, c: initialized parameter tensors
        """
        num_points = adjusted_pts.shape[0]

        # Color initialization
        # Sample pixel colors at splat coordinates
        idx_x = np.clip(np.round(adjusted_pts[:, 0]).astype(int), 0, W - 1)
        idx_y = np.clip(np.round(adjusted_pts[:, 1]).astype(int), 0, H - 1)
        c_init = I_color[idx_y, idx_x]                  # (N,3) float32
        # Add slight noise to diversify parameters
        c_init += np.random.normal(0.0, self.std_c_init, c_init.shape)
        c_init = np.clip(c_init, 0.0, 1.0)              # Safely clip values

        # -------------------- Variance-based radius initialization -------------------- #
        min_radius = self.radii_min

        # Decrease max radius whenever do_prune function is called to focus on details
        max_radius_rate = 0.5 ** self.decrease_max_radius

        if self.radii_max is not None:
            max_radius = max(self.radii_max * max_radius_rate, min_radius + 0.1)
        else:
            max_radius = max(max_radius_rate * min(H, W) / 4, min_radius + 0.1)
        
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
        x = torch.tensor(adjusted_pts[:, 0], dtype=dtype, device=device, requires_grad=requires_grad)
        y = torch.tensor(adjusted_pts[:, 1], dtype=dtype, device=device, requires_grad=requires_grad)
        r = torch.tensor(r_np, dtype=dtype, device=device, requires_grad=requires_grad)
        v = torch.full((num_points,), OPACITY_INIT_VALUE, dtype=dtype, device=device).requires_grad_(requires_grad)

        # Initialize theta: use theta_init if specified, otherwise random
        if self.theta_init is not None:
            theta = torch.full((num_points,), self.theta_init, dtype=dtype, device=device, requires_grad=True)
        else:
            theta = (torch.rand(num_points, dtype=dtype, device=device) * 2 * np.pi).requires_grad_(True)
        c = torch.tensor(c_init, dtype=dtype, device=device, requires_grad=requires_grad)
        return x, y, r, v, theta, c
    
    def reinitialize_subset(self, num_points: int, target_image: torch.Tensor, device: torch.device) -> tuple:
        """
        Re-initialize a subset of primitives for pruning strategy.
        Uses stored adjusted_pts and sampled_variances from the last initialize() call.
        
        Args:
            num_points: Number of primitives to initialize
            target_image: Target image tensor for color sampling
            device: Device for tensor allocation
            
        Returns:
            Tuple of (x, y, r, v, theta, c) tensors for the re-initialized primitives
        """
        if self.adjusted_pts is None or self.sampled_variances is None:
            raise ValueError("Cannot reinitialize: No data from previous initialize() call")

        # 1. Sample positions and variances from stored data (randomly with replacement)
        # This is for the case of using many prunings over multiple iterations
        # In that case, we have to choose randomly again from the stored points
        num_adjusted_pts = len(self.adjusted_pts)
        sample_indices = np.random.choice(num_adjusted_pts, num_points, replace=True)
        sampled_positions = self.adjusted_pts[sample_indices]  # (num_points, 2), (x, y) format
        sampled_variances = self.sampled_variances[sample_indices]  # (num_points,)

        # 2. Prepare image data
        if target_image.ndim == 3:
            H, W, _ = target_image.shape
            I_color = target_image.detach().cpu().numpy()
        else:
            H, W = target_image.shape
            I_gray = target_image.detach().cpu().numpy()
            I_color = np.stack([I_gray] * 3, axis=-1)

        # 3. Reuse _initialize_parameters for the rest (no requires_grad for pruned primitives)
        return self._initialize_parameters(
            sampled_positions, sampled_variances, I_color, H, W, device, 
            renderer=None, requires_grad=False
        )

    def initialize(self, I_target, target_binary_mask = None, I_bg=None, renderer:VectorRenderer=None, opt_conf:Dict[str, Any]=None):
        """
        Specialization for SVG input to match the API expected in main_svg.py
        """
        start_time = time.time()
        device = I_target.device
        N = self.num_init

        # 1. Preprocessing
        H, W, I_color, I_np, variance_map = self._preprocess_image(I_target)

        # 2. Compute sampling probabilities
        adjusted_pts, sampled_variances = self._compute_sampling_probability(
            variance_map, target_binary_mask, H, W, N
        )

        # Visualize initialization points
        self.visualize_initialization_points(I_np, variance_map, adjusted_pts)

        # 3. Initialize parameters
        x, y, r, v, theta, c = self._initialize_parameters(
            adjusted_pts, sampled_variances, I_color, H, W, device, renderer
        )

        # Store for pruning strategy
        self.adjusted_pts = adjusted_pts
        self.sampled_variances = sampled_variances

        print("len(x): ", len(x))

        end_time = time.time()
        formatted_time = str(timedelta(seconds=int(end_time - start_time)))
        print(f"[initialize]total_cost_time: {formatted_time}")
        return x, y, r, v, theta, c
    
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

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
from core.renderer.vector_renderer import VectorRenderer
from typing import Dict, Any
# Record the start time

class StructureAwareInitializer(BaseInitializer):
    def __init__(self, init_opt:Dict[str, Any]):
        super().__init__(init_opt)

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

        # Now use our structure-aware initialization
        edges = cv2.Canny(I_np, 100, 200)
        grad_y, grad_x = np.gradient(I_np.astype(np.float32))
        
        # Start with ORB points
        num_kp = 0
        if self.keypoint_extracting:
            orb = cv2.ORB_create(nfeatures=N, scaleFactor=1.2, nlevels=8, edgeThreshold=15, firstLevel=0, WTA_K=2, patchSize=31, fastThreshold=20)
            keypoints = orb.detect(I_np, None)
        else:
            keypoints = False
        
        # If no keypoints found, do random initialization
        if not keypoints:
            if self.keypoint_extracting:
                print("No ORB keypoints. using coarse-to-fine initialization.")
            else:
                print("keypoint_extracting off. using random initialization.")
            init_pts = np.random.rand(num_kp, 2) * np.array([W, H])
        else:
            # Sort based on less strength and pick top N//5
            sorted_kp = sorted(keypoints, key=lambda kp: -kp.response, reverse=True)
            print("num_kp: ", num_kp)
            init_pts = np.array([kp.pt for kp in sorted_kp[:num_kp]])  # (x, y)
        
        # Apply our structure-aware techniques
        densified_pts, point_levels = self.find_best_densification(edges, N, target_binary_mask)
        adjusted_pts = self.structure_aware_adjustment(densified_pts, grad_x, grad_y, target_binary_mask)
        
        # Visualize densified_pts, point_levels, and adjusted_pts
        self.visualize_initialization_points(I_np, densified_pts, point_levels, adjusted_pts)
        
        print("len(densified_pts): ", len(densified_pts), ", len(adjusted_pts): ", len(adjusted_pts))

        # Print distribution of levels
        unique_levels, counts = np.unique(point_levels, return_counts=True)
        print("Level distribution:")
        for level, count in zip(unique_levels, counts):
            print(f"  Level {level}: {count} points")

        # Color initialization
        # Sample pixel colors at splat coordinates
        idx_x = np.clip(np.round(adjusted_pts[:, 0]).astype(int), 0, W - 1)
        idx_y = np.clip(np.round(adjusted_pts[:, 1]).astype(int), 0, H - 1)
        c_init = I_color[idx_y, idx_x]                  # (N,3) float32

        # Add slight noise to diversify parameters
        c_init += np.random.normal(0.0, 0.02, c_init.shape)
        c_init = np.clip(c_init, 0.0, 1.0)              # Safely clip values
        
        # -------------------- Radius initialization based on levels and edge distance -------------------- #
        num_points = adjusted_pts.shape[0]
        
        # Calculate max level for normalization
        max_level = np.max(point_levels) if len(point_levels) > 0 else 1
        
        # Determine radius based on level - coarser level (smaller value) gets larger radius
        # The formula: r = max_radius * (1 - level/max_level) + min_radius
        max_radius = min(H, W) / 4
        min_radius = self.radii_min
        
        # Calculate distance transform from Canny edges
        inverted_edges = cv2.bitwise_not(edges)
        distance_map = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
        
        # Sample distance at each adjusted point location
        idx_x = np.clip(np.round(adjusted_pts[:, 0]).astype(int), 0, W - 1)
        idx_y = np.clip(np.round(adjusted_pts[:, 1]).astype(int), 0, H - 1)
        point_distances = distance_map[idx_y, idx_x]
        
        # Normalize distances (0 = on edge, 1 = farthest from edge)
        max_distance = np.max(distance_map) if np.max(distance_map) > 0 else 1.0
        normalized_distances = point_distances / max_distance
        
        # Calculate radius for each point - invert the level so lower levels get larger radii
        normalized_levels = 1.0 - (point_levels / max_level)
        level_based_radius = min_radius + normalized_levels * (max_radius - min_radius)
        
        # Combine level-based radius with edge distance (farther from edge = larger radius)
        distance_factor = self.distance_factor  # Weight for distance influence (0.0 = only levels, 1.0 = only distance)
        distance_based_radius = min_radius + normalized_distances * (max_radius - min_radius)
        
        # Weighted combination of level-based and distance-based radius
        r_np = (1.0 - distance_factor) * level_based_radius + distance_factor * distance_based_radius
        
        # Add some noise for variety while preserving the coarse-to-fine relationship
        noise_scale = 0.1  # Scale of noise relative to radius range
        noise = np.random.normal(0, noise_scale * (max_radius - min_radius), num_points)
        r_np = np.clip(r_np + noise, min_radius, max_radius)
        
        # No sorting - keep the original order from the coarse-to-fine process
        # This preserves the natural ordering where coarser level points come first

        # -------------------- Convert to tensors -------------------- #
        x = torch.tensor(adjusted_pts[:, 0], dtype=torch.float32,
                         device=device, requires_grad=True)
        y = torch.tensor(adjusted_pts[:, 1], dtype=torch.float32,
                         device=device, requires_grad=True)

        r = torch.tensor(r_np, dtype=torch.float32,
                         device=device, requires_grad=True)

        # -------------------- Initialize opacity v_i = v_0 -------- #
        v = torch.full((num_points,), -2.0, device=device)
        
        # -------------------- Initialize opacity (layer consistent) -------- #
        #rank = torch.linspace(0.0, 1.0, steps=num_points, device=device)     # 0(bottom)→1(top)
        #v = (self.v_init_bias - 0.5 + self.v_init_slope * rank).clone().detach()
        #v += torch.empty_like(v).normal_(mean=0.0, std=0.05)
        v.requires_grad_(True)

        theta = torch.rand(num_points, device=device, requires_grad=True) * 2 * np.pi
        c = torch.tensor(c_init, dtype=torch.float32,
                         device=device, requires_grad=True)
        print("len(x): ", len(x))
        
        end_time = time.time()
        formatted_time = str(timedelta(seconds=int(end_time - start_time)))
        print(f"[initialize]total_cost_time: {formatted_time}")

        if not return_pts:
            return x, y, r, v, theta, c
        
        return x, y, r, v, theta, c, adjusted_pts
    
    def visualize_initialization_points(self, I_np, densified_pts, point_levels, adjusted_pts):
        """
        Visualize densified_pts, point_levels, and adjusted_pts
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(I_np, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Densified points with level coloring
        axes[1].imshow(I_np, cmap='gray', alpha=0.7)
        if len(densified_pts) > 0:
            # Color by level
            unique_levels = np.unique(point_levels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_levels)))
            
            for i, level in enumerate(unique_levels):
                mask = point_levels == level
                pts_at_level = densified_pts[mask]
                if len(pts_at_level) > 0:
                    axes[1].scatter(pts_at_level[:, 0], pts_at_level[:, 1], 
                                  c=[colors[i]], s=30, alpha=0.8, 
                                  label=f'Level {level}')
            
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].set_title(f'Densified Points by Level\n({len(densified_pts)} points)')
        axes[1].axis('off')
        
        # Adjusted points
        axes[2].imshow(I_np, cmap='gray', alpha=0.7)
        if len(adjusted_pts) > 0:
            # Color by level for adjusted points too
            unique_levels = np.unique(point_levels)
            colors = plt.cm.plasma(np.linspace(0, 1, len(unique_levels)))
            
            for i, level in enumerate(unique_levels):
                mask = point_levels == level
                pts_at_level = adjusted_pts[mask]
                if len(pts_at_level) > 0:
                    axes[2].scatter(pts_at_level[:, 0], pts_at_level[:, 1], 
                                  c=[colors[i]], s=30, alpha=0.8,
                                  label=f'Level {level}')
            
            # Draw arrows showing the adjustment
            if len(densified_pts) == len(adjusted_pts):
                for i in range(0, len(densified_pts), max(1, len(densified_pts)//50)):  # Show every nth arrow to avoid clutter
                    dx = adjusted_pts[i, 0] - densified_pts[i, 0]
                    dy = adjusted_pts[i, 1] - densified_pts[i, 1]
                    if abs(dx) > 1 or abs(dy) > 1:  # Only show significant adjustments
                        axes[2].arrow(densified_pts[i, 0], densified_pts[i, 1], dx, dy,
                                    head_width=3, head_length=2, fc='red', ec='red', alpha=0.3)
            
            axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].set_title(f'Structure-Aware Adjusted Points\n({len(adjusted_pts)} points)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs('visualization_output', exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'visualization_output/initialization_points_{timestamp}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: visualization_output/initialization_points_{timestamp}.png")

import numpy as np
import cv2
import torch
import os
import time
from datetime import timedelta
from abc import ABC, abstractmethod

class BaseInitializer(ABC):
    def __init__(self, num_init=100, alpha=0.3, min_distance=20, 
                 peak_threshold=0.5, radii_min=2, radii_max=None, 
                 v_init_bias=-5.0, v_init_slope=0.0, keypoint_extracting=False, debug_mode=False):
        self.num_init = num_init
        self.alpha = alpha
        self.min_distance = min_distance
        self.peak_threshold = peak_threshold
        self.radii_min = radii_min
        self.radii_max = radii_max
        self.v_init_bias = v_init_bias
        self.v_init_slope = v_init_slope
        self.keypoint_extracting = keypoint_extracting
        self.debug_mode = debug_mode
        
    @abstractmethod
    def initialize(self, I_target):
        pass
    
    def _rand_leaf(self, shape, low, high, device):
        t = torch.empty(shape, device=device).uniform_(low, high)
        return t.requires_grad_(True)   # leaf tensor 
    
    def visualize_points(self, image, init_pts, densified_pts, adjusted_pts, filename='point_visualization.png'):
        """
        Visualize initial, densified, and adjusted points with color-coded overlaps.
        - Red: initial only
        - Green: densified only
        - Blue: adjusted only
        - Magenta: initial + densified
        - Yellow: initial + adjusted
        - Cyan: densified + adjusted
        - Black: all three overlap
        """
        
        def point_key(p):
            return (int(p[0]), int(p[1]))

        # Create white canvas
        H, W = image.shape[:2] if len(image.shape) > 2 else image.shape
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

        # Convert to sets of int tuples for easy comparison
        init_set = set(map(point_key, init_pts))
        dens_set = set(map(point_key, densified_pts))
        adj_set  = set(map(point_key, adjusted_pts))

        all_keys = init_set | dens_set | adj_set

        for x, y in all_keys:
            in_init = (x, y) in init_set
            in_dens = (x, y) in dens_set
            in_adj  = (x, y) in adj_set

            # Skip points outside canvas
            if not (0 <= x < W and 0 <= y < H):
                continue

            # Determine color based on combination
            if in_init and in_dens and in_adj:
                color = (0, 0, 0)           # Black
            elif in_init and in_dens:
                color = (255, 0, 255)       # Magenta
            elif in_init and in_adj:
                color = (0, 255, 255)       # Yellow
            elif in_dens and in_adj:
                color = (255, 255, 0)       # Cyan
            elif in_init:
                color = (0, 0, 255)         # Red
            elif in_dens:
                color = (0, 255, 0)         # Green
            elif in_adj:
                color = (255, 0, 0)         # Blue
            else:
                continue  # Should not happen

            cv2.circle(canvas, (int(x), int(y)), 1, color, -1)

        # Save the visualization
        cv2.imwrite(filename, canvas)
        print(f"Point visualization saved to {filename}")
        
        return canvas
        
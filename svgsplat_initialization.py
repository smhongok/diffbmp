import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET
from copy import deepcopy

class StructureAwareInitializer:
    def __init__(self, num_init=100, alpha=0.3, min_distance=20, 
                 peak_threshold=0.5, radii_min=2, radii_max=None, 
                 v_init_mean=-5.0):
        self.num_init = num_init
        self.alpha = alpha
        self.min_distance = min_distance
        self.peak_threshold = peak_threshold
        self.radii_min = radii_min
        self.radii_max = radii_max
        self.v_init_mean = v_init_mean

    def curvature_aware_densification(self, edge_map, points):
        h, w = edge_map.shape
        new_points = []
        edge_norm = edge_map.astype(np.float32) / 255.0
        low_edge_coords = np.argwhere(edge_norm < 0.2)
        for (y, x) in low_edge_coords:
            if len(points) == 0 or np.min(np.linalg.norm(points - np.array([x, y]), axis=1)) > self.min_distance:
                new_points.append([x, y])
        return np.vstack([points, np.array(new_points)]) if new_points else points

    def structure_aware_adjustment(self, points, grad_x, grad_y):
        h, w = grad_x.shape
        adjusted = []
        for (x, y) in points:
            ix, iy = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            dx, dy = grad_x[iy, ix], grad_y[iy, ix]
            new_x = np.clip(x + self.alpha * dx, 0, w - 1)
            new_y = np.clip(y + self.alpha * dy, 0, h - 1)
            adjusted.append([new_x, new_y])
        return np.array(adjusted)

    def initialize_for_svg(self, I_target):
        """
        Specialization for SVG input to match the API expected in main_svg.py
        """
        device = I_target.device
        # Extract image dimensions
        if I_target.ndim == 3:
            H, W, _ = I_target.shape
        else:
            H, W = I_target.shape
            
        N = self.num_init
        
        # For SVG initialization, we'll use the image structure to guide placement
        if isinstance(I_target, torch.Tensor):
            I_np = I_target.cpu().numpy()
            # If it's a color image, convert to grayscale for structure analysis
            if I_np.ndim == 3:
                I_np = np.mean(I_np, axis=2)
        else:
            I_np = I_target
            if I_np.ndim == 3:
                I_np = cv2.cvtColor(I_np, cv2.COLOR_RGB2GRAY)
                
        # Now use our structure-aware initialization
        edges = cv2.Canny(I_np.astype(np.uint8), 100, 200)
        grad_y, grad_x = np.gradient(I_np.astype(np.float32))
        
        # Start with random points
        init_pts = np.random.rand(N, 2) * np.array([W, H])
        
        # Apply our structure-aware techniques
        densified_pts = self.curvature_aware_densification(edges, init_pts)
        adjusted_pts = self.structure_aware_adjustment(densified_pts, grad_x, grad_y)
        
        # Convert to tensors with requires_grad=True
        x = torch.tensor(adjusted_pts[:, 0], dtype=torch.float32, device=device, requires_grad=True)
        y = torch.tensor(adjusted_pts[:, 1], dtype=torch.float32, device=device, requires_grad=True)
        num_points = len(x)
        r = torch.rand(num_points, device=device, requires_grad=True) * min(H, W) / 8 + min(H, W) / 32
        v = torch.full((num_points,), self.v_init_mean, device=device, requires_grad=True)
        theta = torch.rand(num_points, device=device, requires_grad=True) * 2 * np.pi
        
        # If we have fewer points than requested, pad with random ones
        if num_points < N:
            additional = N - num_points
            x_add = torch.rand(additional, device=device, requires_grad=True) * W
            y_add = torch.rand(additional, device=device, requires_grad=True) * H
            r_add = torch.rand(additional, device=device, requires_grad=True) * min(H, W) / 8 + min(H, W) / 32
            v_add = torch.full((additional,), self.v_init_mean, device=device, requires_grad=True)
            theta_add = torch.rand(additional, device=device, requires_grad=True) * 2 * np.pi
            
            x = torch.cat([x, x_add])
            y = torch.cat([y, y_add])
            r = torch.cat([r, r_add])
            v = torch.cat([v, v_add])
            theta = torch.cat([theta, theta_add])
        
        return x, y, r, v, theta

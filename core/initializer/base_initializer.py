import numpy as np
import cv2
import torch
import os
import time
import traceback
from datetime import timedelta
from abc import ABC, abstractmethod
from core.renderer.vector_renderer import VectorRenderer
from typing import Dict, Any

class BaseInitializer(ABC):
    def __init__(self, init_opt:Dict[str, Any]):
        self.num_init = init_opt.get("N", 100)
        self.alpha = init_opt.get("alpha", 0.3)
        self.min_distance = init_opt.get("min_distance", 20)
        self.peak_threshold = init_opt.get("peak_threshold", 0.5)
        self.radii_min = init_opt.get("radii_min", 2)
        self.radii_max = init_opt.get("radii_max", None)
        self.v_init_bias = init_opt.get("v_init_bias", -5.0)
        self.v_init_slope = init_opt.get("v_init_slope", 10.0)
        self.keypoint_extracting = init_opt.get("keypoint_extracting", False)
        self.debug_mode = init_opt.get("debug_mode", False)
        self.distance_factor = init_opt.get("distance_factor", 0.0)
        
    @abstractmethod
    def initialize(self, I_target, target_binary_mask = None, I_bg=None, renderer:VectorRenderer=None, opt_conf:Dict[str, Any]=None):
        pass
    
    def _rand_leaf(self, shape, low, high, device):
        t = torch.empty(shape, device=device).uniform_(low, high)
        return t.requires_grad_(True)   # leaf tensor 
    
    @staticmethod
    def visualize_points(image, init_pts, densified_pts, adjusted_pts, filename='point_visualization.png'):
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
        
    def curvature_aware_densification(self, edge_map, points, N, min_dist, curvature_threshold=0.2, target_binary_mask = None):
        N_add = N - len(points)
        h, w = edge_map.shape
        edge_norm = edge_map.astype(np.float32) / 255.0

        # 1. Mask only low-curvature regions (below threshold)
        mask = edge_norm < curvature_threshold

        # 1.1. Apply target binary mask if provided
        # This allows for additional constraints on where points can be added
        # : points are only added in regions where both low curvature and not in background 
        if target_binary_mask is not None:
            # Convert to numpy array if it's a torch.Tensor
            if hasattr(target_binary_mask, 'detach'):  # torch.Tensor
                binary_mask_np = target_binary_mask.detach().cpu().numpy()
            else:
                binary_mask_np = target_binary_mask
            
            # Ensure mask is in correct format (0 or 1 values)
            if binary_mask_np.dtype == np.bool_:
                binary_mask_np = binary_mask_np.astype(np.float32)
            elif binary_mask_np.dtype != np.float32:
                binary_mask_np = binary_mask_np.astype(np.float32)
            
            mask = mask * binary_mask_np
        else:
            # If no target mask is provided, keep the original mask
            mask = mask * 1.0 
            
        num_total = edge_norm.size
        num_valid = np.count_nonzero(mask)
        print(f"Low-edge pixels: {num_valid} / {num_total} ({100 * num_valid / num_total:.2f}%)")

        if not np.any(mask):
            print("Warning: No low-edge regions found under threshold. Returning original points.")
            return points

        # 2. Create probability map from masked region
        prob_map = (1.0 - edge_norm) * mask.astype(np.float32)
        prob_flat = prob_map.flatten()
        prob_flat /= prob_flat.sum()  # Normalize to sum=1

        # 3. Get coordinates only in masked region
        all_coords = np.array([(y, x) for y in range(h) for x in range(w)])
        valid_indices = np.flatnonzero(mask.flatten())
        
        # 4. Limit the number of candidates to avoid too many points
        num_candidates = min(len(valid_indices), N_add * 5)
        
        if num_candidates == 0:
            print("Warning: No valid candidates to sample from.")
            return points

        # 5. Sample from only valid (low-edge) positions
        sampled_indices = np.random.choice(valid_indices, size=num_candidates, replace=False, p=prob_flat[valid_indices])
        candidate_coords = all_coords[sampled_indices]

        new_points = []
        for (y, x) in candidate_coords:
            if len(points) == 0 or np.min(np.linalg.norm(points - np.array([x, y]), axis=1)) > min_dist:
                new_points.append([x, y])
                if len(new_points) >= N_add:
                    break

        return np.vstack([points, np.array(new_points)]) if new_points else points

    def structure_aware_adjustment(self, points, grad_x, grad_y, target_binary_mask=None):
        h, w = grad_x.shape
        
        # Convert target_binary_mask to numpy array if it's a torch.Tensor
        mask_np = None
        if target_binary_mask is not None:
            if hasattr(target_binary_mask, 'detach'):  # torch.Tensor
                mask_np = target_binary_mask.detach().cpu().numpy()
            else:
                mask_np = target_binary_mask
            
            # Ensure mask is in correct format (0 or 1 values)
            if mask_np.dtype == np.bool_:
                mask_np = mask_np.astype(np.float32)
            elif mask_np.dtype != np.float32:
                mask_np = mask_np.astype(np.float32)
        
        adjusted = []
        for (x, y) in points:
            ix, iy = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            dx, dy = grad_x[iy, ix], grad_y[iy, ix]
            new_x = np.clip(x + self.alpha * dx, 0, w - 1)
            new_y = np.clip(y + self.alpha * dy, 0, h - 1)
            
            # Check if new position would be in background region (mask=1)
            if mask_np is not None:
                new_ix, new_iy = int(np.clip(new_x, 0, w - 1)), int(np.clip(new_y, 0, h - 1))
                if mask_np[new_iy, new_ix] == 1:  # Background region, bad case
                    # Keep original position
                    adjusted.append([x, y])
                else: # Good case
                    # Use adjusted position
                    adjusted.append([new_x, new_y])
            else:
                # No mask constraint, use adjusted position
                adjusted.append([new_x, new_y])
                
        return np.array(adjusted)
    
    def coarse_to_fine_densification(self, edge_map, N, levels, refine_min_dist=False, target_binary_mask = None): 
        def densify_at_levels(edge_map, N, levels, initial_points, initial_levels, base_min_dist):
            points = initial_points.copy()
            point_levels = initial_levels.copy() if initial_levels is not None else np.array([])

            for level in range(0, levels + 1):
                scale = 2 ** -(levels - level)
                small_edge = cv2.resize(edge_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
                # Handle target_binary_mask type conversion for OpenCV
                small_target_binary_mask = None
                if target_binary_mask is not None:
                    # Convert to numpy array if it's a torch.Tensor
                    if hasattr(target_binary_mask, 'detach'):  # torch.Tensor
                        mask_np = 1-target_binary_mask.detach().cpu().numpy()
                    else:
                        mask_np = 1-target_binary_mask
                    
                    # Ensure it's the right data type for cv2.resize
                    if mask_np.dtype != np.uint8:
                        mask_np = mask_np.astype(np.uint8)
                    
                    small_target_binary_mask = cv2.resize(mask_np, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                
                small_points = points * scale

                scaled_min_dist = base_min_dist * scale
                densified = self.curvature_aware_densification(small_edge, small_points, N, scaled_min_dist, target_binary_mask = small_target_binary_mask)

                # Add only newly added points (after scaling back)
                num_new_points = len(densified) - len(small_points)
                if num_new_points > 0:
                    new_pts = densified[len(small_points):] / scale
                    points = np.vstack([points, new_pts])

                    # Track level information for each new point
                    new_levels = np.ones(num_new_points, dtype=np.int32) * level
                    point_levels = np.concatenate([point_levels, new_levels])

                # Deduplicate by integer rounding and maintain level information
                if len(points) > 0:
                    int_points = points.astype(np.int32)
                    unique_indices = []
                    seen = set()
                    
                    for i, pt in enumerate(int_points):
                        pt_tuple = (pt[0], pt[1])
                        if pt_tuple not in seen:
                            seen.add(pt_tuple)
                            unique_indices.append(i)
                    
                    points = points[unique_indices]
                    point_levels = point_levels[unique_indices]

                if len(points) >= N:
                    break

            return points, point_levels

        print("edge_map.shape: ", edge_map.shape)
        points = np.empty((0, 2), dtype=np.float32)
        point_levels = np.array([], dtype=np.int32)

        # Step 1: Coarse-to-fine densification with default min_distance
        points, point_levels = densify_at_levels(edge_map, N, levels, points, point_levels, self.min_distance)

        # Step 2 (optional): refine with smaller min_distance
        if refine_min_dist and len(points) < N:
            points, point_levels = densify_at_levels(edge_map, N, levels, points, point_levels, base_min_dist=1.0)
            if len(points) < N:
                points, point_levels = densify_at_levels(edge_map, N, levels, points, point_levels, base_min_dist=0.0)

        # Return both points and their level information
        if len(points) > N:
            points = points[:N]
            point_levels = point_levels[:N]

        return points, point_levels
    
    def find_best_densification(self, edge_map, N, W, target_binary_mask = None):
        best_points = None
        best_levels = None
        best_level = 8
        best_min_distance = None
        max_len = -1

        # Scale min_dists based on target_binary_mask width (final_width)
        width_scale = W / 256.0  # normalize to 256 baseline
        min_dists = [1.9 * width_scale, 1.5 * width_scale, 1.0 * width_scale, 0.5 * width_scale]
        while best_level > 0:
            print("best_level: ", best_level)
            try:
                for min_dist in min_dists:
                    self.min_distance = min_dist
                    points, point_levels = self.coarse_to_fine_densification(edge_map, N, best_level, target_binary_mask = target_binary_mask)
                    if len(points) > max_len:
                        print("len(points): ", len(points), ", N: ", N, ", level: ", best_level, ", min_dist: ", min_dist)
                        max_len = len(points)
                        best_points = points[:N]
                        best_levels = point_levels[:N]
                        best_min_distance = min_dist
                        if len(points) >= N:
                            break  # good enough, go shallower
                if len(points) >= N:
                    break  # good enough, go shallower
                
            except Exception as e:
                print(f"Error in find_best_densification: {e}")
                traceback.print_exc()
                
            best_level -= 1
            
        if len(points) < N:
            print("Couldn't generate enough points with given constraints. Try again with min_distance 1.0")
            self.min_distance = best_min_distance
            best_points, best_levels = self.coarse_to_fine_densification(edge_map, N, best_level, refine_min_dist=True, target_binary_mask = target_binary_mask)

        print(f"Using level={best_level}, min_distance={best_min_distance:.2f}")
        return best_points, best_levels
    
    def _random_splat_params(self, N, y_init, x_init, H, W, device):
        x = self._rand_leaf((N,), x_init, x_init + W, device)
        y = self._rand_leaf((N,), y_init, y_init + H, device)

        r_min = self.radii_min
        if self.radii_max is not None:
            r_max = self.radii_max
        else:
            r_max = 0.5 * min(H, W)
        r = self._rand_leaf((N,), r_min, r_max, device)

        v = self._rand_leaf((N,),self.v_init_bias - 0.5, self.v_init_bias + 0.5, device)

        theta = self._rand_leaf((N,), 0, 2 * torch.pi, device)
        c     = self._rand_leaf((N,3), 0, 1, device)
        return x, y, r, v, theta, c
    
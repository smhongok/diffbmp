import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET
from copy import deepcopy
import matplotlib.pyplot as plt
import os

def point_key(p):
    return (int(p[0]), int(p[1]))

class StructureAwareInitializer:
    def __init__(self, num_init=100, alpha=0.3, min_distance=20, 
                 peak_threshold=0.5, radii_min=2, radii_max=None, 
                 v_init_mean=-5.0, keypoint_extracting=False, debug_mode=False):
        self.num_init = num_init
        self.alpha = alpha
        self.min_distance = min_distance
        self.peak_threshold = peak_threshold
        self.radii_min = radii_min
        self.radii_max = radii_max
        self.v_init_mean = v_init_mean
        self.keypoint_extracting = keypoint_extracting
        self.debug_mode = debug_mode
        
    def curvature_aware_densification(self, edge_map, points, N, min_dist):
        N_add = N - len(points)
        h, w = edge_map.shape
        edge_norm = edge_map.astype(np.float32) / 255.0

        # 1. Mask only low-curvature regions (below threshold)
        mask = edge_norm < 0.2
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
    
    def coarse_to_fine_densification(self, edge_map, N, levels, refine_min_dist=False):
        h, w = edge_map.shape
        print("edge_map.shape: ", edge_map.shape)
        points = np.empty((0, 2), dtype=np.float32)

        for level in range(0, levels + 1, 1):
            scale = 2 ** -(levels - level)
            small_edge = cv2.resize(edge_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            small_points = points * scale

            # Scale-aware min_distance
            scaled_min_dist = self.min_distance * scale
            densified = self.curvature_aware_densification(small_edge, small_points, N, scaled_min_dist)

            # Upscale newly added points only
            new_pts = densified[len(small_points):] / scale

            # Append newly added points to current set
            points = np.vstack([points, new_pts])

            # Optionally, deduplicate
            points = np.unique(points.astype(np.int32), axis=0).astype(np.float32)
            
            if len(points) >= N:
                break
            
        if refine_min_dist:
            for level in range(0, levels + 1, 1):
                scale = 2 ** -(levels - level)
                small_edge = cv2.resize(edge_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                small_points = points * scale

                # Scale-aware min_distance
                scaled_min_dist = 1 * scale
                densified = self.curvature_aware_densification(small_edge, small_points, N, scaled_min_dist)

                # Upscale newly added points only
                new_pts = densified[len(small_points):] / scale

                # Append newly added points to current set
                points = np.vstack([points, new_pts])

                # Optionally, deduplicate
                points = np.unique(points.astype(np.int32), axis=0).astype(np.float32)
                
                if len(points) >= N:
                    break
        
        return points[:N]
    
    def find_best_densification(self, edge_map, N):
        best_points = None
        best_level = 7
        best_min_distance = None
        max_len = -1

        for min_dist in range(20, 1, -2):
            self.min_distance = min_dist
            points = self.coarse_to_fine_densification(edge_map, N, best_level)
            if len(points) > max_len:
                print("len(points): ", len(points), ", N: ", N, ", level: ", best_level, ", min_dist: ", min_dist)
                max_len = len(points)
                best_points = points[:N]
                best_min_distance = min_dist
                if len(points) >= N:
                    break  # good enough, go shallower
            
        if len(points) < N:
            print("Couldn't generate enough points with given constraints. Try again with min_distance 1.")
            self.min_distance = best_min_distance
            best_points = self.coarse_to_fine_densification(edge_map, N, best_level, refine_min_dist=True)

        print(f"Using level={best_level}, min_distance={best_min_distance:.2f}")
        return best_points

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
        densified_pts = self.find_best_densification(edges, N)
        adjusted_pts = self.structure_aware_adjustment(densified_pts, grad_x, grad_y)
        print("len(densified_pts): ", len(densified_pts), ", len(adjusted_pts): ", len(adjusted_pts))
        
        # Visualize points if debug mode is enabled
        if self.debug_mode:
            vis_canvas = self.visualize_points(I_np, init_pts, densified_pts, adjusted_pts, 'outputs/point_debug.png')
            
            # Create side-by-side visualization with original image
            if isinstance(I_target, torch.Tensor):
                orig_img = I_target.cpu().numpy()
                if orig_img.ndim == 2:
                    orig_img = np.stack([orig_img] * 3, axis=-1)
                elif orig_img.max() <= 1.0:
                    orig_img = (orig_img * 255).astype(np.uint8)
            else:
                orig_img = I_np
                if orig_img.ndim == 2:
                    orig_img = np.stack([orig_img] * 3, axis=-1)
            
            orig_img = orig_img[:, :, ::-1]
            # Create side-by-side image
            combined = np.hstack((orig_img, vis_canvas))
            
            # Ensure outputs directory exists
            os.makedirs('outputs', exist_ok=True)
            
            cv2.imwrite('outputs/side_by_side_debug.png', combined)
            print("Side-by-side visualization saved to outputs/side_by_side_debug.png")
        
        # Convert to tensors with requires_grad=True
        x = torch.tensor(adjusted_pts[:, 0], dtype=torch.float32, device=device, requires_grad=True)
        y = torch.tensor(adjusted_pts[:, 1], dtype=torch.float32, device=device, requires_grad=True)
        num_points = len(x)
        # r = torch.rand(num_points, device=device, requires_grad=True) * min(H, W) / 8 + min(H, W) / 32
        r = torch.rand(num_points, device=device, requires_grad=True) * min(H, W) / 16 + self.radii_min
        # r = torch.poisson(torch.full((num_points,), rate, device=device, requires_grad=False)) + self.radii_min
        v = torch.full((num_points,), self.v_init_mean, device=device, requires_grad=True)
        theta = torch.rand(num_points, device=device, requires_grad=True) * 2 * np.pi
        print("len(x): ", len(x))
        
        return x, y, r, v, theta

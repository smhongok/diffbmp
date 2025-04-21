import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET
from copy import deepcopy
import matplotlib.pyplot as plt
import os

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
        new_points = []
        edge_norm = edge_map.astype(np.float32) / 255.0
        low_edge_coords = np.argwhere(edge_norm < 0.2)
        for (y, x) in low_edge_coords:
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
    
    def coarse_to_fine_densification(self, edge_map, N, levels=5):
        h, w = edge_map.shape
        points = np.empty((0, 2), dtype=np.float32)

        for level in range(levels):
            scale = 2 ** -(levels - level - 1)
            small_edge = cv2.resize(edge_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            small_points = points * scale

            # Scale-aware min_distance
            scaled_min_dist = self.min_distance * scale

            # Inject into the existing densification function
            densified = self.curvature_aware_densification(small_edge, small_points, N, scaled_min_dist)
            points = densified / scale
            points = np.unique(points.astype(np.int32), axis=0).astype(np.float32)

            if len(points) >= N:
                break
        
        return points[:N]

    def visualize_points(self, image, init_pts, densified_pts, adjusted_pts, filename='point_visualization.png'):
        """
        Visualize densified_pts in red and adjusted_pts in blue on a white canvas.
        
        Args:
            image: Original image (for size reference)
            init_pts: Initial points (red)
            densified_pts: Points after densification (green)
            adjusted_pts: Points after adjustment (blue)
            filename: Output filename
        """
        # Create white canvas with same size as input image
        H, W = image.shape[:2] if len(image.shape) > 2 else image.shape
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
                
        is_densified_dup_list = []
        is_adjusted_dup_list = []
        # Draw initial points in red
        for xi, yi in init_pts:
            xi, yi = int(xi), int(yi)
            is_densified_dup = False
            is_adjusted_dup = False
            if 0 <= xi < W and 0 <= yi < H:
                for idx, (xa, ya) in enumerate(adjusted_pts):
                    xa, ya = int(xa), int(ya)
                    if 0 <= xa < W and 0 <= ya < H:
                        if xi==xa and yi==ya:
                            is_adjusted_dup = True
                            is_adjusted_dup_list.append(idx)
                            break
                for idx, (xd, yd) in enumerate(densified_pts):
                    xd, yd = int(xd), int(yd)
                    if 0 <= xd < W and 0 <= yd < H:
                        if xi==xd and yi==yd:
                            is_densified_dup = True
                            is_densified_dup_list.append(idx)
                            break
            if is_densified_dup and is_adjusted_dup:
                cv2.circle(canvas, (xi, yi), 1, (0, 0, 0), -1)  # Black (BGR)
            elif is_densified_dup:
                cv2.circle(canvas, (xi, yi), 1, (255, 0, 255), -1)  # Magenta (BGR)
            elif is_adjusted_dup:
                cv2.circle(canvas, (xi, yi), 1, (0, 255, 255), -1)  # Yellow (BGR)
            else:
                cv2.circle(canvas, (xi, yi), 1, (0, 0, 255), -1)  # Red (BGR)
        
        # Draw densified points in green
        for idxd, (xd, yd) in enumerate(densified_pts):
            if idxd in is_densified_dup_list:
                continue
            is_adjusted_dup = False
            xd, yd = int(xd), int(yd)
            if 0 <= xd < W and 0 <= yd < H:
                for idxa, (xa, ya) in enumerate(adjusted_pts):
                    xa, ya = int(xa), int(ya)
                    if 0 <= xa < W and 0 <= ya < H:
                        if xd==xa and yd==ya:
                            is_adjusted_dup = True
                            is_adjusted_dup_list.append(idxa)
                            break
                        
            if is_adjusted_dup:
                cv2.circle(canvas, (xd, yd), 1, (255, 255, 0), -1)  # Yellow (BGR)
            else:
                cv2.circle(canvas, (xd, yd), 1, (0, 255, 0), -1)  # Green (BGR)
        
        # Draw adjusted points in blue
        for idx, (x, y) in enumerate(adjusted_pts):
            if idx in is_adjusted_dup_list:
                continue
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 1, (255, 0, 0), -1)  # Blue (BGR)
                
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
        densified_pts = self.coarse_to_fine_densification(edges, N)
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

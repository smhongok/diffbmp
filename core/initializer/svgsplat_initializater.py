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
# 시작 시간 기록

class StructureAwareInitializer(BaseInitializer):
    def __init__(self, num_init=100, alpha=0.3, min_distance=20, 
                 peak_threshold=0.5, radii_min=2, radii_max=None, 
                 v_init_bias=-5.0, v_init_slope=0.0, keypoint_extracting=False, debug_mode=False):
        super().__init__(num_init, alpha, min_distance, peak_threshold, radii_min, 
                         radii_max, v_init_bias, v_init_slope, keypoint_extracting, debug_mode)
        
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
        def densify_at_levels(edge_map, N, levels, initial_points, base_min_dist):
            points = initial_points.copy()

            for level in range(0, levels + 1):
                scale = 2 ** -(levels - level)
                small_edge = cv2.resize(edge_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                small_points = points * scale

                scaled_min_dist = base_min_dist * scale
                densified = self.curvature_aware_densification(small_edge, small_points, N, scaled_min_dist)

                # Add only newly added points (after scaling back)
                new_pts = densified[len(small_points):] / scale
                points = np.vstack([points, new_pts])

                # Deduplicate by integer rounding
                points = np.unique(points.astype(np.int32), axis=0).astype(np.float32)

                if len(points) >= N:
                    break

            return points

        print("edge_map.shape: ", edge_map.shape)
        points = np.empty((0, 2), dtype=np.float32)

        # Step 1: Coarse-to-fine densification with default min_distance
        points = densify_at_levels(edge_map, N, levels, points, self.min_distance)

        # Step 2 (optional): refine with smaller min_distance
        if refine_min_dist and len(points) < N:
            points = densify_at_levels(edge_map, N, levels, points, base_min_dist=1.0)
            if len(points) < N:
                points = densify_at_levels(edge_map, N, levels, points, base_min_dist=0.9)

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
   
    def initialize(self, I_target):
        """
        Specialization for SVG input to match the API expected in main_svg.py
        """
        start_time = time.time()
        device = I_target.device
        # Extract image dimensions
        if I_target.ndim == 3:
            H, W, _ = I_target.shape
            I_color = I_target.cpu().numpy()            # (H,W,3), 0~1
        else:
            H, W = I_target.shape
            gray = np.expand_dims(I_np / 255.0, axis=-1)
            I_color = np.repeat(gray, 3, axis=-1)
            
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

        # Color initialization
        # 스플랫 좌표에 해당하는 픽셀 색 샘플
        idx_x = np.clip(np.round(adjusted_pts[:, 0]).astype(int), 0, W - 1)
        idx_y = np.clip(np.round(adjusted_pts[:, 1]).astype(int), 0, H - 1)
        c_init = I_color[idx_y, idx_x]                  # (N,3) float32

        # 약간의 노이즈로 파라미터 다양화
        c_init += np.random.normal(0.0, 0.02, c_init.shape)
        c_init = np.clip(c_init, 0.0, 1.0)              # 안전 클립
        
        # -------------------- radius 샘플 & 정렬 -------------------- #
        num_points = adjusted_pts.shape[0]
        r_np = np.random.rand(num_points) * min(H, W) / 4 + self.radii_min   # (N,)
        sort_idx = np.argsort(r_np)           # 오름차순 (+)  → 큰 r 이 나중

        # 좌표·색·반경 모두 같은 순서로 재정렬
        adjusted_pts = adjusted_pts[sort_idx]
        r_np         = r_np[sort_idx]
        c_init       = c_init[sort_idx]

        # -------------------- tensor 변환 --------------------------- #
        x = torch.tensor(adjusted_pts[:, 0], dtype=torch.float32,
                         device=device, requires_grad=True)
        y = torch.tensor(adjusted_pts[:, 1], dtype=torch.float32,
                         device=device, requires_grad=True)

        r = torch.tensor(r_np, dtype=torch.float32,
                         device=device, requires_grad=True)

        # -------------------- opacity v 초기화 (레이어 일치) -------- #
        rank = torch.linspace(0.0, 1.0, steps=num_points, device=device)     # 0(아래)→1(위)
        v = (self.v_init_bias + self.v_init_slope * rank).clone().detach()
        v += torch.empty_like(v).normal_(mean=0.0, std=0.05)
        v.requires_grad_(True)

        theta = torch.rand(num_points, device=device, requires_grad=True) * 2 * np.pi
        c = torch.tensor(c_init, dtype=torch.float32,
                         device=device, requires_grad=True)
        print("len(x): ", len(x))
        
        end_time = time.time()
        formatted_time = str(timedelta(seconds=int(end_time - start_time)))
        # 수행 시간 출력
        print(f"[initialize]total_cost_time: {formatted_time}")
        
        return x, y, r, v, theta, c

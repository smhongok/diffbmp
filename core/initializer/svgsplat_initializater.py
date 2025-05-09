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
# 시작 시간 기록

class StructureAwareInitializer(BaseInitializer):
    def __init__(self, init_opt:Dict[str, Any]):
        super().__init__(init_opt)
        
    def initialize(self, I_target, I_bg=None, renderer:VectorRenderer=None, opt_conf:Dict[str, Any]=None):
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
        densified_pts, point_levels = self.find_best_densification(edges, N)
        adjusted_pts = self.structure_aware_adjustment(densified_pts, grad_x, grad_y)
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
        
        # -------------------- Radius initialization based on levels -------------------- #
        num_points = adjusted_pts.shape[0]
        
        # Calculate max level for normalization
        max_level = np.max(point_levels) if len(point_levels) > 0 else 1
        
        # Determine radius based on level - coarser level (smaller value) gets larger radius
        # The formula: r = max_radius * (1 - level/max_level) + min_radius
        max_radius = min(H, W) / 4
        min_radius = self.radii_min
        
        # Calculate radius for each point - invert the level so lower levels get larger radii
        normalized_levels = 1.0 - (point_levels / max_level)
        r_np = min_radius + normalized_levels * (max_radius - min_radius)
        
        # Add some noise for variety while preserving the coarse-to-fine relationship
        noise_scale = 0.15  # Scale of noise relative to radius range
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

        # -------------------- Initialize opacity v (layer consistent) -------- #
        if False:
            # counts 리스트의 길이에 따라 균일하게 분포된 rank 생성
            rank = torch.linspace(0.0, 1.0, steps=len(counts), device=device)

            # 각 level별 v 값 계산
            level_v_values = (self.v_init_bias - 0.5) * (1 - rank) - 2.0 * rank  # rank가 0일 때 v_init_bias-0.5, rank가 1일 때 -1.0

            # 각 point에 대해 해당 level의 v 값 할당
            v = torch.zeros(num_points, device=device)
            start_idx = 0
            for level_idx, count in enumerate(counts):
                end_idx = start_idx + count
                v[start_idx:end_idx] = level_v_values[level_idx]
                start_idx = end_idx
        elif True:
            v = torch.full((num_points,), -2.0, device=device)
        else:
            rank = torch.linspace(0.0, 1.0, steps=num_points, device=device)     # 0(bottom)→1(top)
            v = (self.v_init_bias - 0.5 + self.v_init_slope * rank).clone().detach()
            v += torch.empty_like(v).normal_(mean=0.0, std=0.05)
        v.requires_grad_(True)

        theta = torch.rand(num_points, device=device, requires_grad=True) * 2 * np.pi
        c = torch.tensor(c_init, dtype=torch.float32,
                         device=device, requires_grad=True)
        print("len(x): ", len(x))
        
        end_time = time.time()
        formatted_time = str(timedelta(seconds=int(end_time - start_time)))
        print(f"[initialize]total_cost_time: {formatted_time}")
        
        return x, y, r, v, theta, c

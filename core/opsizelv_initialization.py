import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import time
from datetime import timedelta
from base_initializer import BaseInitializer, _rand_leaf

class OpSizeLvAwareInitializer(BaseInitializer):
    def __init__(self, num_init=100, alpha=0.3, min_distance=20, 
                 peak_threshold=0.5, radii_min=2, radii_max=None, 
                 v_init_bias=-5.0, v_init_slope=0.0, keypoint_extracting=False, whole_random=False, debug_mode=False):
        super().__init__(num_init, alpha, min_distance, peak_threshold, radii_min, 
                         radii_max, v_init_bias, v_init_slope, keypoint_extracting, 
                         whole_random, debug_mode)
   
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

        if self.whole_random:
            print("Using whole random initialization.")
            
            # adjusted_pts = np.random.rand(N, 2) * np.array([W, H])
            # init_pts = np.empty((0, 2))
            # densified_pts = np.empty((0, 2))
            return self._random_splat_params(self.num_init, H, W, device)
        else:
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
            r_np = np.random.rand(num_points) * min(H, W) / 2 + self.radii_min   # (N,)
            sort_idx = np.argsort(+r_np)           # 오름차순 (+)  → 큰 r 이 나중 그래야 밑에 깔림

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

            theta = _rand_leaf((num_points,), 0, 2 * torch.pi, device)
            c = torch.tensor(c_init, dtype=torch.float32,
                            device=device, requires_grad=True)
            print("len(x): ", len(x))

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
        

        
        end_time = time.time()
        formatted_time = str(timedelta(seconds=int(end_time - start_time)))
        # 수행 시간 출력
        print(f"[initialize]total_cost_time: {formatted_time}")
        
        return x, y, r, v, theta, c


    def _random_splat_params(self, N, H, W, device):
        x = _rand_leaf((N,),       0,       W, device)
        y = _rand_leaf((N,),       0,       H, device)

        r_min, r_max = self.radii_min, 0.5 * min(H, W)
        r = _rand_leaf((N,),   r_min,  r_max, device)

        v = _rand_leaf((N,),
                    self.v_init_bias - 0.5,
                    self.v_init_bias + 0.5, device)

        theta = _rand_leaf((N,), 0, 2 * torch.pi, device)
        c     = _rand_leaf((N,3), 0, 1, device)
        return x, y, r, v, theta, c
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

class RandomInitializer(BaseInitializer):
    def __init__(self, num_init=100, alpha=0.3, min_distance=20, 
                 peak_threshold=0.5, radii_min=2, radii_max=None, 
                 v_init_bias=-5.0, v_init_slope=0.0, keypoint_extracting=False, debug_mode=False):
        super().__init__(num_init, alpha, min_distance, peak_threshold, radii_min, 
                         radii_max, v_init_bias, v_init_slope, keypoint_extracting, debug_mode)
   
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

        
        print("Using whole random initialization.")
        
        # adjusted_pts = np.random.rand(N, 2) * np.array([W, H])
        # init_pts = np.empty((0, 2))
        # densified_pts = np.empty((0, 2))
        return self._random_splat_params(self.num_init, H, W, device)
        
    def _random_splat_params(self, N, H, W, device):
        x = self._rand_leaf((N,), 0, W, device)
        y = self._rand_leaf((N,), 0, H, device)

        r_min, r_max = self.radii_min, 0.5 * min(H, W)
        r = self._rand_leaf((N,), r_min, r_max, device)

        v = self._rand_leaf((N,),self.v_init_bias - 0.5, self.v_init_bias + 0.5, device)

        theta = self._rand_leaf((N,), 0, 2 * torch.pi, device)
        c     = self._rand_leaf((N,3), 0, 1, device)
        return x, y, r, v, theta, c
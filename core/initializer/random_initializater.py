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

class RandomInitializer(BaseInitializer):
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
        return self._random_splat_params(self.num_init, 0, 0, H, W, device)

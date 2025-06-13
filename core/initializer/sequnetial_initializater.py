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

class SequentialInitializer(BaseInitializer):
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
        
        return self._sequential_splat_params(self.num_init, 0, 0, H, W, device)

    def _sequential_splat_params(
        self,
        N: int,
        y_init: float,
        x_init: float,
        H: int,
        W: int,
        device: torch.device
    ):
        """
        균일한 크기·각도, 순차적 위치 배치 초기화.
        - N개를 grid_cols x grid_rows 격자로 나누고,
          (col + 0.5)*dx, (row + 0.5)*dy 위치에 splat 배치
        """
        # 1) 격자 크기 계산 (정사각형에 가깝게)
        grid_cols = int(np.ceil(np.sqrt(N)))
        grid_rows = int(np.ceil(N / grid_cols))
        dx = W / grid_cols
        dy = H / grid_rows

        # 2) x, y 좌표 생성
        idxs = np.arange(N)
        x_vals = x_init + ( (idxs % grid_cols) + 0.5 ) * dx
        y_vals = y_init + ( (idxs // grid_cols) + 0.5 ) * dy

        x = torch.tensor(x_vals, device=device, dtype=torch.float32)
        y = torch.tensor(y_vals, device=device, dtype=torch.float32)

        # 3) 반지름(r), 회전(theta) 고정
        r     = torch.full((N,), self.radii_min,      device=device, dtype=torch.float32)
        theta = torch.zeros((N,),          device=device, dtype=torch.float32)

        # 4) 가시성(v)과 색상(c)은 random
        v = self._rand_leaf((N,),
                            self.v_init_bias - 0.5,
                            self.v_init_bias + 0.5,
                            device)
        c = self._rand_leaf((N, 3), 0.0, 1.0, device)

        return x, y, r, v, theta, c
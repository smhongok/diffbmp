import numpy as np
import cv2
import torch
import os
import time
import traceback
from datetime import timedelta
from abc import ABC, abstractmethod
from imbrush.core.renderer.vector_renderer import VectorRenderer
from typing import Dict, Any
from imbrush.util.constants import OPACITY_INIT_VALUE

class BaseInitializer(ABC):
    def __init__(self, init_opt:Dict[str, Any]):
        self.num_init = init_opt.get("N", 100)
        self.radii_min = init_opt.get("radii_min", 2)
        self.radii_max = init_opt.get("radii_max", None)
        self.debug_mode = init_opt.get("debug_mode", False)
        self.detail_first = init_opt.get("detail_first", True)
        self.v_init_bias = init_opt.get("v_init_bias", OPACITY_INIT_VALUE)
        
    @abstractmethod
    def initialize(self, I_target, target_binary_mask = None, I_bg=None, renderer:VectorRenderer=None, opt_conf:Dict[str, Any]=None):
        pass
    
    def _rand_leaf(self, shape, low, high, device):
        t = torch.empty(shape, device=device).uniform_(low, high)
        return t.requires_grad_(True)   # leaf tensor 
    
    def _random_splat_params(self, N, y_init, x_init, H, W, device):
        x = self._rand_leaf((N,), x_init, x_init + W, device)
        y = self._rand_leaf((N,), y_init, y_init + H, device)

        r_min = self.radii_min
        if self.radii_max is not None:
            r_max = self.radii_max
        else:
            r_max = 0.5 * min(H, W)
        r = self._rand_leaf((N,), r_min, r_max, device)

        v = self._rand_leaf((N,), self.v_init_bias - 0.5, self.v_init_bias + 0.5, device)

        theta = self._rand_leaf((N,), 0, 2 * torch.pi, device)
        c     = self._rand_leaf((N,3), 0, 1, device)
        return x, y, r, v, theta, c
    
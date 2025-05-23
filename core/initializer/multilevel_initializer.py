import gc
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from core.renderer.vector_renderer import VectorRenderer
from .singlelevel_initializer import SingleLevelInitializer

class MultiLevelInitializer(SingleLevelInitializer):
    """
    Implements Algorithm 1: MultiLevel-SVGSplat
    """
    def __init__(self, init_opt:Dict[str, Any]):
        super().__init__(init_opt)
        self.level = init_opt.get("level", 2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self, I_tar, I_bg=None, renderer:VectorRenderer=None, opt_conf:Dict[str, Any]=None) -> List[Tuple[torch.Tensor, ...]]:
        """
        Implements the MultiLevel-SVGSplat algorithm (Algorithm 1 in the paper).
        Args:
            I_tar: Target image (torch.Tensor)
            I_bg: Background image (torch.Tensor)
            renderer: VectorRenderer instance
            opt_conf: Dictionary of optimization configuration parameters
        Returns:
            List of tuples: Each tuple is (x, y, r, v, theta, c) for a splat set
        """
        P_all = []  # Global splat set list
        # 1. Blank canvas for background
        # [TODO] I_bg implementation 
        if I_bg is None:
            I_bg = torch.zeros_like(I_tar)
        
        # 2. Coarse pass (global)
        P_0 = super().initialize(I_tar, I_bg, renderer, opt_conf)
        P_all.append(P_0)
        
        # 3. Update background for next level
        I_hat = renderer.render_image(*P_0)
        I_1 = renderer.over(I_hat, I_bg)
        
        # 4. Multi-level refinement
        # Split into level**2 sections
        # Adjust number of points by frequency
        coords = self._split_into_level(I_tar, self.level)
        adjusted_points = self._adjust_points_by_frequency(I_tar, coords)
        print(f"adjusted_points: {adjusted_points}")
        for bbox_k, num_points in zip(coords, adjusted_points):
            # Crop and resize background and target
            I_bg_k = self._crop_and_resize(I_1, bbox_k)
            I_tar_k = self._crop_and_resize(I_tar, bbox_k)
            
            # Run single-level optimizer on section
            self.num_init = num_points
            P_k = super().initialize(I_tar_k, I_bg_k, renderer, opt_conf)
            
            # Map local splats to global coordinates
            s = bbox_k[2] / 256.0  # width/256
            x0, y0 = bbox_k[0], bbox_k[1]
            x, y, r, v, theta, c = P_k
            x = x * s + x0
            y = y * s + y0
            r = r * s
            P_k_global = (x, y, r, v, theta, c)
            P_all.append(P_k_global)
            
        return self._split_params(P_all)
    
    def _split_params(self, params_list):
        x_all, y_all, r_all, v_all, theta_all, c_all = [], [], [], [], [], []
        for params in params_list:
            x, y, r, v, theta, c = params
            x_all.append(x)
            y_all.append(y)
            r_all.append(r)
            v_all.append(v)
            theta_all.append(theta)
            c_all.append(c)
        x = torch.cat(x_all, dim=0)
        y = torch.cat(y_all, dim=0)
        r = torch.cat(r_all, dim=0)
        v = torch.cat(v_all, dim=0)
        theta = torch.cat(theta_all, dim=0)
        c = torch.cat(c_all, dim=0)
        
        return x, y, r, v, theta, c
    
    def _split_into_level(self, image: torch.Tensor, level):
        """
        Splits the image into level^2 quadrants and returns list of (cropped_image, bbox) tuples.
        bbox = (x0, y0, width, height)
        """
        H, W = image.shape[-2], image.shape[-1]
        h2, w2 = H // level, W // level
        h_mod, w_mod = H%level, W%level
        coords = [[i, j, w2, h2] for j in range(0, h2*level, h2) for i in range(0, w2*level, w2)] 
        for i in range(level-1, level**2, level):
            coords[i][2] += w_mod
        for i in range(level*(level-1), level**2):
            coords[i][3] += h_mod
            
        return coords

    def _crop_and_resize(self, image: torch.Tensor, bbox):
        """
        Crops and resizes the image to 256x256 for quadrant processing.
        bbox = (x0, y0, width, height)
        """
        x0, y0, w, h = bbox
        cropped = image[..., y0:y0+h, x0:x0+w]
        resized = torch.nn.functional.interpolate(cropped.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        # ---- Channel correction ----
        if resized.shape[-1] != 3:
            resized = resized[..., :3].contiguous()
        return resized

    def _calculate_high_frequency_ratio(self, image):
        """
        Calculate the ratio of high-frequency components in the image.
        
        Args:
            image: Image to analyze (numpy array)
            
        Returns:
            High-frequency component ratio (value between 0~1)
        """
        # 1) Prevent empty array
        if image is None or image.size == 0:
            raise ValueError(f"Empty image passed: shape={image.shape}")

        # 2) (C,H,W) → (H,W,C)
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
            
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                # If range is [0..1]
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
        # Convert image to grayscale if it's 3D (color)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Convert image to uint8 if it's float type
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # FFT transformation
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # Create frequency mask (exclude central low-frequency region)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        center_radius = min(rows, cols) // 4
        cv2.circle(mask, (ccol, crow), center_radius, 0, -1)
        
        # Extract high-frequency components
        high_freq = fshift * mask
        
        # Calculate high-frequency energy
        high_freq_energy = np.sum(np.abs(high_freq))
        total_energy = np.sum(np.abs(fshift))
        
        # Calculate high-frequency ratio (value between 0~1)
        ratio = high_freq_energy / (total_energy + 1e-10)
        
        return ratio

    def _adjust_points_by_frequency(self, image, coords):
        """
        Adjust the number of points for each quadrant based on the high-frequency ratio.
        
        Args:
            sections: List of split image coordinates
            
        Returns:
            List of adjusted point counts
        """
        if isinstance(image, torch.Tensor):
            img_t = image.detach().cpu()
            # If shape is (C,H,W), convert to (H,W,C)
            if img_t.ndim == 3 and img_t.shape[0] in (1, 3):
                img_t = img_t.permute(1, 2, 0)
            image_np = img_t.numpy()
        else:
            image_np = image

        # Calculate high-frequency ratio for each quadrant
        freq_ratios = []
        for xs, ys, xe, ye in coords:
            patch = image_np[ys:ys+ye, xs:xs+xe, :]
            ratio = self._calculate_high_frequency_ratio(patch)
            freq_ratios.append(ratio)
        
        # Sum of high-frequency ratios
        total_ratio = sum(freq_ratios)
        
        # Base number of points (1/4 of the total number of points)
        base_points = self.num_init
        
        # Adjust number of points for each quadrant
        adjusted_points = []
        for ratio in freq_ratios:
            # Adjust number of points based on high-frequency ratio (ensure at least 1)
            points = max(1, int(base_points * (ratio / (total_ratio + 1e-10))))
            adjusted_points.append(points)
        
        # Add more points if the total is less than expected
        total_points = sum(adjusted_points)
        print("base_points - total_points: ", base_points - total_points)
        while total_points < base_points:
            i = random.randint(0, (self.level ** 2) - 1)
            adjusted_points[i] += 1
            total_points += 1
        
        print(f"freq_ratios: {freq_ratios}")
        print(f"adjusted_points: {adjusted_points}")
        
        return adjusted_points
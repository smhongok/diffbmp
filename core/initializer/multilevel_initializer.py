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
        I_hat = self.renderer.render_image(*P_0)
        I_1 = self.renderer.over(I_hat, I_bg)
        
        # 4. Multi-level refinement
        # Split into level**2 sections
        # Adjust number of points by frequency
        sections = self._split_into_level(I_tar, self.level)
        adjusted_points = self._adjust_points_by_frequency(sections)
        for bbox_k, num_points in zip(sections, adjusted_points):
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
            
        return P_all

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
        return resized

    def _calculate_high_frequency_ratio(self, image):
        """
        이미지의 고주파 성분 비율을 계산합니다.
        
        Args:
            image: 분석할 이미지 (numpy array)
            
        Returns:
            고주파 성분 비율 (0~1 사이 값)
        """
        # 이미지가 3차원(컬러)인 경우 그레이스케일로 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 이미지가 float 타입인 경우 uint8로 변환
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # FFT 변환
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 주파수 마스크 생성 (중앙 저주파 영역 제외)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        center_radius = min(rows, cols) // 4
        cv2.circle(mask, (ccol, crow), center_radius, 0, -1)
        
        # 고주파 성분 추출
        high_freq = fshift * mask
        
        # 고주파 에너지 계산
        high_freq_energy = np.sum(np.abs(high_freq))
        total_energy = np.sum(np.abs(fshift))
        
        # 고주파 비율 계산 (0~1 사이 값)
        ratio = high_freq_energy / (total_energy + 1e-10)
        
        return ratio

    def _adjust_points_by_frequency(self, sections):
        """
        각 사분면의 고주파 비율에 따라 점의 개수를 조정합니다.
        
        Args:
            sections: 분할 이미지 좌표 리스트
            
        Returns:
            조정된 점의 개수 리스트
        """
        # 각 사분면의 고주파 비율 계산
        freq_ratios = []
        for section in sections:
            ratio = self._calculate_high_frequency_ratio(section[0].cpu().numpy())
            freq_ratios.append(ratio)
        
        # 고주파 비율 합계
        total_ratio = sum(freq_ratios)
        
        # 기본 점 개수 (전체 점 개수의 1/4)
        base_points = max(1, int(self.num_init * self.per_quad_frac * (self.num_levels ** 2)))
        
        # 각 사분면별 점 개수 조정
        adjusted_points = []
        for ratio in freq_ratios:
            # 고주파 비율에 따라 점 개수 조정 (최소 1개는 보장)
            points = max(1, int(base_points * (ratio / (total_ratio + 1e-10))))
            adjusted_points.append(points)
        
        # 점 개수가 적으면 더 많이 생성
        total_points = sum(adjusted_points)
        print("base_points - total_points: ", base_points - total_points)
        while total_points < base_points:
            i = random.randint(0, (self.num_levels ** 2) - 1)
            adjusted_points[i] += 1
            total_points += 1
        
        print(f"freq_ratios: {freq_ratios}")
        print(f"adjusted_points: {adjusted_points}")
        
        return adjusted_points
import torch
import gc
from typing import Tuple, List
from core.initializer.base_initializer import BaseInitializer
from core.renderer.vector_renderer import VectorRenderer
import numpy as np
import cv2
import random

class MultiLevelInitializer(BaseInitializer):
    """
    A multi-level initializer that applies SVG splatting at multiple levels:
      - Level 0: Divide into 4 overlapping quadrants, initialize & render each, then blend.
      - Levels 1..N-1: On the residual (target – current reconstruction), initialize & render full image.
    """
    def __init__(self,
                 num_init=100, alpha=0.3, min_distance=20,
                 peak_threshold=0.5, radii_min=2, radii_max=None,
                 v_init_bias=-5.0, v_init_slope=0.0,
                 keypoint_extracting=False, debug_mode=False):
        super().__init__(num_init, alpha, min_distance,
                          peak_threshold, radii_min,
                          radii_max, v_init_bias, v_init_slope,
                          keypoint_extracting, debug_mode)
        self.overlap_pixels = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.num_levels = 4
        #self.level_fracs = [0.2, 0.2, 0.2, 0.2]  
        #self.per_quad_frac = self.level_fracs[0] / 4
        self.num_levels = 2
        self.level_fracs = [0.4, 0.4]  
        self.per_quad_frac = self.level_fracs[0] / self.num_levels
        self.residual_frac = 0.2
        self.base_init_mode = "svgsplat"

    def _divide_into_quadrants(self, image: torch.Tensor):
        H, W, C = image.shape
        h2, w2 = H // 2, W // 2
        o = self.overlap_pixels
        coords = [
            (0,      h2+o, 0,      w2+o),
            (0,      h2+o, w2-o,   W),
            (h2-o,   H,    0,      w2+o),
            (h2-o,   H,    w2-o,   W),
        ]
        quads = []
        for ys, ye, xs, xe in coords:
            quads.append((image[ys:ye, xs:xe].clone(), (ys, ye, xs, xe)))
        return quads

    def _merge_quadrants(self,
                          rendered_quads: List[Tuple[torch.Tensor, Tuple[int,int,int,int]]],
                          original_shape: Tuple[int,int,int]) -> torch.Tensor:
        H, W, C = original_shape
        merged = torch.zeros((H, W, C), device=self.device)
        weight = torch.zeros((H, W), device=self.device)

        for img_q, (ys, ye, xs, xe) in rendered_quads:
            # Get actual dimensions of the quadrant image
            hq, wq, _ = img_q.shape
            
            # Debug prints to identify the issue
            print(f"Quadrant shape: {img_q.shape}, Region: ({ys}:{ye}, {xs}:{xe})")
            print(f"Expected region size: ({ye-ys}, {xe-xs})")
            
            # Create weight mask with the exact same dimensions as the quadrant image
            wm = torch.ones((hq, wq), device=self.device)
            o = self.overlap_pixels
            
            # feather edges
            if ys>0:
                for i in range(o): wm[i,:]   *= (i/o)
            if ye< H:
                for i in range(o): wm[-1-i,:]*= (i/o)
            if xs>0:
                for i in range(o): wm[:,i]   *= (i/o)
            if xe< W:
                for i in range(o): wm[:,-1-i]*= (i/o)

            # Expand weight mask to match image channels
            wm = wm.unsqueeze(-1).expand(-1,-1,C)
            
            # Check if dimensions match
            if wm.shape[0] != img_q.shape[0] or wm.shape[1] != img_q.shape[1]:
                print(f"Dimension mismatch: wm={wm.shape}, img_q={img_q.shape}")
                # Resize weight mask to match image dimensions if needed
                wm = wm[:img_q.shape[0], :img_q.shape[1], :]
            
            # Add weighted quadrant to merged image
            merged[ys:ye, xs:xe] += img_q * wm
            weight[ys:ye, xs:xe] += wm[:,:,0]

        # Normalize by weights
        weight = weight.clamp(min=1e-8).unsqueeze(-1).expand(-1,-1,C)
        return merged / weight
    
    def _render_image(self, 
                     renderer: VectorRenderer, 
                     params: Tuple[torch.Tensor, ...], 
                     bmp_image: torch.Tensor,
                     target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Render an image using the given parameters.
        
        Args:
            renderer: Vector renderer
            params: Rendering parameters
            bmp_image: Base bitmap image
            target_size: Optional target size for the rendered image (H, W)
            
        Returns:
            Rendered image tensor
        """
        x, y, r, v, theta, c = params
        
        # Generate masks
        cached_masks = renderer._batched_soft_rasterize(
            bmp_image, x, y, r, theta,
            sigma=0.0
        )
        
        # Render image
        rendered = renderer.render(cached_masks, v, c)
        
        # Resize if target size is provided and different from current size
        if target_size is not None:
            H, W = target_size
            if rendered.shape[0] != H or rendered.shape[1] != W:
                # Use interpolation to resize the rendered image
                rendered = torch.nn.functional.interpolate(
                    rendered.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)
        
        # Clear memory
        del cached_masks
        torch.cuda.empty_cache()
        
        return rendered
    
    def _random_initialize(self, image: torch.Tensor, num_init=None) -> Tuple[torch.Tensor, ...]:
        """
        Initialize parameters using random initialization.
        
        Args:
            image: Target image to match
            num_init: Number of circles to initialize (if None, use self.num_init)
            
        Returns:
            Tuple of initialized parameters (x, y, r, v, theta, c)
        """
        if num_init is None:
            num_init = self.num_init
            
        H, W, C = image.shape
        
        # Convert image to grayscale if it's color
        if C > 1:
            gray_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray_image = image[:, :, 0]
        
        # Normalize to [0, 1]
        gray_image = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min() + 1e-8)
        
        # Generate random positions
        x = torch.rand(num_init, device=self.device) * W
        y = torch.rand(num_init, device=self.device) * H
        
        # Generate random radii
        if self.radii_max is None:
            self.radii_max = min(H, W) // 4
        
        r = torch.rand(num_init, device=self.device) * (self.radii_max - self.radii_min) + self.radii_min
        
        # Generate random angles
        theta = torch.rand(num_init, device=self.device) * 2 * np.pi
        
        # Initialize v based on image intensity at the positions
        v = torch.zeros(num_init, device=self.device)
        for i in range(num_init):
            x_idx = min(int(x[i]), W-1)
            y_idx = min(int(y[i]), H-1)
            v[i] = self.v_init_bias + self.v_init_slope * gray_image[y_idx, x_idx]
        
        # Initialize c based on image color at the positions
        c = torch.zeros(num_init, C, device=self.device)
        for i in range(num_init):
            x_idx = min(int(x[i]), W-1)
            y_idx = min(int(y[i]), H-1)
            c[i] = image[y_idx, x_idx]
        
        return x, y, r, v, theta, c

    def _structure_aware_initialize(self, I_target, num_init):
        """
        Initialize parameters using structure-aware approach.
        """
        
        device = I_target.device
        # Extract image dimensions
        if I_target.ndim == 3:
            H, W, _ = I_target.shape
            try:
                I_color = I_target.cpu().numpy()            # (H,W,3), 0~1
            except:
                I_color = I_target.detach().cpu().numpy()            # (H,W,3), 0~1
        else:
            H, W = I_target.shape
            gray = np.expand_dims(I_np / 255.0, axis=-1)
            I_color = np.repeat(gray, 3, axis=-1)

        # Convert to grayscale for structure analysis
        if isinstance(I_target, torch.Tensor):
            try:
                I_np = I_target.cpu().numpy()
            except:
                I_np = I_target.detach().cpu().numpy()
            if I_np.ndim == 3:
                I_np = np.mean(I_np, axis=2)
        else:
            I_np = I_target
            if I_np.ndim == 3:
                I_np = cv2.cvtColor(I_np, cv2.COLOR_RGB2GRAY)
            
        # Ensure image is in correct format
        if I_np.dtype != np.uint8:
            I_np = (I_np * 255).astype(np.uint8)

        # Use structure-aware initialization
        edges = cv2.Canny(I_np, 100, 200)
        grad_y, grad_x = np.gradient(I_np.astype(np.float32))
        
        # Start with ORB points
        num_kp = 0
        if self.keypoint_extracting:
            orb = cv2.ORB_create(nfeatures=num_init, scaleFactor=1.2, nlevels=8, edgeThreshold=15, firstLevel=0, WTA_K=2, patchSize=31, fastThreshold=20)
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
        
        # Apply structure-aware techniques
        densified_pts = self.find_best_densification(edges, num_init)
        adjusted_pts = self.structure_aware_adjustment(densified_pts, grad_x, grad_y)
        print("len(densified_pts): ", len(densified_pts), ", len(adjusted_pts): ", len(adjusted_pts))

        # Color initialization
        idx_x = np.clip(np.round(adjusted_pts[:, 0]).astype(int), 0, W - 1)
        idx_y = np.clip(np.round(adjusted_pts[:, 1]).astype(int), 0, H - 1)
        c_init = I_color[idx_y, idx_x]                  # (N,3) float32

        # Add slight noise for parameter diversity
        c_init += np.random.normal(0.0, 0.02, c_init.shape)
        c_init = np.clip(c_init, 0.0, 1.0)              # Safe clip
        
        # Sample and sort radius
        num_points = adjusted_pts.shape[0]
        r_np = np.random.rand(num_points) * min(H, W) / 4 + self.radii_min   # (N,)
        sort_idx = np.argsort(r_np)           # Ascending (+) → larger r later

        # Reorder coordinates, colors, and radii in the same order
        adjusted_pts = adjusted_pts[sort_idx]
        r_np = r_np[sort_idx]
        c_init = c_init[sort_idx]

        # Convert to tensors
        x = torch.tensor(adjusted_pts[:, 0], dtype=torch.float32, device=device, requires_grad=True)
        y = torch.tensor(adjusted_pts[:, 1], dtype=torch.float32, device=device, requires_grad=True)
        r = torch.tensor(r_np, dtype=torch.float32, device=device, requires_grad=True)

        # Initialize opacity v (layer matching)
        rank = torch.linspace(0.0, 1.0, steps=num_points, device=device)     # 0(bottom)→1(top)
        v = (self.v_init_bias + self.v_init_slope * rank).clone().detach()
        v += torch.empty_like(v).normal_(mean=0.0, std=0.05)
        v.requires_grad_(True)

        theta = torch.rand(num_points, device=device, requires_grad=True) * 2 * np.pi
        c = torch.tensor(c_init, dtype=torch.float32,
                         device=device, requires_grad=True)
        print("len(x): ", len(x))
        
        return x, y, r, v, theta, c

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

    def _adjust_points_by_frequency(self, quads):
        """
        각 사분면의 고주파 비율에 따라 점의 개수를 조정합니다.
        
        Args:
            quads: 사분면 이미지와 좌표 리스트
            
        Returns:
            조정된 점의 개수 리스트
        """
        # 각 사분면의 고주파 비율 계산
        freq_ratios = []
        for quad_img, _ in quads:
            ratio = self._calculate_high_frequency_ratio(quad_img.cpu().numpy())
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

    def initialize(self, target_image: torch.Tensor) -> Tuple[torch.Tensor,...]:
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        H, W, C = target_image.shape
        renderer = VectorRenderer((H, W), device=self.device)
        bmp_whole = torch.ones((H, W), device=self.device)

        # Accumulate all levels' parameters here
        all_x, all_y, all_r = [], [], []
        all_v, all_theta, all_c = [], [], []

        # CURRENT reconstruction starts at zero
        current_recon = torch.zeros_like(target_image)

        # --- Level 0: quadrants ---
        quads = self._divide_into_quadrants(target_image)
        rendered_quads = []
        
        # 고주파 비율에 따라 각 사분면별 점 개수 조정
        per_quad_points = self._adjust_points_by_frequency(quads)

        for i, (quad_img, (ys, ye, xs, xe)) in enumerate(quads):
            # Get the expected size for this quadrant
            expected_h, expected_w = ye - ys, xe - xs
            
            # 고주파 비율에 따라 조정된 점 개수 사용
            per_quad = per_quad_points[i]
            
            # initialize this quadrant
            if self.base_init_mode == "svgsplat":
                x, y, r, v, theta, c = self._structure_aware_initialize(quad_img, num_init=per_quad)
            else:
                x, y, r, v, theta, c = self._random_splat_params(per_quad, ys, xs, expected_h, expected_w, self.device)
            # shift into full-image coords
            x = x + xs; y = y + ys

            # render this quadrant (local bmp)
            bmp_q = torch.ones((quad_img.shape[0], quad_img.shape[1]), device=self.device)
            rendered_q = self._render_image(
                renderer, 
                (x, y, r, v, theta, c), 
                bmp_q,
                target_size=(expected_h, expected_w)
            )

            rendered_quads.append((rendered_q, (ys, ye, xs, xe)))

            # collect params
            all_x.append(x); all_y.append(y); all_r.append(r); all_v.append(v); all_theta.append(theta); all_c.append(c)

        # blend quadrants back into full image
        merged = self._merge_quadrants(rendered_quads, (H, W, C))
        current_recon = merged

        # stop if residual is negligible
        residual = (target_image - current_recon).clamp(0.0,1.0)
        if residual.norm() >= self.residual_frac:
            # primitives for this level: decay by quadrant_scale_factor each level
            nr = max(1, int(self.num_init * self.residual_frac))
            if self.base_init_mode == "svgsplat":
                x, y, r, v, theta, c = self._structure_aware_initialize(residual, num_init=nr)
            else:
                x, y, r, v, theta, c = self._random_splat_params(residual, num_init=nr)

            # render full-resolution residual primitives
            rendered_res = self._render_image(renderer, (x, y, r, v, theta, c), bmp_whole)
            current_recon = (current_recon + rendered_res).clamp(0.0, 1.0)

            # collect these level's params
            all_x.append(x); all_y.append(y); all_r.append(r); all_v.append(v); all_theta.append(theta); all_c.append(c)

        # concatenate all
        x_final     = torch.cat(all_x,     dim=0)
        y_final     = torch.cat(all_y,     dim=0)
        r_final     = torch.cat(all_r,     dim=0)
        v_final     = torch.cat(all_v,     dim=0)
        theta_final = torch.cat(all_theta, dim=0)
        c_final     = torch.cat(all_c,     dim=0)
        print(f"x_final: {x_final.shape}, y_final: {y_final.shape}, r_final: {r_final.shape}, v_final: {v_final.shape}, theta_final: {theta_final.shape}, c_final: {c_final.shape}")

        return x_final, y_final, r_final, v_final, theta_final, c_final
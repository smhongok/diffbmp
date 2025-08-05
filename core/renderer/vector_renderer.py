from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch import nn
import torch.utils.checkpoint as cp
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
from util.utils import gaussian_blur, make_batch_indices
import os
import gc
import pkg_resources
from collections import defaultdict
import datetime
from PIL import Image
import tempfile
import subprocess
import glob
import wandb

class VectorRenderer:
    """
    A class for rendering vector graphics using differentiable primitives.
    This class handles the core rendering functionality including mask generation,
    alpha compositing, and parameter optimization.
    """
    def __init__(self, 
                 canvas_size: Tuple[int, int],
                 S: torch.Tensor,
                 alpha_upper_bound: float = 0.5,
                 device: str = 'cuda',
                 use_fp16: bool = False,
                 gamma: float = 1.0,
                 output_path: str = None):
        """
        Initialize the vector renderer.
        
        Args:
            canvas_size: Tuple of (height, width) for the output canvas
            alpha_upper_bound: Maximum alpha value for rendering (default: 0.5)
            device: Device to use for computation ('cuda' or 'cpu')
            use_fp16: Whether to use half precision (FP16) for memory efficiency
        """
        self.H, self.W = canvas_size
        self.alpha_upper_bound = alpha_upper_bound
        self.device = device
        self.use_checkpointing = False
        self.use_fp16 = use_fp16
        self.gamma = gamma
        self.output_path = output_path
        # Convert S to appropriate precision during initialization
        if self.use_fp16:
            self.S = S.to(dtype=torch.float16)
        else:
            self.S = S
        
        # Pre-compute pixel coordinates
        self.X, self.Y = self._create_coordinate_grid()
        
        # Initialize video creation tracking
        self.saved_frames = []
        
        # Initialize wandb
        wandb.login(key="08d57452958261449694f652099a45203ab23a2e")
        wandb.init(entity="svgsplat",project="svgsplat", name=f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpointing = True
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_checkpointing = False
    
    def _create_coordinate_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create the coordinate grid for rendering."""
        if self.use_fp16:
            X, Y = torch.meshgrid(
                torch.arange(self.W, device=self.device, dtype=torch.float16),
                torch.arange(self.H, device=self.device, dtype=torch.float16),
                indexing='xy'
            )
        else:
            X, Y = torch.meshgrid(
                torch.arange(self.W, device=self.device),
                torch.arange(self.H, device=self.device),
                indexing='xy'
            )
        return X.unsqueeze(0), Y.unsqueeze(0)  # (1, H, W)
    
    def _batched_soft_rasterize(self,
                               x: torch.Tensor,
                               y: torch.Tensor,
                               r: torch.Tensor,
                               theta: torch.Tensor,
                               sigma: float = 0.0) -> torch.Tensor:
        """
        Generate soft masks for each primitive, processing in smaller chunks to save memory.

        Now supports a sequence of p different primitives in self.S of shape (p, H, W),
        assigning them periodically to the B instances.

        Args:
            x, y:       [N] position coordinates for N shapes
            r:          [N] scales
            theta:      [N] rotations
            sigma:      Gaussian blur std
        Returns:
            masks:     [N, H, W]  (one soft mask per shape)
        """
        context = autocast('cuda') if self.use_fp16 else nullcontext()
        with context:
            B = len(x)
            _, H, W = self.X.shape
            target_dtype = torch.float16 if self.use_fp16 else torch.float32

            # Prepare the bitmap(s): either single template or a sequence of p templates
            if sigma > 0.0:
                bmp = self.S.unsqueeze(0)       # -> [1, p, H, W] or [1, H, W]
                bmp = gaussian_blur(bmp, sigma)
                bmp_image = bmp.squeeze(0).to(dtype=target_dtype).contiguous()
            else:
                bmp_image = self.S.to(dtype=target_dtype)

            # Expand coordinates and parameters
            X_exp = self.X.expand(B, H, W)
            Y_exp = self.Y.expand(B, H, W)
            x_exp = x.view(B, 1, 1).expand(B, H, W)
            y_exp = y.view(B, 1, 1).expand(B, H, W)
            r_exp = r.view(B, 1, 1).expand(B, H, W)

            # Normalize and rotate positions
            pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            R_inv = torch.zeros(B, 2, 2, device=self.device)
            R_inv[:, 0, 0] = cos_t; R_inv[:, 0, 1] = sin_t
            R_inv[:, 1, 0] = -sin_t; R_inv[:, 1, 1] = cos_t
            uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
            grid = uv.permute(0, 2, 3, 1)  # (B, H, W, 2)

            # Build bmp_exp: one bitmap per instance, cycling through p if provided
            if bmp_image.dim() == 2:
                # single primitive template [H, W]
                bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
            elif bmp_image.dim() == 3:
                # sequence of p templates [p, H, W]
                p = bmp_image.size(0)
                # periodic assignment
                idx = torch.arange(B, device=self.device, dtype=torch.long) % p  # [B]
                idx = idx.flip(0)
                bmp_sel = bmp_image[idx, :, :]            # [B, H, W]
                bmp_exp = bmp_sel.unsqueeze(1).contiguous()  # [B, 1, H, W]
            else:
                raise ValueError(f"Unsupported self.S shape: {bmp_image.shape}")

            # Sample masks via grid_sample (with optional checkpointing)
            if self.use_checkpointing:
                def grid_fn(img, g):
                    return F.grid_sample(img, g, mode='bilinear', padding_mode='zeros', align_corners=True)
                sampled = cp.checkpoint(grid_fn, bmp_exp, grid, use_reentrant=False)
            else:
                sampled = F.grid_sample(bmp_exp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            # Return single-channel masks
            return sampled.squeeze(1)  # (B, H, W)
    
    def _transmit_over(self, m: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        
        """
        Vectorized weighted-sum alpha compositing (Porter–Duff “Over”).
        Inputs:
        m: (N, H, W, 3) premultiplied colors
        a: (N, H, W) alphas
        Outputs:
        comp_m: (H, W, 3) composited RGB before blending white background
        comp_a: (H, W)   composited alpha
        """
        # 1. Compute transmittance T[k] = prod_{i<k}(1 - a[i]) for k=0..N
        #    T has shape (N+1, H, W), with T[0] = 1
        alpha_inv = 1.0 - a                               # (N, H, W)
        T_partial = torch.cumprod(alpha_inv, dim=0)       # (N, H, W)
        ones = torch.ones_like(T_partial[0:1])            # (1, H, W)
        T = torch.cat([ones, T_partial], dim=0)           # (N+1, H, W)

        # 2. Composite alpha: A_out = 1 - T[N]
        comp_a = 1.0 - T[-1]                               # (H, W)

        # 3. Composite color before blending white background:
        #    C_out = sum_{k=0..N-1} (m[k] * T[k]) + T[N]*white
        #    Here m[k] already = a[k] * c[k], so weight = T[k]
        #    Weighted sum is sum_{k=0..N-1} (m[k] * T[k])
        weights = T[:-1].unsqueeze(-1)                     # (N, H, W, 1)
        weighted = m * weights                            # (N, H, W, 3)
        comp_m = weighted.sum(dim=0)                     # (H, W, 3)

        return comp_m, comp_a

        # as a lab intern who edited this code without deep knowledge about computing, 
        # carfully suggest this method because
        # 1. Similar PSNR, increased SSIM, VIF, and lowered LPIPS compared to the original method
        # 2. significant decrease of total cost time


    
    def _get_checkpoint_kwargs(self):
        """
        Returns the correct checkpoint keyword arguments based on the PyTorch version.
        Older versions don't support use_reentrant.
        """
        # Check PyTorch version
        torch_version = pkg_resources.get_distribution("torch").version
        major, minor = map(int, torch_version.split('.')[:2])
        
        # use_reentrant supported only in PyTorch 1.12 and later
        if (major > 1) or (major == 1 and minor >= 12):
            return {"use_reentrant": False}
        else:
            # No such option in earlier versions
            return {}
    
    def _safe_checkpoint(self, func, *tensors):
        """
        Wrapper function to safely perform checkpointing regardless of PyTorch version
        """
        kwargs = self._get_checkpoint_kwargs()
        return torch.utils.checkpoint.checkpoint(func, *tensors, **kwargs)
    
    def render_with_white_bg(
            self,
            cached_masks: torch.Tensor,
            v: torch.Tensor,
            c: torch.Tensor,
            return_alpha: bool = False
        ):
        """
        Render the final image (optionally alpha).

        Args:
            cached_masks : (N, H, W)   – pre-computed soft masks
            v            : (N,)        – visibility logits
            c            : (N, 3)      – RGB logits
            return_alpha : If True  → (rgb, alpha) returned
                        If False → rgb only returned

        Returns
        -------
        - rgb  : (H, W, 3)  (always)
        - alpha: (H, W, 1)  (optional, when return_alpha=True)
        """
        context = autocast('cuda') if self.use_fp16 else nullcontext()
        
        with context:    
            target_dtype = torch.float16 if self.use_fp16 else torch.float32
            cached_masks = cached_masks
            v = v
            c = c
            N = v.shape[0]

            # 1. per-primitive alpha & color
            v_alpha = self.alpha_upper_bound * torch.sigmoid(v).view(N, 1, 1)
            a = v_alpha * cached_masks                     # (N, H, W)
            c_eff = torch.sigmoid(c).view(N, 1, 1, 3)      # (N, 1, 1, 3)
            
            # Create color tensor with minimal memory overhead
            m = a.unsqueeze(-1) * c_eff                    # (N, H, W, 3)

            # 2. Porter–Duff reduction (tree)
            if self.use_checkpointing:
                comp_m, comp_a = self._safe_checkpoint(
                    lambda mm, aa: self._transmit_over(mm, aa),
                    m, a
                )
            else:
                comp_m, comp_a = self._transmit_over(m, a)    
            
            # Free large tensors as soon as possible if in FP16 mode
            if self.use_fp16:
                del m, a, v_alpha, c_eff
            
            if return_alpha:
                # (H, W) → (H, W, 1) to match for broadcasting
                return comp_m, comp_a.unsqueeze(-1)

            # 3. Composite with white background
            ones = torch.ones_like(comp_m)
            final = comp_m + (1.0 - comp_a).unsqueeze(-1) * ones
            
            # Free temporary tensors if in FP16 mode
            if self.use_fp16:
                del comp_m, comp_a, ones
            
            return final
 

    def render(
            self,
            cached_masks: torch.Tensor,
            v: torch.Tensor,
            c: torch.Tensor,
            return_alpha: bool = False
        ):
        """
        Render the final image (optionally alpha).

        Args:
            cached_masks : (N, H, W)   – pre-computed soft masks
            v            : (N,)        – visibility logits
            c            : (N, 3)      – RGB logits
            return_alpha : If True  → (rgb, alpha) returned
                        If False → rgb only returned

        Returns
        -------
        - rgb  : (H, W, 3)  (always)
        - alpha: (H, W, 1)  (optional, when return_alpha=True)
        """
        context = autocast('cuda') if self.use_fp16 else nullcontext()
        
        with context:    
            target_dtype = torch.float16 if self.use_fp16 else torch.float32
            cached_masks = cached_masks
            v = v
            c = c
            N = v.shape[0]

            # 1. per-primitive alpha & color
            v_alpha = self.alpha_upper_bound * torch.sigmoid(v).view(N, 1, 1)
            a = v_alpha * cached_masks                     # (N, H, W)
            c_eff = torch.sigmoid(c).view(N, 1, 1, 3)      # (N, 1, 1, 3)
            
            # Create color tensor with minimal memory overhead
            m = a.unsqueeze(-1) * c_eff                    # (N, H, W, 3)

            # 2. Porter–Duff reduction (tree)
            if self.use_checkpointing:
                comp_m, comp_a = self._safe_checkpoint(
                    lambda mm, aa: self._transmit_over(mm, aa),
                    m, a
                )
            else:
                comp_m, comp_a = self._transmit_over(m, a)    
            
            # Free large tensors as soon as possible if in FP16 mode
            if self.use_fp16:
                del m, a, v_alpha, c_eff
            
            if return_alpha:
                # (H, W) → (H, W, 1) to match for broadcasting
                return comp_m, comp_a.unsqueeze(-1)

            # 3. We don't composite with white background
            final = comp_m
            
            # Free temporary tensors if in FP16 mode
            if self.use_fp16:
                del comp_m, comp_a
            
            return final
    
    def render_export_mp4(
        self,
        cached_masks: torch.Tensor,  # (N, H, W)
        v: torch.Tensor,             # (N,)
        c: torch.Tensor,             # (N, 3)
        video_path: str,
        fps: int = 60
    ):
        """
        Sequential-over compositing 으로 primitive를 한 장씩 쌓아가며
        중간 이미지를 MP4로 바로 기록.

        Args:
        cached_masks: (N, H, W)   – soft masks
        v           : (N,)        – visibility logits
        c           : (N, 3)      – RGB logits
        video_path  : 저장될 MP4 경로
        fps         : 프레임레이트
        """
        context = autocast('cuda') if self.use_fp16 else nullcontext()

        with context:
            # 1.1 per-primitive alpha & color
            v_alpha = self.alpha_upper_bound * torch.sigmoid(v).view(-1, 1, 1)    # (N,1,1)
            a_all   = v_alpha * cached_masks                                       # (N,H,W)
            c_eff   = torch.sigmoid(c).view(-1, 1, 1, 3)                           # (N,1,1,3)
            m_all   = a_all.unsqueeze(-1) * c_eff                                  # (N,H,W,3)

        # --- 2. 비디오 초기화를 위한 첫 프레임 계산 ---
        N, H, W = a_all.shape
        
        # 임시 파일 생성 (OpenCV 중간 결과용)
        temp_video = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
                temp_video = tmp.name

            # sequential over: comp_m, comp_a 초기값 (맨 아래층 = 첫 프리미티브)
            comp_m = m_all[0]   # (H,W,3)
            comp_a = a_all[0]   # (H,W)
            # 흰 배경과 합성
            first = comp_m + (1.0 - comp_a).unsqueeze(-1)
            first_np = (first.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
            # OpenCV는 BGR 순서
            first_bgr = first_np[..., ::-1]

            # 임시 비디오 작성 (무손실 코덱 사용)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 무손실 중간 포맷
            writer = cv2.VideoWriter(temp_video, fourcc, fps, (W, H))

            if not writer.isOpened():
                raise RuntimeError("Failed to open video writer")

            # 첫 프레임 기록
            writer.write(first_bgr)

            # --- 3. 나머지 프리미티브 순차 합성 & 기록 ---
            for i in range(1, N):
                with context:
                    # comp = comp + (1 - comp_a) * next
                    comp_m = comp_m + (1.0 - comp_a).unsqueeze(-1) * m_all[i]
                    comp_a = comp_a + (1.0 - comp_a) * a_all[i]

                # 흰 배경과 합성
                frame = comp_m + (1.0 - comp_a).unsqueeze(-1)
                frame_np = (frame.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
                frame_bgr = frame_np[..., ::-1]
                writer.write(frame_bgr)

            writer.release()

            # --- 4. FFmpeg으로 웹 호환 H.264 MP4 변환 ---
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # 덮어쓰기 허용
                '-i', temp_video,
                
                # 비디오 코덱 설정 (웹 호환)
                '-c:v', 'libx264',
                '-profile:v', 'baseline',  # 최대 브라우저 호환성
                '-level', '3.1',
                '-crf', '23',              # 품질 (18-28, 낮을수록 고품질)
                
                # 픽셀 포맷 (필수)
                '-pix_fmt', 'yuv420p',
                
                # 웹 스트리밍 최적화
                '-movflags', '+faststart',
                
                # 오디오 없음 (비디오만)
                '-an',
                
                video_path
            ]
            
            # FFmpeg 실행
            result = subprocess.run(
                ffmpeg_cmd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            if result.returncode != 0:
                print(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
            
            print(f"Saved web-compatible H.264 MP4 to {video_path}")
            
        finally:
            # 임시 파일 정리
            if temp_video and os.path.exists(temp_video):
                try:
                    os.unlink(temp_video)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_video}: {e}")

        return frame  # Return the last frame for reference
    
    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between rendered and target images.
        This method should be overridden by subclasses to implement different loss functions.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement compute_loss")
    
    def initialize_parameters(self,
                            initializer: Any,
                            target_image: torch.Tensor,
                            target_binary_mask=None) -> Tuple[torch.Tensor, ...]:
        """
        Initialize parameters using the provided initializer.
        
        Args:
            initializer: Initializer object (e.g., StructureAwareInitializer)
            target_image: Target image to match
            
        Returns:
            Tuple of initialized parameters (x, y, r, v, theta, c)
        """
        # We have to modify this method for better performance with none background images,
        # since we don't have to splat primitives on the background(which is transparent) at initialization.        
        if target_image.shape[2] == 4:
            target_image = target_image[:, :, :3]  # Ignore alpha channel

        # Initialize from target image
        x, y, r, v, theta, c = initializer.initialize(target_image, target_binary_mask = target_binary_mask)
        
        # Convert to leaf tensors for optimization
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        r = r.detach().clone().requires_grad_(True)
        v = v.detach().clone().requires_grad_(True)
        theta = theta.detach().clone().requires_grad_(True)
        c = c.detach().clone().requires_grad_(True)
        
        return x, y, r, v, theta, c
    
    def optimize_parameters(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      r: torch.Tensor,
                      v: torch.Tensor,
                      theta: torch.Tensor,
                      c: torch.Tensor,
                      target_image: torch.Tensor,
                      opt_conf: Dict[str, Any],
                      target_binary_mask: Optional[torch.Tensor] = None,
                      target_dist_mask: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, ...]:
        """Optimize rendering parameters to match target image."""
        N = x.shape[0]
        self.sample_size = max(1, int(0.2 * N))  # 20% of primitives
        # Randomly sample indices
        self.sample_indices = torch.randperm(N, device=self.device)[:self.sample_size]

        # Use whole optimization (batch optimization removed as it's never used)
        if target_binary_mask is not None:
            return self._optimize_parameters_whole_no_bg(
            x, y, r, v, theta, c, target_image, opt_conf, target_binary_mask, target_dist_mask
        )
        else:
            return self._optimize_parameters_whole(
                x, y, r, v, theta, c, target_image, opt_conf
            )
        
    def _optimize_parameters_whole(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          r: torch.Tensor,
                          v: torch.Tensor,
                          theta: torch.Tensor,
                          c: torch.Tensor,
                          target_image: torch.Tensor,
                          opt_conf: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        Optimize the rendering parameters to match the target image.
        
        Args:
            x, y, r, v, theta, c: Initial parameters
            target_image: Target image to match
            opt_conf: Optimization configuration
            
        Returns:
            Tuple of optimized parameters (x, y, r, v, theta, c)
        """
        # Get optimization parameters from config
        num_iterations = opt_conf.get("num_iterations", 100)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Initial memory report
        if opt_conf.get("debug_memory", False):
            self.memory_report("Before optimization")
        
        # Mixed-precision scaler (only used if use_fp16 is True)
        scaler = GradScaler('cuda') if self.use_fp16 else None
        
        # Pre-calculate configurations
        blur_sigma = opt_conf.get("blur_sigma", 1.0)
        streaming_render = opt_conf.get("streaming_render", False) and self.use_fp16  # Only use streaming render in FP16 mode
        raster_chunk_size = opt_conf.get("raster_chunk_size", 20)
        
        # Create output directory for saving images if it doesn't exist
        save_image_intervals = [1, 5, 10, 20, 50, 100]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_path, exist_ok=True)
       
        # Create optimizers - separate xyr from the rest if doing sparsifying
        sparsify_conf = opt_conf.get("sparsifying", {})
        do_sparsify = sparsify_conf.get("do_sparsify", False)
        
        N = x.shape[0]  # Number of primitives

        # For Gaussian blur transition if enabled
        do_gaussian_blur = opt_conf.get("do_gaussian_blur", False)
        do_adapt_gaussian_blur = opt_conf.get("do_adapt_gaussian_blur", False)
        sigma_start = opt_conf.get("blur_sigma_start", 0.0)
        sigma_end = opt_conf.get("blur_sigma_end", 0.0)
        
        if do_sparsify:
            # Create separate optimizers for sparsification
            optimizer_xyr = torch.optim.Adam([
                {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            ])
            
            optimizer_rest = torch.optim.Adam([
                {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0/x.numel())},
                {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)}, # * (1000.0/x.numel())},
                {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
            ])
            
            # Create scheduler if decay is enabled
            do_decay = opt_conf.get("do_decay", False)
            sched_xyr = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_xyr, gamma=opt_conf.get("decay_rate", 0.99)) if do_decay else None
                
            # Initialize sparsification variables
            z = torch.zeros_like(v, requires_grad=False)
            lam = torch.zeros_like(v, requires_grad=False)
            iters_warmup = sparsify_conf.get("sparsify_warmup", num_iterations//3)
            sparsify_duration = sparsify_conf.get("sparsify_duration", num_iterations//3)
            sparsify_loss_coeff = sparsify_conf.get("sparsify_loss_coeff", 0.5)
            sparsifying_period = sparsify_conf.get("sparsifying_period", 20)
            sparsified_N = int(sparsify_conf.get("sparsified_N", int(0.6 * N)))
            
            # --- Dynamic Sparse Training (DST) configuration ---
            dst_enabled = sparsify_conf.get("dst_enabled", True)
            dst_period = sparsify_conf.get("dst_period", 10)  # DST update frequency
            dst_prune_frac = sparsify_conf.get("dst_prune_frac", 0.2)  # Prune ratio
            
            # Initialize initial mask (SET style)
            if dst_enabled:
                # Calculate active ratio (1 - sparsity)
                density = sparsified_N / N
                # Create initial random mask
                v_mask = torch.zeros_like(v, requires_grad=False)
                active_indices = torch.randperm(N, device=self.device)[:sparsified_N]
                v_mask[active_indices] = 1.0
                
                # Buffer for storing gradients for each parameter
                grad_history = {
                    'x': torch.zeros_like(x),
                    'y': torch.zeros_like(y),
                    'r': torch.zeros_like(r),
                    'v': torch.zeros_like(v),
                    'theta': torch.zeros_like(theta),
                    'c': torch.zeros_like(c)
                }
                
                # Apply initial mask
                print(f"Initialized DST with density {density:.2f} ({sparsified_N}/{N} primitives active)")
            
            assert sparsified_N < N, "sparsified_N must be less than N"
            assert num_iterations > iters_warmup + sparsify_duration, "num_iterations must be greater than warmup + duration"
                        
            optimizer = None  # We'll use the separate optimizers instead
        else:
            # Standard single optimizer if not doing sparsification
            param_groups = [
                {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
                {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0/x.numel())},
                {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)}, # * (1000.0/x.numel())},
                {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
            ]
            optimizer = torch.optim.Adam(param_groups)
            optimizer_xyr = None
            optimizer_rest = None
        
        print(f"Starting optimization, {num_iterations} iterations...")
        for epoch in tqdm(range(num_iterations)):
            # Reset gradients
            if do_sparsify:
                optimizer_xyr.zero_grad()
                optimizer_rest.zero_grad()
                
                # Current sigma for Gaussian blur
                if do_adapt_gaussian_blur:
                    sigma = sigma_start - (sigma_start - sigma_end) * (epoch / num_iterations)
                    print(f"Gaussian blur sigma: {sigma}")
                elif do_gaussian_blur:
                    sigma = blur_sigma #sigma_start * (1 - epoch / num_iterations) + sigma_end * (epoch / num_iterations)
                else:
                    sigma = 0.0
            else:
                optimizer.zero_grad()
                sigma = blur_sigma
            
            # Render image - use different approaches based on precision mode
            if self.use_fp16:
                with autocast('cuda'):
                    if opt_conf.get("multi_level", False):
                        rendered = self.render_with_white_bg(self.S, v, c)
                    else:
                        if streaming_render:
                            # Use standard rendering instead of inefficient streaming
                            cached_masks = self._batched_soft_rasterize(x, y, r, theta, sigma=sigma)
                            rendered = self.render_with_white_bg(cached_masks, v, c)
                        else:
                            # Generate masks (memory efficient)
                            cached_masks = self._batched_soft_rasterize(
                                x, y, r, theta,
                                sigma=sigma
                            )
                            
                            # Memory report after mask generation
                            if opt_conf.get("debug_memory", False) and epoch == 0:
                                self.memory_report("After mask generation")
                            
                            rendered = self.render_with_white_bg(cached_masks, v, c)
                            
                            # Save image at specified epochs
                            if opt_conf.get("save_epoch", False):
                                output_path = os.path.join(self.output_path, f"epoch_{epoch+1}_{timestamp}.jpg")
                                rendered_np = rendered.detach().cpu().numpy()
                                rendered_np = (rendered_np * 255).astype(np.uint8)
                                Image.fromarray(rendered_np).save(output_path)
                                self.saved_frames.append(output_path)
                                del rendered_np
                            
                            # Save reference for cleanup
                            cached_masks_ref = cached_masks
                            cached_masks = None
                    
                    # Memory report after rendering
                    if opt_conf.get("debug_memory", False) and epoch == 0:
                        self.memory_report("After rendering")
                    
                    # Compute loss
                    loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)
                    
                    # Add sparsification loss if needed
                    if do_sparsify and epoch >= iters_warmup and epoch < iters_warmup + sparsify_duration:
                        # Don't detach for gradient computation
                        _alpha = self.alpha_upper_bound * torch.sigmoid(v)
                        
                        # Use adaptive rho parameter
                        rho = sparsify_loss_coeff * (1.0 - (epoch - iters_warmup) / sparsify_duration)
                        
                        # Calculate loss including mask when DST is applied
                        if dst_enabled:
                            loss += 0.5 * rho * F.mse_loss(_alpha * v_mask, z - lam)
                        else:
                            loss += 0.5 * rho * F.mse_loss(_alpha, z - lam)
                
                    # Calculate gradients and update parameters
                    if do_sparsify:
                        loss.backward()
                        optimizer_xyr.step()
                        optimizer_rest.step()
                        if do_decay and sched_xyr is not None:
                            sched_xyr.step()
                    else:
                        # Standard update with scaler
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                    # Free memory immediately
                    del rendered
                    if self.use_fp16:
                        torch.cuda.empty_cache()
                    
            else:
                # Non-FP16 mode - standard approach without autocast or scaler
                if opt_conf.get("multi_level", False):
                    rendered = self.render_with_white_bg(self.S, v, c)
                else:
                    # Generate masks
                    cached_masks = self._batched_soft_rasterize(
                        x, y, r, theta,
                        sigma=sigma
                    )
                    
                    # Render with masks
                    rendered = self.render_with_white_bg(cached_masks, v, c)
                    
                    # Save image at specified epochs
                    if opt_conf.get("save_epoch", False):
                        output_path = os.path.join(self.output_path, f"epoch_{epoch+1}_{timestamp}.jpg")
                        rendered_np = rendered.detach().cpu().numpy()
                        rendered_np = (rendered_np * 255).astype(np.uint8)
                        Image.fromarray(rendered_np).save(output_path)
                        self.saved_frames.append(output_path)
                        del rendered_np
                
                # Compute loss
                loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)
                
                # Add sparsification loss if needed
                if do_sparsify and epoch >= iters_warmup and epoch < iters_warmup + sparsify_duration:
                    # Don't detach for gradient computation
                    _alpha = self.alpha_upper_bound * torch.sigmoid(v)
                    
                    # Use adaptive rho parameter
                    rho = sparsify_loss_coeff * (1.0 - (epoch - iters_warmup) / sparsify_duration)
                    
                    # Calculate loss including mask when DST is applied
                    if dst_enabled:
                        loss += 0.5 * rho * F.mse_loss(_alpha * v_mask, z - lam)
                    else:
                        loss += 0.5 * rho * F.mse_loss(_alpha, z - lam)
                
                # Calculate gradients and update parameters
                loss.backward()
                
                if do_sparsify:
                    optimizer_xyr.step()
                    optimizer_rest.step()
                    if do_decay and sched_xyr is not None:
                        sched_xyr.step()
                else:
                    optimizer.step()
                
            # Record loss value for next epoch logging
            loss_value = loss.item()
            
            # Log gradient statistics to wandb
            if opt_conf.get("log_gradients", False):
                self.log_gradient_statistics(x, y, r, v, theta, c, epoch)
            
            if self.use_fp16:
                del loss
                torch.cuda.empty_cache()
            
            # Memory report after cleanup
            if opt_conf.get("debug_memory", False) and epoch == 0:
                self.memory_report("After cleanup")
            
            # Clamp parameters (memory efficient approach)
            with torch.no_grad():
                x.clamp_(0, self.W)
                y.clamp_(0, self.H)
                r.clamp_(2, min(self.H, self.W) // 4)
                # theta.clamp_(0, 2 * np.pi)
                
                # Sparsification logic
                if do_sparsify:
                    if epoch >= iters_warmup and epoch <= iters_warmup + sparsify_duration:
                        # DST mask update
                        if dst_enabled and epoch % dst_period == 0:
                            with torch.no_grad():
                                # DST mask update
                                for param_name in ['v', 'x', 'y', 'r', 'theta', 'c']:
                                    param = locals()[param_name]
                                    if param.grad is not None:
                                        grad_history[param_name] = 0.9 * grad_history[param_name] + 0.1 * param.grad.abs()
                                
                                # 3. Hierarchical relationship analysis
                                # a) Size-based importance (larger ones are "parents" of smaller ones)
                                r_norm = r / r.max()
                                
                                # Final importance calculation: visibility(60%) + size(40%)
                                alpha_importance = self.alpha_upper_bound * torch.sigmoid(v.detach()) * r_norm
                                weights = torch.tensor([0.6, 0.4], device=self.device)
                                factors = torch.stack([alpha_importance, r_norm], dim=0)
                                alpha_importance = torch.matmul(weights, factors)
                                
                                # Remove low-importance parameters from active ones (prune)
                                k_prune = int(sparsified_N * dst_prune_frac)
                                active_idxs = (v_mask > 0).nonzero(as_tuple=True)[0]
                                if len(active_idxs) > 0:
                                    importance = alpha_importance[active_idxs]
                                    prune_candidates = active_idxs[torch.topk(importance, k=k_prune, largest=False).indices]
                                    v_mask[prune_candidates] = 0.0
                                
                                # Add considering both importance and gradients from inactive parameters (grow)
                                inactive_idxs = (v_mask == 0).nonzero(as_tuple=True)[0]
                                if len(inactive_idxs) > 0:
                                    # Combined score of gradient magnitude and visual importance
                                    grow_scores = 0.7 * grad_history['v'][inactive_idxs] + 0.3 * alpha_importance[inactive_idxs]
                                    grow_candidates = inactive_idxs[torch.topk(grow_scores, k=k_prune, largest=True).indices]
                                    v_mask[grow_candidates] = 1.0
                                
                                print(f"DST update - Active: {v_mask.sum().item()}/{N} primitives")
                        
                        # Regular ADMM z, λ update (periodically)
                        if epoch % sparsifying_period == 0:
                            # Compute alpha values from current parameters
                            _alpha = self.alpha_upper_bound * torch.sigmoid(v.detach())
                           
                            combined_score = _alpha 
                            
                            # Integration with DST: mask-based score calculation
                            if dst_enabled:
                                # Consider mask: only active parameters are candidates for selection
                                masked_score = combined_score * v_mask
                                keep_idx = torch.topk(masked_score, sparsified_N).indices
                            else:
                                # Traditional approach: select from all parameters
                                keep_idx = torch.topk(combined_score, sparsified_N).indices
                            
                            mask = torch.zeros_like(_alpha, dtype=torch.bool)
                            mask[keep_idx] = True
                            
                            # ADMM update
                            z.zero_()
                            if dst_enabled:
                                z[mask] = (_alpha * v_mask)[mask]  # Only update values with applied mask
                            else:
                                z[mask] = _alpha[mask]
                            lam += (_alpha - z)
                        
                        if epoch == iters_warmup + sparsify_duration:
                            # actual sparsification
                            # 1) helper: existing tensor -> trimmed tensor (maintains requires_grad)
                            def _prune(t):
                                t_new = t[keep_idx].clone().detach()
                                if t.requires_grad:
                                    t_new.requires_grad_(True)
                                return t_new
                            
                            # 2) Actually prune parameters and auxiliary variables
                            x, y, r, theta, v, c = map(_prune, (x, y, r, theta, v, c))
                            z, lam = map(_prune, (z, lam))
                            
                            # Also prune DST related variables
                            if dst_enabled:
                                v_mask = v_mask[keep_idx].clone()
                                for key in grad_history:
                                    if key == 'v':
                                        grad_history[key] = grad_history[key][keep_idx].clone()
                            
                            # 3) Create fresh optimizers after pruning without attempting state transfer
                            # This avoids tensor size mismatch issues
                            optimizer_xyr = torch.optim.Adam([
                                {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                                {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                                {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
                            ])
                            
                            optimizer_rest = torch.optim.Adam([
                                {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},
                                {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
                                {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
                            ])
                            
                            # Create scheduler if decay is enabled
                            sched_xyr = torch.optim.lr_scheduler.ExponentialLR(
                                optimizer_xyr, gamma=opt_conf.get("decay_rate", 0.99)) if do_decay else None

            if not streaming_render and not opt_conf.get("multi_level", False) and 'cached_masks_ref' in locals():
                del cached_masks_ref
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.4f}")

        # Final memory report
        if opt_conf.get("debug_memory", False):
            self.memory_report("After optimization")
        
        # Finish wandb run
        if opt_conf.get("log_gradients", False):
            wandb.finish()
        
        if opt_conf.get("save_epoch", False):
            video_path = os.path.join(self.output_path, f"optimization_progress_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            self.create_video_from_collected_frames(video_path, fps=30)
            print(f"Optimization progress video saved to: {video_path}")
            
        return x, y, r, v, theta, c

    def _optimize_parameters_whole_no_bg(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          r: torch.Tensor,
                          v: torch.Tensor,
                          theta: torch.Tensor,
                          c: torch.Tensor,
                          target_image: torch.Tensor,
                          opt_conf: Dict[str, Any],
                          target_binary_mask: Optional[torch.Tensor] = None,
                          target_dist_mask: Optional[torch.Tensor] = None,
                          ) -> Tuple[torch.Tensor, ...]:
        """
        Optimize the rendering parameters to match the target image.
        
        Args:
            x, y, r, v, theta, c: Initial parameters
            target_image: Target image to match
            opt_conf: Optimization configuration
            
        Returns:
            Tuple of optimized parameters (x, y, r, v, theta, c)
        """

        # Get optimization parameters from config
        num_iterations = opt_conf.get("num_iterations", 100)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Initial memory report
        if opt_conf.get("debug_memory", False):
            self.memory_report("Before optimization")
        
        # Mixed-precision scaler (only used if use_fp16 is True)
        scaler = GradScaler('cuda') if self.use_fp16 else None
        
        # Pre-calculate configurations
        blur_sigma = opt_conf.get("blur_sigma", 1.0)
        streaming_render = opt_conf.get("streaming_render", False) and self.use_fp16  # Only use streaming render in FP16 mode
        raster_chunk_size = opt_conf.get("raster_chunk_size", 20)
        
        # Create output directory for saving images if it doesn't exist
        save_image_intervals = [1, 5, 10, 20, 50, 100]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_path, exist_ok=True)
       
        # Create optimizers - separate xyr from the rest if doing sparsifying
        sparsify_conf = opt_conf.get("sparsifying", {})
        do_sparsify = sparsify_conf.get("do_sparsify", False)
        
        N = x.shape[0]  # Number of primitives

        # For Gaussian blur transition if enabled
        do_gaussian_blur = opt_conf.get("do_gaussian_blur", False)
        do_adapt_gaussian_blur = opt_conf.get("do_adapt_gaussian_blur", False)
        sigma_start = opt_conf.get("blur_sigma_start", 0.0)
        sigma_end = opt_conf.get("blur_sigma_end", 0.0)
        
        if do_sparsify:
            # Create separate optimizers for sparsification
            optimizer_xyr = torch.optim.Adam([
                {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            ])
            
            optimizer_rest = torch.optim.Adam([
                {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0/x.numel())},
                {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)}, # * (1000.0/x.numel())},
                {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
            ])
            
            # Create scheduler if decay is enabled
            do_decay = opt_conf.get("do_decay", False)
            sched_xyr = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_xyr, gamma=opt_conf.get("decay_rate", 0.99)) if do_decay else None
                
            # Initialize sparsification variables
            z = torch.zeros_like(v, requires_grad=False)
            lam = torch.zeros_like(v, requires_grad=False)
            iters_warmup = sparsify_conf.get("sparsify_warmup", num_iterations//3)
            sparsify_duration = sparsify_conf.get("sparsify_duration", num_iterations//3)
            sparsify_loss_coeff = sparsify_conf.get("sparsify_loss_coeff", 0.5)
            sparsifying_period = sparsify_conf.get("sparsifying_period", 20)
            sparsified_N = int(sparsify_conf.get("sparsified_N", int(0.6 * N)))
            
            # --- Dynamic Sparse Training (DST) configuration ---
            dst_enabled = sparsify_conf.get("dst_enabled", True)
            dst_period = sparsify_conf.get("dst_period", 10)  # DST update frequency
            dst_prune_frac = sparsify_conf.get("dst_prune_frac", 0.2)  # Prune ratio
            
            # Initialize initial mask (SET style)
            if dst_enabled:
                # Calculate active ratio (1 - sparsity)
                density = sparsified_N / N
                # Create initial random mask
                v_mask = torch.zeros_like(v, requires_grad=False)
                active_indices = torch.randperm(N, device=self.device)[:sparsified_N]
                v_mask[active_indices] = 1.0
                
                # Buffer for storing gradients for each parameter
                grad_history = {
                    'x': torch.zeros_like(x),
                    'y': torch.zeros_like(y),
                    'r': torch.zeros_like(r),
                    'v': torch.zeros_like(v),
                    'theta': torch.zeros_like(theta),
                    'c': torch.zeros_like(c)
                }
                
                # Apply initial mask
                print(f"Initialized DST with density {density:.2f} ({sparsified_N}/{N} primitives active)")
            
            assert sparsified_N < N, "sparsified_N must be less than N"
            assert num_iterations > iters_warmup + sparsify_duration, "num_iterations must be greater than warmup + duration"
                        
            optimizer = None  # We'll use the separate optimizers instead
        else:
            # Standard single optimizer if not doing sparsification
            param_groups = [
                {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
                {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0/x.numel())},
                {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)}, # * (1000.0/x.numel())},
                {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
            ]
            optimizer = torch.optim.Adam(param_groups)
            optimizer_xyr = None
            optimizer_rest = None
        
        print(f"Starting optimization, {num_iterations} iterations...")
        for epoch in tqdm(range(num_iterations)):
            # Reset gradients
            if do_sparsify:
                optimizer_xyr.zero_grad()
                optimizer_rest.zero_grad()
                
                # Current sigma for Gaussian blur
                if do_adapt_gaussian_blur:
                    sigma = sigma_start - (sigma_start - sigma_end) * (epoch / num_iterations)
                    print(f"Gaussian blur sigma: {sigma}")
                elif do_gaussian_blur:
                    sigma = blur_sigma #sigma_start * (1 - epoch / num_iterations) + sigma_end * (epoch / num_iterations)
                else:
                    sigma = 0.0
            else:
                optimizer.zero_grad()
                sigma = blur_sigma
            
            # Render image - use different approaches based on precision mode
            if self.use_fp16:
                with autocast('cuda'):
                    cached_masks_ref = None

                    if opt_conf.get("multi_level", False):
                        rendered = self.render(self.S, v, c)
                    else:
                        if streaming_render:
                            # Use standard rendering instead of inefficient streaming
                            cached_masks = self._batched_soft_rasterize(x, y, r, theta, sigma=sigma)
                            rendered = self.render(cached_masks, v, c)
                        else:
                            # Generate masks (memory efficient)
                            cached_masks = self._batched_soft_rasterize(
                                x, y, r, theta,
                                sigma=sigma
                            )
                            
                            # Memory report after mask generation
                            if opt_conf.get("debug_memory", False) and epoch == 0:
                                self.memory_report("After mask generation")
                            
                            rendered = self.render(cached_masks, v, c)
                            
                            # Save image at specified epochs
                            if opt_conf.get("save_epoch", False):
                                output_path = os.path.join(self.output_path, f"epoch_{epoch+1}_{timestamp}.jpg")
                                rendered_np = rendered.detach().cpu().numpy()
                                rendered_np = (rendered_np * 255).astype(np.uint8)
                                Image.fromarray(rendered_np).save(output_path)
                                self.saved_frames.append(output_path)
                                del rendered_np
                            
                            # Save reference for cleanup
                            cached_masks_ref = cached_masks
                            cached_masks = None
                    
                    # Memory report after rendering
                    if opt_conf.get("debug_memory", False) and epoch == 0:
                        self.memory_report("After rendering")
                    
                    # Compute loss
                    recon_loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)

                    if target_image.shape[2] == 4 and target_binary_mask is not None:
                        loss_w_conf = opt_conf.get("loss_weights", {})
                        recon_loss_weight = loss_w_conf.get("recon_loss_weight", 1.0)
                        binary_alpha_loss_weight = loss_w_conf.get("binary_alpha_loss_weight", 0.0) 
                        distance_alpha_loss_weight = loss_w_conf.get("distance_alpha_loss_weight", 0.0)

                        # r for non-gradient computation
                        r_nongrad = r.clone().detach()

                        # Don't apply gaussian blur
                        alpaha_cached_masks = self._batched_soft_rasterize(
                            x, y, r, theta,
                            sigma=0.0
                        )

                        target_mask_expanded = target_binary_mask.unsqueeze(0)  # (1, H, W)
                        alpha_penalties_binary = (target_mask_expanded * alpaha_cached_masks) / r_nongrad.unsqueeze(-1).unsqueeze(-1)  # (N, H, W)
                        alpha_loss_binary = alpha_penalties_binary.sum(dim=0).mean()  # Sum over primitives, then mean over pixels

                        alpha_penalties_dist = (target_dist_mask.unsqueeze(0) * alpaha_cached_masks) / r_nongrad.unsqueeze(-1).unsqueeze(-1)
                        alpha_loss_dist = alpha_penalties_dist.sum(dim=0).mean()  # Sum over primitives, then mean over pixels

                        loss = recon_loss_weight*recon_loss + binary_alpha_loss_weight*alpha_loss_binary + distance_alpha_loss_weight*alpha_loss_dist
                    else:
                        loss = recon_loss

                    # Add sparsification loss if needed
                    if do_sparsify and epoch >= iters_warmup and epoch < iters_warmup + sparsify_duration:
                        # Don't detach for gradient computation
                        _alpha = self.alpha_upper_bound * torch.sigmoid(v)
                        
                        # Use adaptive rho parameter
                        rho = sparsify_loss_coeff * (1.0 - (epoch - iters_warmup) / sparsify_duration)
                        
                        # Calculate loss including mask when DST is applied
                        if dst_enabled:
                            loss += 0.5 * rho * F.mse_loss(_alpha * v_mask, z - lam)
                        else:
                            loss += 0.5 * rho * F.mse_loss(_alpha, z - lam)
                
                    # Calculate gradients and update parameters
                    if do_sparsify:
                        loss.backward()
                        optimizer_xyr.step()
                        optimizer_rest.step()
                        if do_decay and sched_xyr is not None:
                            sched_xyr.step()
                    else:
                        # Standard update with scaler
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                    # Free memory immediately
                    del rendered
                    if self.use_fp16:
                        torch.cuda.empty_cache()
                    
            else:
                # Non-FP16 mode - standard approach without autocast or scaler
                if opt_conf.get("multi_level", False):
                    rendered = self.render(self.S, v, c)
                else:
                    # Generate masks
                    cached_masks = self._batched_soft_rasterize(
                        x, y, r, theta,
                        sigma=sigma
                    )
                    
                    # Render with masks
                    rendered = self.render(cached_masks, v, c)
                    
                    # Save image at specified epochs
                    if opt_conf.get("save_epoch", False):
                        output_path = os.path.join(self.output_path, f"epoch_{epoch+1}_{timestamp}.jpg")
                        rendered_np = rendered.detach().cpu().numpy()
                        rendered_np = (rendered_np * 255).astype(np.uint8)
                        Image.fromarray(rendered_np).save(output_path)
                        self.saved_frames.append(output_path)
                        del rendered_np
                
                # Compute loss
                loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)
                
                # Add sparsification loss if needed
                if do_sparsify and epoch >= iters_warmup and epoch < iters_warmup + sparsify_duration:
                    # Don't detach for gradient computation
                    _alpha = self.alpha_upper_bound * torch.sigmoid(v)
                    
                    # Use adaptive rho parameter
                    rho = sparsify_loss_coeff * (1.0 - (epoch - iters_warmup) / sparsify_duration)
                    
                    # Calculate loss including mask when DST is applied
                    if dst_enabled:
                        loss += 0.5 * rho * F.mse_loss(_alpha * v_mask, z - lam)
                    else:
                        loss += 0.5 * rho * F.mse_loss(_alpha, z - lam)
                
                # Calculate gradients and update parameters
                loss.backward()
                
                if do_sparsify:
                    optimizer_xyr.step()
                    optimizer_rest.step()
                    if do_decay and sched_xyr is not None:
                        sched_xyr.step()
                else:
                    optimizer.step()
                
            # Record loss value for next epoch logging
            loss_value = loss.item()
            # alpha_loss_value = alpha_loss.item()
            
            # Log gradient statistics to wandb
            if opt_conf.get("log_gradients", False):
                self.log_gradient_statistics(x, y, r, v, theta, c, epoch)
            
            if self.use_fp16:
                del loss
                torch.cuda.empty_cache()
            
            # Memory report after cleanup
            if opt_conf.get("debug_memory", False) and epoch == 0:
                self.memory_report("After cleanup")
            
            # Clamp parameters (memory efficient approach)
            with torch.no_grad():
                x.clamp_(0, self.W)
                y.clamp_(0, self.H)
                r.clamp_(2, min(self.H, self.W) // 4)
                # theta.clamp_(0, 2 * np.pi)
                
                # Sparsification logic
                if do_sparsify:
                    if epoch >= iters_warmup and epoch <= iters_warmup + sparsify_duration:
                        # DST mask update
                        if dst_enabled and epoch % dst_period == 0:
                            with torch.no_grad():
                                # DST mask update
                                for param_name in ['v', 'x', 'y', 'r', 'theta', 'c']:
                                    param = locals()[param_name]
                                    if param.grad is not None:
                                        grad_history[param_name] = 0.9 * grad_history[param_name] + 0.1 * param.grad.abs()
                                
                                # 3. Hierarchical relationship analysis
                                # a) Size-based importance (larger ones are "parents" of smaller ones)
                                r_norm = r / r.max()
                                
                                # Final importance calculation: visibility(60%) + size(40%)
                                alpha_importance = self.alpha_upper_bound * torch.sigmoid(v.detach()) * r_norm
                                weights = torch.tensor([0.6, 0.4], device=self.device)
                                factors = torch.stack([alpha_importance, r_norm], dim=0)
                                alpha_importance = torch.matmul(weights, factors)
                                
                                # Remove low-importance parameters from active ones (prune)
                                k_prune = int(sparsified_N * dst_prune_frac)
                                active_idxs = (v_mask > 0).nonzero(as_tuple=True)[0]
                                if len(active_idxs) > 0:
                                    importance = alpha_importance[active_idxs]
                                    prune_candidates = active_idxs[torch.topk(importance, k=k_prune, largest=False).indices]
                                    v_mask[prune_candidates] = 0.0
                                
                                # Add considering both importance and gradients from inactive parameters (grow)
                                inactive_idxs = (v_mask == 0).nonzero(as_tuple=True)[0]
                                if len(inactive_idxs) > 0:
                                    # Combined score of gradient magnitude and visual importance
                                    grow_scores = 0.7 * grad_history['v'][inactive_idxs] + 0.3 * alpha_importance[inactive_idxs]
                                    grow_candidates = inactive_idxs[torch.topk(grow_scores, k=k_prune, largest=True).indices]
                                    v_mask[grow_candidates] = 1.0
                                
                                print(f"DST update - Active: {v_mask.sum().item()}/{N} primitives")
                        
                        # Regular ADMM z, λ update (periodically)
                        if epoch % sparsifying_period == 0:
                            # Compute alpha values from current parameters
                            _alpha = self.alpha_upper_bound * torch.sigmoid(v.detach())
                           
                            combined_score = _alpha 
                            
                            # Integration with DST: mask-based score calculation
                            if dst_enabled:
                                # Consider mask: only active parameters are candidates for selection
                                masked_score = combined_score * v_mask
                                keep_idx = torch.topk(masked_score, sparsified_N).indices
                            else:
                                # Traditional approach: select from all parameters
                                keep_idx = torch.topk(combined_score, sparsified_N).indices
                            
                            mask = torch.zeros_like(_alpha, dtype=torch.bool)
                            mask[keep_idx] = True
                            
                            # ADMM update
                            z.zero_()
                            if dst_enabled:
                                z[mask] = (_alpha * v_mask)[mask]  # Only update values with applied mask
                            else:
                                z[mask] = _alpha[mask]
                            lam += (_alpha - z)
                        
                        if epoch == iters_warmup + sparsify_duration:
                            # actual sparsification
                            # 1) helper: existing tensor -> trimmed tensor (maintains requires_grad)
                            def _prune(t):
                                t_new = t[keep_idx].clone().detach()
                                if t.requires_grad:
                                    t_new.requires_grad_(True)
                                return t_new
                            
                            # 2) Actually prune parameters and auxiliary variables
                            x, y, r, theta, v, c = map(_prune, (x, y, r, theta, v, c))
                            z, lam = map(_prune, (z, lam))
                            
                            # Also prune DST related variables
                            if dst_enabled:
                                v_mask = v_mask[keep_idx].clone()
                                for key in grad_history:
                                    if key == 'v':
                                        grad_history[key] = grad_history[key][keep_idx].clone()
                            
                            # 3) Create fresh optimizers after pruning without attempting state transfer
                            # This avoids tensor size mismatch issues
                            optimizer_xyr = torch.optim.Adam([
                                {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                                {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                                {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
                            ])
                            
                            optimizer_rest = torch.optim.Adam([
                                {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},
                                {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
                                {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
                            ])
                            
                            # Create scheduler if decay is enabled
                            sched_xyr = torch.optim.lr_scheduler.ExponentialLR(
                                optimizer_xyr, gamma=opt_conf.get("decay_rate", 0.99)) if do_decay else None

            if not streaming_render and not opt_conf.get("multi_level", False) and 'cached_masks_ref' in locals():
                del cached_masks_ref
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.4f}")
                # print(f"Epoch {epoch}, Alpha Loss: {alpha_loss_value:.4f}")

        # Final memory report
        if opt_conf.get("debug_memory", False):
            self.memory_report("After optimization")
        
        # Finish wandb run
        if opt_conf.get("log_gradients", False):
            wandb.finish()
        
        if opt_conf.get("save_epoch", False):
            video_path = os.path.join(self.output_path, f"optimization_progress_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            self.create_video_from_collected_frames(video_path, fps=30)
            print(f"Optimization progress video saved to: {video_path}")
            
        return x, y, r, v, theta, c

    def save_rendered_image(self,
                          cached_masks: torch.Tensor,
                          v: torch.Tensor,
                          c: torch.Tensor,
                          output_path: str) -> None:
        """
        Save the rendered image to a file.
        
        Args:
            cached_masks: Pre-computed masks
            v: Visibility parameters
            c: Color parameters
            output_path: Path to save the rendered image
        """
        final_render = self.render_with_white_bg(cached_masks, v, c)
        final_render_np = final_render.detach().cpu().numpy()
        final_render_np = (final_render_np * 255).astype(np.uint8)
        
        # Save the image using PIL
        Image.fromarray(final_render_np).save(output_path)
       
    def render_export_mp4_hires(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              r: torch.Tensor,
                              theta: torch.Tensor,
                              v: torch.Tensor,
                              c: torch.Tensor,
                              video_path: str,
                              scale_factor: float = 4.0,
                              fps: int = 60) -> None:
        """
        Export high-resolution MP4 video using streaming approach to avoid VRAM overflow.
        Shows progressive primitive addition at high resolution.
        
        Args:
            x, y, r, theta, v, c: Primitive parameters
            video_path: Path to save the MP4 video
            scale_factor: Resolution multiplier (e.g., 4.0 = 4x resolution)
            chunk_size: Number of primitives to process at once (smaller = less VRAM)
            fps: Frames per second for the video
        """
        import cv2
        
        # Store original resolution
        orig_H, orig_W = self.H, self.W
        
        # Temporarily scale up resolution
        self.H = int(orig_H * scale_factor)
        self.W = int(orig_W * scale_factor)
        
        # Scale up coordinates and sizes
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        r_scaled = r * scale_factor
        
        # Recreate coordinate grid for new resolution
        old_X, old_Y = self.X, self.Y
        self.X, self.Y = self._create_coordinate_grid()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (self.W, self.H))
        
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        
        try:
            N = len(x_scaled)
            print(f"Generating high-resolution MP4 with {N} primitives at {self.W}x{self.H}...")
            
            # Initialize with white background
            current_frame = torch.ones((self.H, self.W, 3), device=x.device, dtype=torch.float32)
            
            # Write first frame (white background)
            frame_np = current_frame.detach().cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            
            # Process primitives incrementally (efficient approach)
            for frame_idx in range(N):
                if frame_idx % max(1, N // 20) == 0:  # Progress updates
                    print(f"Processing frame {frame_idx}/{N}...")
                
                # Get current primitive parameters (single primitive)
                curr_x = x_scaled[frame_idx:frame_idx + 1]
                curr_y = y_scaled[frame_idx:frame_idx + 1]
                curr_r = r_scaled[frame_idx:frame_idx + 1]
                curr_theta = theta[frame_idx:frame_idx + 1]
                curr_v = v[frame_idx:frame_idx + 1]
                curr_c = c[frame_idx:frame_idx + 1]
                
                # Render only the current primitive
                with torch.no_grad():
                    # Generate mask for current primitive
                    mask = self._batched_soft_rasterize(
                        curr_x, curr_y, curr_r, curr_theta, sigma=0.0
                    ).squeeze(0)  # Remove batch dimension -> [H, W]
                    
                    # Calculate alpha and color for current primitive
                    alpha = (self.alpha_upper_bound * torch.sigmoid(curr_v)).item()
                    color = torch.sigmoid(curr_c).squeeze(0)  # [3]
                    
                    # Apply alpha to mask
                    alpha_mask = alpha * mask  # [H, W]
                    
                    # Create colored primitive: alpha_mask * color
                    colored_primitive = alpha_mask.unsqueeze(-1) * color.unsqueeze(0).unsqueeze(0)  # [H, W, 3]
                    
                    # Alpha composite: new_frame = colored_primitive + (1 - alpha_mask) * current_frame
                    inv_alpha = 1.0 - alpha_mask.unsqueeze(-1)  # [H, W, 1]
                    current_frame = colored_primitive + inv_alpha * current_frame
                
                # Convert to numpy and write frame
                frame_np = current_frame.detach().cpu().numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                
                # Clear GPU memory periodically
                if frame_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            print(f"Saved high-resolution MP4 ({self.W}x{self.H}) to {video_path}")
            
        finally:
            # Clean up
            writer.release()
            
            # Restore original resolution and coordinate grid
            self.H, self.W = orig_H, orig_W
            self.X, self.Y = old_X, old_Y
            
            # Clear GPU memory
            torch.cuda.empty_cache()

    def create_video_from_collected_frames(self, output_video_path: str, fps: int = 30) -> str:
        """
        Create an MP4 video from frames collected during optimization.
        
        Args:
            output_video_path: Path for the output MP4 file
            fps: Frames per second for the video
            
        Returns:
            Path to the created video file
        """
        if not self.saved_frames:
            print("No frames collected during optimization.")
            return None
        
        print(f"Creating video from {len(self.saved_frames)} collected frames...")
        
        # Read the first frame to get dimensions
        first_frame = cv2.imread(self.saved_frames[0])
        if first_frame is None:
            print(f"Could not read first frame: {self.saved_frames[0]}")
            return None
        
        height, width, layers = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Could not create video writer for: {output_video_path}")
            return None
        
        # Add frames to video
        for i, frame_path in enumerate(tqdm(self.saved_frames, desc="Creating video")):
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_path}")
        
        # Release video writer
        video_writer.release()
        
        print(f"Video created successfully: {output_video_path}")
        print(f"Video contains {len(self.saved_frames)} frames at {fps} FPS")
        
        return output_video_path 
        
    def render_image(self, x, y, r, v, theta, c):
        """
        Render an image from primitive parameters.
        Args:
            x, y, r, v, theta, c: Primitive parameters (torch.Tensor)
        Returns:
            Rendered image (H, W, 3)
        """
        cached_masks = self._batched_soft_rasterize(x, y, r, theta)
        return self.render(cached_masks, v, c)
    
    def over(self, I_hat, I_bg):
        """
        Alpha blending (Porter-Duff 'over') between I_hat and I_bg.
        I_hat: (H, W, 3) foreground
        I_bg:  (H, W, 3) background
        Returns:
            Blended image (H, W, 3)
        """
        # If alpha channel exists, use it, otherwise assume 1
        if I_hat.shape[-1] == 4:
            alpha = I_hat[..., 3:4]
            rgb = I_hat[..., :3]
        else:
            alpha = torch.ones_like(I_hat[..., :1])
            rgb = I_hat
        return rgb * alpha + I_bg * (1 - alpha)
    
    def memory_report(self, message="Memory usage"):
        """
        Detailed memory usage report.
        """
        torch.cuda.synchronize()
        
        current = torch.cuda.memory_allocated() / 1024 / 1024
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        print(f"{message}: Current={current:.2f}MB, Peak={peak:.2f}MB, Reserved={reserved:.2f}MB")
        
        # Print top tensors by memory usage
        tensors_report = defaultdict(int)
        sizes_report = {}
        precision_report = {}
        
        # Check tensors in self
        for name, obj in vars(self).items():
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                size_bytes = obj.nelement() * obj.element_size()
                tensors_report[name] = size_bytes
                sizes_report[name] = obj.shape
                precision_report[name] = obj.dtype
        
        # Print sorted tensors by memory usage (largest first)
        if tensors_report:
            print("  Top tensors by memory usage:")
            sorted_tensors = sorted(tensors_report.items(), key=lambda x: x[1], reverse=True)
            for name, size in sorted_tensors[:5]:  # Print top 5 only
                print(f"    {name}: {size/1024/1024:.2f}MB, shape={sizes_report[name]}, dtype={precision_report[name]}")
        
        # Compare memory usage between FP16 and FP32
        fp16_count = sum(1 for dtype in precision_report.values() if dtype == torch.float16)
        fp32_count = sum(1 for dtype in precision_report.values() if dtype == torch.float32)
        fp16_bytes = sum(size for name, size in tensors_report.items() if precision_report.get(name) == torch.float16)
        fp32_bytes = sum(size for name, size in tensors_report.items() if precision_report.get(name) == torch.float32)
        
        if fp16_count + fp32_count > 0:
            print(f"  FP16: {fp16_count} tensors, {fp16_bytes/1024/1024:.2f}MB")
            print(f"  FP32: {fp32_count} tensors, {fp32_bytes/1024/1024:.2f}MB")
        
        return current, peak, reserved

    def log_gradient_statistics(self, x, y, r, v, theta, c, epoch):
        """
        Log gradient statistics for 20% of primitives and 0th primitive to wandb.
        
        Args:
            x, y, r, v, theta, c: Parameter tensors
            epoch: Current epoch number
        """
        
        # Collect gradients for sampled primitives
        gradients = {}
        params = {'x': x, 'y': y, 'r': r, 'v': v, 'theta': theta, 'c': c}
        
        # for param_name, param in params.items():
        #     if param.grad is not None:
        #         # Get gradients for sampled indices
        #         if param_name == 'c':
        #             # c has shape (N, 3), so we need to handle it differently
        #             grad_sample = param.grad[self.sample_indices]  # (sample_size, 3)
        #             gradients[f'{param_name}_grad_mean'] = grad_sample.mean().item()
        #             gradients[f'{param_name}_grad_std'] = grad_sample.std().item()
        #             gradients[f'{param_name}_grad_norm'] = grad_sample.norm().item()
                    
        #             # Log individual RGB channel gradients
        #             for i, color in enumerate(['r', 'g', 'b']):
        #                 gradients[f'{param_name}_{color}_grad_mean'] = grad_sample[:, i].mean().item()
        #                 gradients[f'{param_name}_{color}_grad_std'] = grad_sample[:, i].std().item()
        #         else:
        #             # Other parameters have shape (N,)
        #             grad_sample = param.grad[self.sample_indices]
        #             gradients[f'{param_name}_grad_mean'] = grad_sample.mean().item()
        #             gradients[f'{param_name}_grad_std'] = grad_sample.std().item()
        #             gradients[f'{param_name}_grad_norm'] = grad_sample.norm().item()
        #     else:
        #         # No gradients available
        #         gradients[f'{param_name}_grad_mean'] = 0.0
        #         gradients[f'{param_name}_grad_std'] = 0.0
        #         gradients[f'{param_name}_grad_norm'] = 0.0
        #         if param_name == 'c':
        #             for color in ['r', 'g', 'b']:
        #                 gradients[f'{param_name}_{color}_grad_mean'] = 0.0
        #                 gradients[f'{param_name}_{color}_grad_std'] = 0.0
        
        # Log 0th primitive gradients and values
        for param_name, param in params.items():
            if param.grad is not None:
                if param_name == 'c':
                    # c has shape (N, 3)
                    grad_0th = param.grad[0]  # (3,)
                    param_0th = param[0]      # (3,)
                    
                    # Log 0th primitive gradient values
                    gradients[f'{param_name}_0th_r'] = grad_0th[0].item()
                    gradients[f'{param_name}_0th_g'] = grad_0th[1].item()
                    gradients[f'{param_name}_0th_b'] = grad_0th[2].item()
                else:
                    # Other parameters have shape (N,)
                    grad_0th = param.grad[0]
                    param_0th = param[0]
                    
                    # Log 0th primitive gradient and value
                    gradients[f'{param_name}_0th_grad'] = grad_0th.item()
            else:
                # No gradients available for 0th primitive
                if param_name == 'c':
                    for color in ['r', 'g', 'b']:
                        gradients[f'{param_name}_0th_grad_{color}'] = 0.0
                else:
                    gradients[f'{param_name}_0th_grad'] = 0.0
        
        # Log to wandb
        wandb.log(gradients)

    def clear_cuda_cache(self):
        """
        Completely clear CUDA cache and perform garbage collection.
        """
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        # Reset peak memory stats for memory usage minimization
        torch.cuda.reset_peak_memory_stats()
        
        current = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"After cache clearing: Current={current:.2f}MB, Reserved={reserved:.2f}MB")

    def create_stream(self):
        """
        Create a new CUDA stream to handle memory operations asynchronously.
        This allows multiple operations to be performed simultaneously, reducing memory usage.
        """
        return torch.cuda.Stream()

    def with_stream(self, stream):
        """
        Return a context manager to execute code block in given stream.
        """
        class StreamContext:
            def __init__(self, stream):
                self.stream = stream
                self.prev_stream = None
                
            def __enter__(self):
                self.prev_stream = torch.cuda.current_stream()
                torch.cuda.set_stream(self.stream)
                return self.stream
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                torch.cuda.set_stream(self.prev_stream)
                self.stream.synchronize()
                
        return StreamContext(stream)

from collections import defaultdict

def pretty_mem(x):       # byte → MB unit string
    return f"{x/1024/1024:8.2f} MB"

def tensor_vram_report(namespace: dict):
    """
    namespace(dict): globals()   or locals()
    Prints a table of all torch.Tensor objects (including .grad) on GPU.
    """
    seen_data_ptr = set()
    rows, total = [], 0
    for name, obj in namespace.items():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            # same storage can be shared among multiple variables
            ptr = obj.data_ptr()
            if ptr in seen_data_ptr:
                continue
            seen_data_ptr.add(ptr)

            size_bytes = obj.numel() * obj.element_size()
            rows.append((name, str(tuple(obj.shape)), obj.dtype, pretty_mem(size_bytes)))
            total += size_bytes

            # grad buffer also exists if it exists
            if obj.grad is not None:
                gbytes = obj.grad.numel() * obj.grad.element_size()
                rows.append((name + ".grad", str(tuple(obj.grad.shape)),
                             obj.grad.dtype, pretty_mem(gbytes)))
                total += gbytes

    # Sort: largest first
    rows.sort(key=lambda r: float(r[3].split()[0]), reverse=True)

    print(f"{'tensor':25} {'shape':20} {'dtype':9} {'mem':>10}")
    print("-"*70)
    for n,s,d,m in rows:
        print(f"{n:25} {s:20} {str(d):9} {m:>10}")
    print("-"*70)
    print(f"{'TOTAL':25} {'':20} {'':9} {pretty_mem(total):>10}")


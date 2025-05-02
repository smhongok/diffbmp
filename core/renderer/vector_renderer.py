import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.utils.checkpoint as cp
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
from util.utils import gaussian_blur, make_batch_indices
import os
import gc
import pkg_resources
from collections import defaultdict

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
                 device: str = 'cuda'):
        """
        Initialize the vector renderer.
        
        Args:
            canvas_size: Tuple of (height, width) for the output canvas
            alpha_upper_bound: Maximum alpha value for rendering (default: 0.5)
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.H, self.W = canvas_size
        self.alpha_upper_bound = alpha_upper_bound
        self.device = device
        self.use_checkpointing = False
        # Convert S to half precision during initialization
        self.S = S.to(dtype=torch.float16)
        
        # Pre-compute pixel coordinates
        self.X, self.Y = self._create_coordinate_grid()
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpointing = True
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_checkpointing = False
    
    def _create_coordinate_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create the coordinate grid for rendering."""
        X, Y = torch.meshgrid(
            torch.arange(self.W, device=self.device, dtype=torch.float16),
            torch.arange(self.H, device=self.device, dtype=torch.float16),
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

        Args:
            x, y:       [N] position coordinates for N shapes
            r:          [N] scales
            theta:      [N] rotations
            sigma:      Gaussian blur std
        Returns:
            masks:     [N, H, W]  (one soft mask per shape)
        """
        with autocast():
            B = len(x)
            _, H, W = self.X.shape
            
            # Apply Gaussian blur if needed
            if sigma > 0.0:
                bmp = self.S.unsqueeze(0)
                bmp = gaussian_blur(bmp, sigma)
                bmp_image = bmp.squeeze(0)
            else:
                bmp_image = self.S
            
            # Expand parameters to match grid dimensions and convert to half precision
            X_exp = self.X.expand(B, H, W).half()
            Y_exp = self.Y.expand(B, H, W).half()
            x_exp = x.view(B, 1, 1).expand(B, H, W).half()
            y_exp = y.view(B, 1, 1).expand(B, H, W).half()
            r_exp = r.view(B, 1, 1).expand(B, H, W).half()
            
            # Position normalization and rotation
            pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
            cos_t = torch.cos(theta).half()
            sin_t = torch.sin(theta).half()
            R_inv = torch.zeros(B, 2, 2, device=self.device, dtype=torch.float16)
            R_inv[:, 0, 0] = cos_t
            R_inv[:, 0, 1] = sin_t
            R_inv[:, 1, 0] = -sin_t
            R_inv[:, 1, 1] = cos_t
            uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
            
            # Prepare for grid sampling
            grid = uv.permute(0, 2, 3, 1)  # (B, H, W, 2)
            bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1).half()
            
            # Use gradient checkpointing if enabled
            if self.use_checkpointing:
                def grid_sample_func(x, grid):
                    return F.grid_sample(
                        x,
                        grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=True
                    )

                sampled = torch.utils.checkpoint.checkpoint(
                    grid_sample_func,
                    bmp_exp,
                    grid,
                    use_reentrant=False
                )
            else:
                sampled = F.grid_sample(
                    bmp_exp,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )
            
            return sampled.squeeze(1)  # (B, H, W)
            B = len(x)
            _, H, W = self.X.shape
    
    def _tree_over(self, m: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient tree-based alpha compositing.
        
        Args:
            m: Color tensor
            a: Alpha tensor
            
        Returns:
            Tuple of (composited color, composited alpha)
        """
        with autocast():
            # Convert to half precision to save memory
            m = m.half()
            a = a.half()
            N, H, W, _ = m.shape
            
            # 인스턴스 버퍼 대신 로컬 버퍼를 사용하여 체크포인팅 호환성 보장
            comp_m = torch.zeros(H, W, 3, device=m.device, dtype=torch.float16)
            comp_a = torch.zeros(H, W, device=m.device, dtype=torch.float16)
                
 
            for i in range(N):
                ai = a[i]
                mi = m[i]
                
                # 1-ai를 미리 계산하여 재사용
                inv_ai = 1.0 - ai
                
                # inplace 연산을 사용하여 메모리 사용 최적화
                comp_m *= inv_ai.unsqueeze(-1)
                comp_m += mi
                
                comp_a *= inv_ai
                comp_a += ai
                
                # 임시 텐서 즉시 해제
                del ai, mi, inv_ai
                
            
            return comp_m, comp_a
    
    def _get_checkpoint_kwargs(self):
        """
        Returns the correct checkpoint keyword arguments based on the PyTorch version.
        Older versions don't support use_reentrant.
        """
        # PyTorch 버전 확인
        torch_version = pkg_resources.get_distribution("torch").version
        major, minor = map(int, torch_version.split('.')[:2])
        
        # PyTorch 1.12 이상에서만 use_reentrant 지원
        if (major > 1) or (major == 1 and minor >= 12):
            return {"use_reentrant": False}
        else:
            # 이전 버전에서는 해당 옵션 없음
            return {}
    
    def _safe_checkpoint(self, func, *tensors):
        """
        PyTorch 버전에 관계없이 안전하게 체크포인팅을 수행하는 래퍼 함수
        """
        kwargs = self._get_checkpoint_kwargs()
        return torch.utils.checkpoint.checkpoint(func, *tensors, **kwargs)
            
    def render(
            self,
            cached_masks: torch.Tensor,
            v: torch.Tensor,
            c: torch.Tensor,
            return_alpha: bool = False   # ★ 새 인자
        ):
        """
        Render the final image (optionally alpha).

        Args:
            cached_masks : (N, H, W)   – pre-computed soft masks
            v            : (N,)        – visibility logits
            c            : (N, 3)      – RGB logits
            return_alpha : If True  → (rgb, alpha) 반환
                        If False → rgb 만 반환

        Returns
        -------
        - rgb  : (H, W, 3)  (always)
        - alpha: (H, W, 1)  (옵션, return_alpha=True일 때)
        """
        with autocast():    
            # Ensure all inputs have consistent types
            input_dtype = cached_masks.dtype
            cached_masks = cached_masks.to(dtype=input_dtype)
            v = v.to(dtype=input_dtype)
            c = c.to(dtype=input_dtype)
            
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
                    lambda mm, aa: self._tree_over(mm, aa),
                    m, a
                )
            else:
                comp_m, comp_a = self._tree_over(m, a)    
            
            # Free large tensors as soon as possible
            del m, a, v_alpha, c_eff
            
            if return_alpha:
                # (H, W) → (H, W, 1) 로 맞춰 두면 브로드캐스트 편함
                return comp_m, comp_a.unsqueeze(-1)

            # 3. 흰 배경까지 합성해서 완성 RGB
            ones = torch.ones_like(comp_m)
            final = comp_m + (1.0 - comp_a).unsqueeze(-1) * ones
            
            # Free temporary tensors
            del comp_m, comp_a, ones
            
            return final

    
    def _stream_render(self,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    theta: torch.Tensor,
                    v: torch.Tensor,
                    c: torch.Tensor,
                    sigma: float = 0.0,
                    raster_chunk_size: int = 20) -> torch.Tensor:
        """
        Streaming‐composite render: rasterize in small chunks and
        composite immediately, never buffering all N masks.
        """
        N = x.shape[0]
        H, W = self.H, self.W
        device = x.device

        # 1) Pre-blur the source bitmap once (FP32), then to FP16
        with torch.no_grad(), autocast():
            bmp = self.S.unsqueeze(0)
            if sigma > 0:
                bmp = gaussian_blur(bmp, sigma)
                bmp16 = bmp.half().squeeze(0)  # [C_bmp, H, W]
            else:
                bmp16 = self.S.half()
            del bmp

        # 2) Prepare alphas & colors
        with torch.no_grad(), autocast():
            # 더 작은 청크로 v와 c를 처리
            chunk_size = min(100, max(1, N // 2))
            
            # 3) Running composite buffers (FP16) - 로컬 변수로 전환
            comp_m = torch.zeros((H, W, 3), device=device, dtype=torch.float16)
            comp_a = torch.zeros((H, W), device=device, dtype=torch.float16)
        
        # 4) Process in small shape‐chunks
        for i in range(0, N, raster_chunk_size):
            j = min(i + raster_chunk_size, N)
            curr_chunk_size = j - i
            
            # 현재 청크의 가시성과 색상 계산
            with torch.no_grad(), autocast():
                v_chunk = v[i:j].half()
                c_chunk = c[i:j].half()
                v_alpha = (self.alpha_upper_bound * torch.sigmoid(v_chunk)).view(curr_chunk_size, 1, 1).half()
                c_eff = torch.sigmoid(c_chunk).view(curr_chunk_size, 1, 1, 3).half()
                
                # Convert position parameters to half precision
                x_chunk = x[i:j].half()
                y_chunk = y[i:j].half()
                r_chunk = r[i:j].half()
                theta_chunk = theta[i:j].half()
                
            # a) Rasterize just this subset with autocast
            with autocast():
                masks_chunk = self._batched_soft_rasterize(
                    x_chunk, y_chunk,
                    r_chunk, theta_chunk,
                    sigma=0.0                # 이미 bmp16에 블러 적용됨
                ).half()                     # [C, H, W], FP16
                
                # b) 한 번에 모든 마스크 합성하지 않고 더 작은 하위 청크로 분할
                sub_chunk_size = 5  # 작은 하위 청크 크기
                
                for k in range(0, curr_chunk_size, sub_chunk_size):
                    end_k = min(k + sub_chunk_size, curr_chunk_size)
                    
                    # 현재 하위 청크의 마스크, 알파, 색상
                    masks_subchunk = masks_chunk[k:end_k]
                    v_subchunk = v_alpha[k:end_k]
                    c_subchunk = c_eff[k:end_k]
                    
                    # 각 모양 순차 합성
                    for s in range(end_k - k):
                        a_s = v_subchunk[s] * masks_subchunk[s]          # [H, W]
                        m_s = a_s.unsqueeze(-1) * c_subchunk[s]       # [H, W, 3]
                        
                        # inplace 연산 사용
                        inv_a_s = 1.0 - a_s
                        comp_m = m_s + inv_a_s.unsqueeze(-1) * comp_m
                        comp_a = a_s + inv_a_s * comp_a
                        
                        # 임시 텐서 즉시 해제
                        del a_s, m_s, inv_a_s
                    
                    # 하위 청크 텐서 해제
                    del masks_subchunk, v_subchunk, c_subchunk
                    torch.cuda.empty_cache()

            # 청크 텐서 해제
            del masks_chunk, v_chunk, c_chunk, v_alpha, c_eff, x_chunk, y_chunk, r_chunk, theta_chunk
            torch.cuda.empty_cache()

        # 5) finalize with white background
        with torch.no_grad(), autocast():
            final = comp_m + (1 - comp_a).unsqueeze(-1)
            # Free memory before return
            del comp_m, comp_a
        
        return final
    
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
                            target_image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Initialize parameters using the provided initializer.
        
        Args:
            initializer: Initializer object (e.g., StructureAwareInitializer)
            target_image: Target image to match
            
        Returns:
            Tuple of initialized parameters (x, y, r, v, theta, c)
        """
        # Initialize from target image
        x, y, r, v, theta, c = initializer.initialize(target_image)
        
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
                          opt_conf: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        if opt_conf.get("batch_optimization", False):
            return self._optimize_parameters_batched(
                x, y, r, v, theta, c, target_image, opt_conf
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
        num_iterations = opt_conf.get("num_iterations", 300)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Initial memory report
        if opt_conf.get("debug_memory", False):
            self.memory_report("Before optimization")
        
        # Mixed-precision용 Scaler
        scaler = GradScaler()
        
        # Pre-calculate configurations
        blur_sigma = opt_conf.get("blur_sigma", 0.0)
        streaming_render = opt_conf.get("streaming_render", False)
        raster_chunk_size = opt_conf.get("raster_chunk_size", 20)  # 더 작은 청크 크기로 변경
        
        # 메모리 효율성을 위해 그룹별로 최적화
        # 그룹 1: 위치 매개변수 (x, y)
        # 그룹 2: 크기 및 회전 매개변수 (r, theta)
        # 그룹 3: 외관 매개변수 (v, c)
        param_groups = [
            {'params': [x, y], 'names': ['x', 'y'], 'lr': lr * lr_conf.get("gain_x", 1.0)},
            {'params': [r, theta], 'names': ['r', 'theta'], 'lr': lr * lr_conf.get("gain_r", 1.0)},
            {'params': [v, c], 'names': ['v', 'c'], 'lr': lr * lr_conf.get("gain_v", 1.0)}
        ]
        
        optimizers = []
        for group in param_groups:
            optimizers.append(torch.optim.Adam(group['params'], lr=group['lr']))
        
        # Gradient accumulation steps (더 작은 배치로 메모리 사용량 감소)
        accum_steps = opt_conf.get("gradient_accumulation_steps", 1)
        
        print(f"Starting optimization with {len(param_groups)} parameter groups, {accum_steps} accumulation steps for {num_iterations} iterations...")
        for epoch in tqdm(range(num_iterations)):
            # 그래디언트 누적 스텝을 통한 메모리 최적화
            for accum_step in range(accum_steps):
                # 모든 최적화기의 그래디언트 초기화
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                # Render image
                with autocast():
                    if opt_conf.get("multi_level", False):
                        rendered = self.render(self.S, v, c)
                    else:
                        if streaming_render:
                            rendered = self._stream_render(
                                x, y, r, theta,
                                v, c,
                                sigma=blur_sigma,
                                raster_chunk_size=raster_chunk_size
                            )
                        else:
                            # 소량 마스크 생성 (메모리 효율성)
                            cached_masks = self._batched_soft_rasterize(
                                x, y, r, theta,
                                sigma=blur_sigma
                            )
                            
                            # Memory report after mask generation
                            if opt_conf.get("debug_memory", False) and epoch == 0 and accum_step == 0:
                                self.memory_report("After mask generation")
                            
                            rendered = self.render(cached_masks, v, c)
                            
                            # Immediately free memory
                            cached_masks_ref = cached_masks
                            cached_masks = None
                    
                    # Memory report after rendering
                    if opt_conf.get("debug_memory", False) and epoch == 0 and accum_step == 0:
                        self.memory_report("After rendering")
                
                    # Compute loss (still within autocast)
                    loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)
                    
                    # 그래디언트 누적을 위해 loss 스케일링
                    scaled_loss = loss / accum_steps
                
                # 그래디언트 계산 (각 스텝에서 누적)
                scaler.scale(scaled_loss).backward()
                
                # 메모리 즉시 해제
                del rendered, scaled_loss
                if not streaming_render and not opt_conf.get("multi_level", False):
                    del cached_masks_ref
                torch.cuda.empty_cache()
            
            # Memory report after backward
            if opt_conf.get("debug_memory", False) and epoch == 0:
                self.memory_report("After gradient accumulation")
            
            # 누적된 그래디언트로 매개변수 업데이트
            for optimizer in optimizers:
                scaler.step(optimizer)
            
            scaler.update()
            
            # Loss값 기록 (다음 epoch 로깅용)
            loss_value = loss.item()
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
                theta.clamp_(0, 2 * np.pi)
                c.clamp_(0.0, 1.0)
                v.clamp_(0.0, 1.0)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.4f}")
                
        # Final memory report
        if opt_conf.get("debug_memory", False):
            self.memory_report("After optimization")
            
        return x, y, r, v, theta, c
    
    def _optimize_parameters_batched(self,
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
        num_iterations = opt_conf.get("num_iterations", 300)
        chunk = opt_conf.get("batch_size", 256)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
            {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
            {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},
            {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
            {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
        ])

        # 배치 최적화에서는 캐시된 마스크를 미리 생성
        with torch.no_grad():
            cached_masks = self._batched_soft_rasterize(
                x, y, r, theta, sigma=opt_conf.get("blur_sigma", 0.0)
            )

        N = x.numel()
        print(f"Starting optimization for Mini-batch GD: N={N}  chunk={chunk}  iterations={num_iterations}...")
        for step in tqdm(range(num_iterations)):
            # -------- 0. 앞·배치·뒤 인덱스 슬라이스 --------
            start = (step * chunk) % N
            end   = min(start + chunk, N)
            fg_sl     = slice(0, start)   # 앞쪽(near)
            batch_sl  = slice(start, end) # 학습 대상
            bg_sl     = slice(end, N)     # 뒤쪽(far)

            # -------- 1. no-grad 합성(FG,BG) --------
            with torch.no_grad():
                # 앞 레이어 → rgb_fg, alpha_fg
                if start > 0:
                    # 매번 새로운 버퍼로 렌더링
                    rgb_fg, alpha_fg = self.render(
                        cached_masks[fg_sl], v[fg_sl], c[fg_sl],
                        return_alpha=True
                    )
                else:  # 아무것도 없으면 투명
                    rgb_fg  = torch.zeros((self.H, self.W, 3), device=self.device, dtype=torch.float16)
                    alpha_fg = torch.zeros((self.H, self.W, 1), device=self.device, dtype=torch.float16)
                    
                # 뒤 레이어 → rgb_bg (흰 배경까지 합성, alpha 버림)
                if end < N:
                    rgb_bg = self.render(
                        cached_masks[bg_sl], v[bg_sl], c[bg_sl],
                        return_alpha=False
                    )
                else:
                    rgb_bg = torch.ones((self.H, self.W, 3), device=self.device, dtype=torch.float16)

            # -------- 2. 학습 배치 forward --------
            optimizer.zero_grad()
            # Render image
            if opt_conf.get("multi_level", False):
                # 멀티레벨 렌더링 미구현
                raise NotImplementedError("Multi-level rendering not implemented")
            else:
                # 현재 배치 마스크 생성
                masks_bt = self._batched_soft_rasterize(
                    x[batch_sl], y[batch_sl], r[batch_sl], theta[batch_sl],
                    sigma=opt_conf.get("blur_sigma", 0.0)
                )
                
                # 배치 렌더링
                rgb_bt, alpha_bt = self.render(
                        masks_bt, v[batch_sl], c[batch_sl],
                        return_alpha=True
                    )
            
            # -------- 3. Porter-Duff over(fg → batch → bg) --------
            # 새 텐서를 명시적으로 생성하여 체크포인팅 호환성 향상
            comp1 = rgb_fg + (1 - alpha_fg) * rgb_bt
            
            # 마지막 합성도 새 텐서를 생성
            inv_alpha_product = (1 - alpha_fg) * (1 - alpha_bt)
            final = comp1 + inv_alpha_product * rgb_bg

            # 손실 계산
            loss = self.compute_loss(final, target_image,
                                    x, y, r, v, theta, c)
            
            # 역전파 및 업데이트
            loss.backward()
            optimizer.step()            
            
            # 사용한 텐서 명시적 정리
            del masks_bt, rgb_bt, alpha_bt, comp1, inv_alpha_product, final, rgb_fg, alpha_fg, rgb_bg
            torch.cuda.empty_cache()
            
            # Clamp parameters
            with torch.no_grad():
                x.clamp_(0, self.W)
                y.clamp_(0, self.H)
                r.clamp_(2, min(self.H, self.W) // 4)
                theta.clamp_(0, 2 * np.pi)
            
                # ── (NEW) 바뀐 batch 슬라이스만 다시 마스크 계산해서 캐시에 업데이트 ──
                cached_masks[batch_sl] = self._batched_soft_rasterize(
                    x[batch_sl], y[batch_sl], r[batch_sl], theta[batch_sl],
                    sigma=opt_conf.get("blur_sigma", 0.0)        # 동일 블러 파라미터 사용
                )

            if step % 20 == 0:
                print(f"Epoch {step}, Loss: {loss.item():.4f}")
        
        # 캐시된 마스크 정리
        del cached_masks
        torch.cuda.empty_cache()
        
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
        final_render = self.render(cached_masks, v, c)
        final_render_np = final_render.detach().cpu().numpy()
        final_render_np = (final_render_np * 255).astype(np.uint8)
        
        # Save the image using PIL
        from PIL import Image
        Image.fromarray(final_render_np).save(output_path) 
        
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
        # 알파 채널이 있으면 사용, 없으면 1로 가정
        if I_hat.shape[-1] == 4:
            alpha = I_hat[..., 3:4]
            rgb = I_hat[..., :3]
        else:
            alpha = torch.ones_like(I_hat[..., :1])
            rgb = I_hat
        return rgb * alpha + I_bg * (1 - alpha)
    
    def memory_report(self, message="Memory usage"):
        """
        메모리 사용량에 대한 상세 보고서를 출력합니다.
        """
        torch.cuda.synchronize()
        
        current = torch.cuda.memory_allocated() / 1024 / 1024
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        print(f"{message}: Current={current:.2f}MB, Peak={peak:.2f}MB, Reserved={reserved:.2f}MB")
        
        # 주요 클래스 텐서 사용량 출력
        tensors_report = defaultdict(int)
        sizes_report = {}
        precision_report = {}
        
        # self 내 텐서 확인
        for name, obj in vars(self).items():
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                size_bytes = obj.nelement() * obj.element_size()
                tensors_report[name] = size_bytes
                sizes_report[name] = obj.shape
                precision_report[name] = obj.dtype
        
        # 정렬된 텐서 사용량 출력 (큰 것부터)
        if tensors_report:
            print("  Top tensors by memory usage:")
            sorted_tensors = sorted(tensors_report.items(), key=lambda x: x[1], reverse=True)
            for name, size in sorted_tensors[:5]:  # 상위 5개만 출력
                print(f"    {name}: {size/1024/1024:.2f}MB, shape={sizes_report[name]}, dtype={precision_report[name]}")
        
        # FP16 vs FP32 메모리 사용량 비교
        fp16_count = sum(1 for dtype in precision_report.values() if dtype == torch.float16)
        fp32_count = sum(1 for dtype in precision_report.values() if dtype == torch.float32)
        fp16_bytes = sum(size for name, size in tensors_report.items() if precision_report.get(name) == torch.float16)
        fp32_bytes = sum(size for name, size in tensors_report.items() if precision_report.get(name) == torch.float32)
        
        if fp16_count + fp32_count > 0:
            print(f"  FP16: {fp16_count} tensors, {fp16_bytes/1024/1024:.2f}MB")
            print(f"  FP32: {fp32_count} tensors, {fp32_bytes/1024/1024:.2f}MB")
        
        return current, peak, reserved

    def clear_cuda_cache(self):
        """
        CUDA 캐시를 완전히 비우고 가비지 컬렉션을 수행합니다.
        """
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        # 메모리 사용량 최소화를 위해 peak 기록 재설정
        torch.cuda.reset_peak_memory_stats()
        
        current = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"After cache clearing: Current={current:.2f}MB, Reserved={reserved:.2f}MB")

    def create_stream(self):
        """
        새로운 CUDA 스트림을 생성하여 메모리 작업을 비동기적으로 처리합니다.
        이를 통해 여러 연산을 동시에 수행할 수 있어 메모리 사용량이 감소할 수 있습니다.
        """
        return torch.cuda.Stream()

    def with_stream(self, stream):
        """
        주어진 스트림에서 코드 블록을 실행하기 위한 컨텍스트 매니저를 반환합니다.
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

def pretty_mem(x):       # byte → MB 단위 문자열
    return f"{x/1024/1024:8.2f} MB"

def tensor_vram_report(namespace: dict):
    """
    namespace(dict): globals()  또는 locals()
    Prints a table of all torch.Tensor objects (including .grad) on GPU.
    """
    seen_data_ptr = set()
    rows, total = [], 0
    for name, obj in namespace.items():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            # same storage 여러 변수에 공유될 수 있으니 data_ptr 기준 dedup
            ptr = obj.data_ptr()
            if ptr in seen_data_ptr:
                continue
            seen_data_ptr.add(ptr)

            size_bytes = obj.numel() * obj.element_size()
            rows.append((name, str(tuple(obj.shape)), obj.dtype, pretty_mem(size_bytes)))
            total += size_bytes

            # grad 버퍼도 있으면 같이 체크
            if obj.grad is not None:
                gbytes = obj.grad.numel() * obj.grad.element_size()
                rows.append((name + ".grad", str(tuple(obj.grad.shape)),
                             obj.grad.dtype, pretty_mem(gbytes)))
                total += gbytes

    # 정렬: 큰 순서
    rows.sort(key=lambda r: float(r[3].split()[0]), reverse=True)

    print(f"{'tensor':25} {'shape':20} {'dtype':9} {'mem':>10}")
    print("-"*70)
    for n,s,d,m in rows:
        print(f"{n:25} {s:20} {str(d):9} {m:>10}")
    print("-"*70)
    print(f"{'TOTAL':25} {'':20} {'':9} {pretty_mem(total):>10}")


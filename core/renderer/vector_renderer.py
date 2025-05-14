from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch import nn
import torch.utils.checkpoint as cp
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

        Args:
            x, y:       [N] position coordinates for N shapes
            r:          [N] scales
            theta:      [N] rotations
            sigma:      Gaussian blur std
        Returns:
            masks:     [N, H, W]  (one soft mask per shape)
        """
        # Choose context manager based on precision mode
        context = autocast('cuda') if self.use_fp16 else nullcontext()
        
        with context:
            B = len(x)
            _, H, W = self.X.shape
            
            # Get target dtype based on use_fp16 flag
            target_dtype = torch.float16 if self.use_fp16 else torch.float32
            
            # Apply Gaussian blur if needed
            if sigma > 0.0:
                bmp = self.S.unsqueeze(0)
                bmp = gaussian_blur(bmp, sigma)
                bmp_image = bmp.squeeze(0).to(dtype=target_dtype).contiguous()
            else:
                bmp_image = self.S.to(dtype=target_dtype)
            
            # Expand parameters to match grid dimensions and convert to appropriate precision
            X_exp = self.X.expand(B, H, W)#.to(dtype=target_dtype)
            Y_exp = self.Y.expand(B, H, W)#.to(dtype=target_dtype)
            x_exp = x.view(B, 1, 1).expand(B, H, W)#.to(dtype=target_dtype)
            y_exp = y.view(B, 1, 1).expand(B, H, W)#.to(dtype=target_dtype)
            r_exp = r.view(B, 1, 1).expand(B, H, W)#.to(dtype=target_dtype)
            
            # Position normalization and rotation
            pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
            cos_t = torch.cos(theta)#.to(dtype=target_dtype)
            sin_t = torch.sin(theta)#.to(dtype=target_dtype)
            R_inv = torch.zeros(B, 2, 2, device=self.device)#, dtype=target_dtype)
            R_inv[:, 0, 0] = cos_t
            R_inv[:, 0, 1] = sin_t
            R_inv[:, 1, 0] = -sin_t
            R_inv[:, 1, 1] = cos_t
            uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
            
            # Prepare for grid sampling
            grid = uv.permute(0, 2, 3, 1)  # (B, H, W, 2)
            bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1).contiguous() #.to(dtype=target_dtype)
            
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
    
    def _tree_over(self, m: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient tree-based alpha compositing.
        
        Args:
            m: Color tensor
            a: Alpha tensor
            
        Returns:
            Tuple of (composited color, composited alpha)
        """
        if self.use_fp16:
            with autocast('cuda'):
                '''
                N0 = m.size(0)
                # 2의 거듭제곱으로 맞출 크기 계산
                Npad = 1 << (N0 - 1).bit_length()
                if Npad != N0:
                    pad = Npad - N0
                    pad_m = torch.zeros((pad, *m.shape[1:]), device=m.device, dtype=m.dtype)
                    pad_a = torch.zeros((pad, *a.shape[1:]), device=a.device, dtype=a.dtype)
                    m = torch.cat([m, pad_m], dim=0)
                    a = torch.cat([a, pad_a], dim=0)

                # 반복문 내에선 더 이상 torch.cat/zeros 호출 없음
                while m.size(0) > 1:
                    n2 = m.size(0) // 2
                    m = m.view(n2, 2, *m.shape[1:])  # view: 메모리 추가 없이 reshape
                    a = a.view(n2, 2, *a.shape[1:])
                    inv = (1 - a[:, 0]).unsqueeze(-1)  # shape=(n2,H,W,1)
                    # in-place update
                    m0 = m[:, 0]
                    m0.mul_(inv).add_(m[:, 1])
                    a0 = a[:, 0]
                    a0.mul_(inv.squeeze(-1)).add_(a[:, 1])
                    m, a = m0, a0

                return m.squeeze(0), a.squeeze(0)
                '''
                while m.size(0) > 1:
                    n = m.size(0)
                    if n % 2 == 1:
                        pad_m = torch.zeros((1, *m.shape[1:]), device=m.device, dtype=m.dtype)
                        pad_a = torch.zeros((1, *a.shape[1:]), device=a.device, dtype=a.dtype)
                        m = torch.cat([m, pad_m], dim=0)
                        a = torch.cat([a, pad_a], dim=0)
                        n += 1
                    new_n = n // 2
                    # reshape은 view → 메모리 추가 없음
                    m = m.view(new_n, 2, *m.shape[1:])
                    a = a.view(new_n, 2, *a.shape[1:])
                    # pairwise compositing (in-place로 덮어쓸 수도 있음)
                    m = m[:,0] + (1 - a[:,0]).unsqueeze(-1) * m[:,1]
                    a = a[:,0] + (1 - a[:,0]) * a[:,1]
                return m.squeeze(0), a.squeeze(0)
        else:
            while m.size(0) > 1:
                n = m.size(0)
                if n % 2 == 1:
                    pad_m = torch.zeros((1, *m.shape[1:]), device=m.device, dtype=m.dtype)
                    pad_a = torch.zeros((1, *a.shape[1:]), device=a.device, dtype=a.dtype)
                    m = torch.cat([m, pad_m], dim=0)
                    a = torch.cat([a, pad_a], dim=0)
                    n = m.size(0)
                new_n = n // 2
                m = m.reshape(new_n, 2, m.size(1), m.size(2), 3)
                a = a.reshape(new_n, 2, a.size(1), a.size(2))
                m = m[:, 0] + (1 - a[:, 0]).unsqueeze(-1) * m[:, 1]
                a = a[:, 0] + (1 - a[:, 0]) * a[:, 1]
                
            return m.squeeze(0), a.squeeze(0)
    
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
            cached_masks = cached_masks#.to(dtype=target_dtype)
            v = v#.to(dtype=target_dtype)
            c = c#.to(dtype=target_dtype)
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
        target_dtype = x.dtype  # Use the same dtype as input tensors

        # 1) Pre-blur the source bitmap once
        with torch.no_grad():
            bmp = self.S.unsqueeze(0)
            if sigma > 0:
                bmp = gaussian_blur(bmp, sigma)
                #bmp_processed = bmp.to(dtype=target_dtype).squeeze(0)  # [C_bmp, H, W]
                bmp_processed = bmp.squeeze(0)  # [C_bmp, H, W]
            else:
                bmp_processed = self.S#.to(dtype=target_dtype)
            del bmp

        # 2) Prepare alphas & colors
        with torch.no_grad():
            # Process v and c in smaller chunks
            chunk_size = min(100, max(1, N // 2))
            
            # 3) Running composite buffers - use local variables for checkpointing compatibility
            comp_m = torch.zeros((H, W, 3), device=device, dtype=target_dtype)
            comp_a = torch.zeros((H, W), device=device, dtype=target_dtype)
        
        # 4) Process in small shape‐chunks
        for i in range(0, N, raster_chunk_size):
            j = min(i + raster_chunk_size, N)
            curr_chunk_size = j - i
            
            # Calculate visibility and color for current chunk
            with torch.no_grad():
                v_chunk = v[i:j]#.to(dtype=target_dtype)
                c_chunk = c[i:j]#.to(dtype=target_dtype)
                v_alpha = (self.alpha_upper_bound * torch.sigmoid(v_chunk)).view(curr_chunk_size, 1, 1)#.to(dtype=target_dtype)
                c_eff = torch.sigmoid(c_chunk).view(curr_chunk_size, 1, 1, 3)#.to(dtype=target_dtype)
                
                # Get position parameters
                x_chunk = x[i:j]#.to(dtype=target_dtype)
                y_chunk = y[i:j]#.to(dtype=target_dtype)
                r_chunk = r[i:j]#.to(dtype=target_dtype)
                theta_chunk = theta[i:j]#.to(dtype=target_dtype)
                
            # a) Rasterize just this subset
            context = autocast('cuda') if self.use_fp16 else nullcontext()
            with context:
                masks_chunk = self._batched_soft_rasterize(
                    x_chunk, y_chunk,
                    r_chunk, theta_chunk,
                    sigma=0.0                # already applied blur to bmp_processed
                )#.to(dtype=target_dtype)     # [C, H, W]
                
                # b) Split into smaller sub-chunks instead of compositing all masks at once
                sub_chunk_size = 5  # small sub-chunk size
                
                for k in range(0, curr_chunk_size, sub_chunk_size):
                    end_k = min(k + sub_chunk_size, curr_chunk_size)
                    
                    # Current sub-chunk masks, alphas, colors
                    masks_subchunk = masks_chunk[k:end_k]
                    v_subchunk = v_alpha[k:end_k]
                    c_subchunk = c_eff[k:end_k]
                    
                    # Sequential compositing for each shape
                    for s in range(end_k - k):
                        a_s = v_subchunk[s] * masks_subchunk[s]          # [H, W]
                        m_s = a_s.unsqueeze(-1) * c_subchunk[s]       # [H, W, 3]
                        
                        # Use inplace operations
                        inv_a_s = 1.0 - a_s
                        comp_m = m_s + inv_a_s.unsqueeze(-1) * comp_m
                        comp_a = a_s + inv_a_s * comp_a
                        
                        # Free temporary tensors immediately if in FP16 mode
                        if self.use_fp16:
                            del a_s, m_s, inv_a_s
                    
                    # Free sub-chunk tensors if in FP16 mode
                    if self.use_fp16:
                        del masks_subchunk, v_subchunk, c_subchunk
                        torch.cuda.empty_cache()

            # Free chunk tensors if in FP16 mode
            if self.use_fp16:
                del masks_chunk, v_chunk, c_chunk, v_alpha, c_eff, x_chunk, y_chunk, r_chunk, theta_chunk
                torch.cuda.empty_cache()

        # 5) Finalize with white background
        with torch.no_grad():
            final = comp_m + (1 - comp_a).unsqueeze(-1)
            # Free memory before return if in FP16 mode
            if self.use_fp16:
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
        
        if do_sparsify:
            # Create separate optimizers for sparsification
            optimizer_xyr = torch.optim.Adam([
                {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            ])
            
            optimizer_rest = torch.optim.Adam([
                {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0/x.numel())},
                {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0) * (1000.0/x.numel())},
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
            
            assert sparsified_N < N, "sparsified_N must be less than N"
            assert num_iterations > iters_warmup + sparsify_duration, "num_iterations must be greater than warmup + duration"
            
            # For Gaussian blur transition if enabled
            do_gaussian_blur = opt_conf.get("do_gaussian_blur", False)
            sigma_start = opt_conf.get("blur_sigma_start", 0.0)
            sigma_end = opt_conf.get("blur_sigma_end", 0.0)
            
            optimizer = None  # We'll use the separate optimizers instead
        else:
            # Standard single optimizer if not doing sparsification
            param_groups = [
                {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
                {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0/x.numel())},
                {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0) * (1000.0/x.numel())},
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
                if do_gaussian_blur:
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
                        rendered = self.render(self.S, v, c)
                    else:
                        if streaming_render:
                            rendered = self._stream_render(
                                x, y, r, theta,
                                v, c,
                                sigma=sigma,
                                raster_chunk_size=raster_chunk_size
                            )
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
                            if opt_conf.get("save_epoch", False) and epoch + 1 in save_image_intervals:
                                output_path = os.path.join(self.output_path, f"epoch_{epoch+1}_{timestamp}.jpg")
                                rendered_np = rendered.detach().cpu().numpy()
                                rendered_np = (rendered_np * 255).astype(np.uint8)
                                Image.fromarray(rendered_np).save(output_path)
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
                        _alpha = self.alpha_upper_bound * torch.sigmoid(v)
                        loss += 0.5 * sparsify_loss_coeff * F.mse_loss(_alpha, z - lam)  # Sparsification loss
                
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
                    if not streaming_render and not opt_conf.get("multi_level", False) and 'cached_masks_ref' in locals():
                        del cached_masks_ref
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
                    if opt_conf.get("save_epoch", False) and epoch + 1 in save_image_intervals:
                        output_path = os.path.join(self.output_path, f"epoch_{epoch+1}_{timestamp}.jpg")
                        rendered_np = rendered.detach().cpu().numpy()
                        rendered_np = (rendered_np * 255).astype(np.uint8)
                        Image.fromarray(rendered_np).save(output_path)
                        del rendered_np
                
                # Compute loss
                loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)
                
                # Add sparsification loss if needed
                if do_sparsify and epoch >= iters_warmup and epoch < iters_warmup + sparsify_duration:
                    _alpha = self.alpha_upper_bound * torch.sigmoid(v)
                    loss += 0.5 * sparsify_loss_coeff * F.mse_loss(_alpha, z - lam)  # Sparsification loss
                
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
                theta.clamp_(0, 2 * np.pi)
                
                # Sparsification logic
                if do_sparsify:
                    if epoch >= iters_warmup and epoch <= iters_warmup + sparsify_duration:
                        if epoch % sparsifying_period == 0:
                            # Sparsification step
                            _alpha = self.alpha_upper_bound * torch.sigmoid(v.detach())
                            keep_idx = torch.topk(_alpha, sparsified_N).indices
                            mask = torch.zeros_like(_alpha, dtype=torch.bool)
                            mask[keep_idx] = True
                            
                            z.zero_(); z[mask] = _alpha[mask]  # z ← sparse projection
                            lam += (_alpha - z)
                        
                        if epoch == iters_warmup + sparsify_duration:
                            # actual sparsification
                            # 1) helper: 기존 tensor -> 잘라낸 tensor(requires_grad 유지)
                            def _prune(t):
                                t_new = t[keep_idx].clone().detach()
                                if t.requires_grad:
                                    t_new.requires_grad_(True)
                                return t_new
                            
                            # 2) 실제 파라미터·보조변수 잘라내기
                            x, y, r, theta, v, c = map(_prune, (x, y, r, theta, v, c))
                            z, lam = map(_prune, (z, lam))
                            
                            # 3) optimizer 재생성 (Adam state 깔끔히 초기화)
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
                            
                            sched_xyr = torch.optim.lr_scheduler.ExponentialLR(
                                optimizer_xyr, gamma=opt_conf.get("decay_rate", 0.99)) if do_decay else None
            
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
        num_iterations = opt_conf.get("num_iterations", 100)
        chunk = opt_conf.get("batch_size", 256)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Create output directory for saving images if it doesn't exist
        save_image_intervals = [1, 5, 10, 20, 50, 100]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_path, exist_ok=True)
        
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
                x, y, r, theta, sigma=opt_conf.get("blur_sigma", 1.0)
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
                    sigma=opt_conf.get("blur_sigma", 1.0)
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
                    sigma=opt_conf.get("blur_sigma", 1.0)        # 동일 블러 파라미터 사용
                )

            # Save image at specified epochs
            if opt_conf.get("save_epoch", False) and step + 1 in save_image_intervals:
                with torch.no_grad():
                    # Render the complete image for saving
                    rendered_img = self.render(cached_masks, v, c)
                    output_path = os.path.join(self.output_path, f"epoch_{step+1}_{timestamp}.jpg")
                    rendered_np = rendered_img.detach().cpu().numpy()
                    rendered_np = (rendered_np * 255).astype(np.uint8)
                    Image.fromarray(rendered_np).save(output_path)
                    del rendered_img, rendered_np
                    torch.cuda.empty_cache()
            
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


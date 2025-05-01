import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
from util.utils import gaussian_blur, make_batch_indices
import os
import gc

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
        Generate soft masks for each primitive.
        
        Args:
            x, y: Position coordinates
            r: Scale (radius)
            theta: Rotation angle
            sigma: Gaussian blur standard deviation
            
        Returns:
            Tensor of shape (B, H, W) containing soft masks
        """
        B = len(x)
        _, H, W = self.X.shape
        
        # Apply Gaussian blur if needed
        if sigma > 0.0:
            bmp = self.S.unsqueeze(0)
            bmp = gaussian_blur(bmp, sigma)
            bmp_image = bmp.squeeze(0)
        else:
            bmp_image = self.S
        
        # Expand parameters to match grid dimensions
        X_exp = self.X.expand(B, H, W)
        Y_exp = self.Y.expand(B, H, W)
        x_exp = x.view(B, 1, 1).expand(B, H, W)
        y_exp = y.view(B, 1, 1).expand(B, H, W)
        r_exp = r.view(B, 1, 1).expand(B, H, W)
        
        # Position normalization and rotation
        pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        R_inv = torch.zeros(B, 2, 2, device=self.device)
        R_inv[:, 0, 0] = cos_t
        R_inv[:, 0, 1] = sin_t
        R_inv[:, 1, 0] = -sin_t
        R_inv[:, 1, 1] = cos_t
        uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
        
        # Prepare for grid sampling
        grid = uv.permute(0, 2, 3, 1)  # (B, H, W, 2)
        bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        
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
        N = v.shape[0]

        # 1. per-primitive alpha & color
        v_alpha = self.alpha_upper_bound * torch.sigmoid(v).view(N, 1, 1)
        a       = v_alpha * cached_masks                     # (N, H, W)
        c_eff   = torch.sigmoid(c).view(N, 1, 1, 3)          # (N, 1, 1, 3)
        m       = a.unsqueeze(-1) * c_eff                    # (N, H, W, 3)

        # 2. Porter–Duff reduction (tree)
        if self.use_checkpointing:
            comp_m, comp_a = torch.utils.checkpoint.checkpoint(
                lambda mm, aa: self._tree_over(mm, aa),
                m, a, use_reentrant=False
            )
        else:
            comp_m, comp_a = self._tree_over(m, a)           # comp_a: (H, W)

        if return_alpha:
            # (H, W) → (H, W, 1) 로 맞춰 두면 브로드캐스트 편함
            return comp_m, comp_a.unsqueeze(-1)

        # 3. 흰 배경까지 합성해서 완성 RGB
        background = torch.ones_like(comp_m)                 # white
        final = comp_m + (1.0 - comp_a).unsqueeze(-1) * background
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
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
            {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
            {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},
            {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
            {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
        ])
        
        print(f"Starting optimization for {num_iterations} iterations...")
        for epoch in tqdm(range(num_iterations)):
            optimizer.zero_grad()
           
            # Render image
            if opt_conf.get("multi_level", False):
                rendered = self.render(self.S, v, c)
            else:
                cached_masks = self._batched_soft_rasterize(
                    x, y, r, theta,
                    sigma=opt_conf.get("blur_sigma", 0.0)
                )
                rendered = self.render(cached_masks, v, c)
            
            # Compute loss
            loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Clamp parameters
            with torch.no_grad():
                x.clamp_(0, self.W)
                y.clamp_(0, self.H)
                r.clamp_(2, min(self.H, self.W) // 4)
                theta.clamp_(0, 2 * np.pi)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
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
                    # rgb_fg, alpha_fg = self._tree_over(
                    #     m      = cached_masks[fg_sl].unsqueeze(-1) *
                    #             torch.sigmoid(c[fg_sl]).view(-1,1,1,3),
                    #     a      = self.alpha_upper_bound *
                    #             torch.sigmoid(v[fg_sl]).view(-1,1,1)
                    # )
                    rgb_fg, alpha_fg = self.render(
                        cached_masks[fg_sl], v[fg_sl], c[fg_sl],
                        return_alpha=True
                    )
                else:  # 아무것도 없으면 투명
                    rgb_fg  = torch.zeros_like(target_image)
                    alpha_fg = torch.zeros_like(target_image[..., :1])
                # 뒤 레이어 → rgb_bg (흰 배경까지 합성, alpha 버림)
                if end < N:
                    rgb_bg = self.render(
                        cached_masks[bg_sl], v[bg_sl], c[bg_sl],
                        return_alpha=False
                    )
                else:
                    rgb_bg = torch.ones_like(target_image)

            # -------- 2. 학습 배치 forward --------
            optimizer.zero_grad()
            # Render image
            if opt_conf.get("multi_level", False):
                # rendered = self.render(self.S, v, c)
                raise NotImplementedError("Multi-level rendering not implemented")
            else:
                masks_bt = self._batched_soft_rasterize(
                    x[batch_sl], y[batch_sl], r[batch_sl], theta[batch_sl],
                    sigma=opt_conf.get("blur_sigma", 0.0)
                )
                rgb_bt, alpha_bt = self.render(
                        masks_bt, v[batch_sl], c[batch_sl],
                        return_alpha=True
                    )
            # -------- 3. Porter-Duff over(fg → batch → bg) --------
            comp1 = rgb_fg + (1 - alpha_fg) * rgb_bt
            final = comp1 + (1 - alpha_fg) * (1 - alpha_bt) * rgb_bg

            loss = self.compute_loss(final, target_image,
                                    x, y, r, v, theta, c)
            loss.backward()
            optimizer.step()            
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


import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
from util.utils import gaussian_blur
import os

class VectorRenderer:
    """
    A class for rendering vector graphics using differentiable primitives.
    This class handles the core rendering functionality including mask generation,
    alpha compositing, and parameter optimization.
    """
    def __init__(self, 
                 canvas_size: Tuple[int, int],
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
                               bmp_image: torch.Tensor,
                               x: torch.Tensor,
                               y: torch.Tensor,
                               r: torch.Tensor,
                               theta: torch.Tensor,
                               sigma: float = 0.0) -> torch.Tensor:
        """
        Generate soft masks for each primitive.
        
        Args:
            bmp_image: Base bitmap image tensor
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
            bmp = bmp_image.unsqueeze(0)
            bmp = gaussian_blur(bmp, sigma)
            bmp_image = bmp.squeeze(0)
        
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
    
    def render(self,
               cached_masks: torch.Tensor,
               v: torch.Tensor,
               c: torch.Tensor) -> torch.Tensor:
        """
        Render the final image using cached masks and parameters.
        
        Args:
            cached_masks: Pre-computed masks for each primitive
            v: Visibility parameters
            c: Color parameters
            
        Returns:
            Rendered image tensor of shape (H, W, 3)
        """
        N = v.shape[0]
        v_alpha = self.alpha_upper_bound * torch.sigmoid(v).view(N, 1, 1)
        a = v_alpha * cached_masks
        c_eff = torch.sigmoid(c).view(N, 1, 1, 3)
        m = a.unsqueeze(-1) * c_eff
        
        # Use gradient checkpointing if enabled
        if self.use_checkpointing:
            def tree_over_func(x, y):
                return self._tree_over(x, y)
            comp_m, comp_a = torch.utils.checkpoint.checkpoint(tree_over_func, m, a, use_reentrant=False)
        else:
            comp_m, comp_a = self._tree_over(m, a)
            
        background = torch.ones_like(comp_m)
        final = comp_m + (1 - comp_a).unsqueeze(-1) * background
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
                          bmp_image: torch.Tensor,
                          opt_conf: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        Optimize the rendering parameters to match the target image.
        
        Args:
            x, y, r, v, theta, c: Initial parameters
            target_image: Target image to match
            bmp_image: Base bitmap image for rasterization
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
                rendered = self.render(bmp_image, v, c)
            else:
                cached_masks = self._batched_soft_rasterize(
                    bmp_image, x, y, r, theta,
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
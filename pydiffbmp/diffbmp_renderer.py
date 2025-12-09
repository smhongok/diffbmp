"""
High-level API for applying image overlays to images using DiffBMP rendering.

This module provides a simple, pip-package-style interface for image rendering
that can be easily integrated into diffusion model pipelines.

Example:
    >>> from pydiffbmp import DiffBMPRenderer
    >>> 
    >>> renderer = DiffBMPRenderer(
    ...     overlay_path="logo.png",
    ...     canvas_size=(256, 256),
    ...     device='cuda'
    ... )
    >>> 
    >>> # In diffusion loop
    >>> output = renderer.apply(image_tensor, alpha=0.3)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional, Literal
from pathlib import Path

from pydiffbmp.util.primitive_loader import PrimitiveLoader
from pydiffbmp.core.renderer.simple_tile_renderer import SimpleTileRenderer

# Optimization constants
DEFAULT_NUM_ITERATIONS = 50
DEFAULT_LEARNING_RATE = 0.01

class DiffBMPRenderer:
    """
    High-level DiffBMP renderer for easy integration with diffusion models.
    
    This class handles all the complexity of DiffBMP rendering and provides
    a simple API for applying overlays to image tensors.
    
    Args:
        overlay_path: Path to overlay image (PNG, JPG, or SVG)
        canvas_size: (height, width) of the target images
        device: 'cuda' or 'cpu'
        position: Where to place overlay ('center', 'bottom-right', 'top-left', 
                 'top-right', 'bottom-left', or (x, y) tuple)
        scale: Scale factor for overlay size (0.0-1.0 relative to canvas)
        alpha_range: (min, max) opacity range for overlay
        tile_size: Tile size for rendering (larger = more memory, faster)
        use_fp16: Use half precision for faster rendering
        cache_primitive: Cache loaded primitive for repeated use
    """
    
    def __init__(
        self,
        overlay_path: Union[str, Path],
        canvas_size: Tuple[int, int],
        device: str = 'cuda',
        position: Union[str, Tuple[float, float]] = 'bottom-right',
        scale: float = 0.15,
        alpha_range: Tuple[float, float] = (0.3, 0.5),
        tile_size: int = 32,
        use_fp16: bool = False,
        cache_primitive: bool = True,
    ):
        self.overlay_path = Path(overlay_path)
        self.canvas_size = canvas_size  # (H, W)
        self.device = torch.device(device)
        self.position = position
        self.scale = scale
        self.alpha_range = alpha_range
        self.tile_size = tile_size
        self.use_fp16 = use_fp16
        self.cache_primitive = cache_primitive
        
        # Cache for primitive and renderer
        self._primitive_cache = None
        self._primitive_colors = None
        self._renderer = None
        
        # Pre-compute fixed parameters
        self._fixed_params = None
        
        # Initialize primitive
        if cache_primitive:
            self._load_primitive()
    
    def _load_primitive(self):
        """Load and cache the overlay primitive."""
        H, W = self.canvas_size
        
        # Determine primitive size based on scale
        prim_size = int(min(H, W) * self.scale)
        
        # Load primitive using PrimitiveLoader
        loader = PrimitiveLoader(
            str(self.overlay_path),
            output_width=prim_size,
            device=self.device,
            radial_transparency=False,
        )
        
        self._primitive_cache = loader.load_alpha_bitmap()
        if self.use_fp16:
            self._primitive_cache = self._primitive_cache.to(dtype=torch.float16)
        
        self._primitive_colors = loader.get_primitive_color_maps()
        
        print(f"✓ Overlay primitive loaded: {self._primitive_cache.shape}")
        if self._primitive_colors is not None and len(self._primitive_colors) > 0:
            # Flatten and compute mean color across all pixels
            # Handle different possible shapes
            color_tensor = self._primitive_colors
            while color_tensor.dim() > 2:
                color_tensor = color_tensor.reshape(-1, color_tensor.shape[-1])
            if color_tensor.dim() == 2:
                mean_color = color_tensor.mean(dim=0)  # (3,)
            else:
                mean_color = color_tensor  # Already (3,)
            
            # Ensure mean_color is 1D with 3 elements
            if mean_color.numel() == 3:
                r, g, b = mean_color[0].item(), mean_color[1].item(), mean_color[2].item()
                print(f"  Primitive colors: shape={self._primitive_colors.shape}, mean RGB=({r:.3f}, {g:.3f}, {b:.3f})")
            else:
                print(f"  ⚠️ Unexpected primitive color shape: {self._primitive_colors.shape}")
        else:
            print(f"  ⚠️ No primitive colors loaded (will use default)")
    
    def _get_renderer(self):
        """Get or create renderer."""
        if self._renderer is None:
            if self._primitive_cache is None:
                self._load_primitive()
            
            H, W = self.canvas_size
            
            # Create renderer with safe defaults
            try:
                # Use small tile_size to ensure parallel processing (avoids sequential path bugs)
                # Parallel path is required for proper gradient computation
                # For small images (<=64), use tile_size=12 to get >4 tiles and trigger parallel path
                effective_tile_size = 12 if min(H, W) <= 64 else self.tile_size
                
                self._renderer = SimpleTileRenderer(
                    canvas_size=(H, W),
                    S=self._primitive_cache,
                    alpha_upper_bound=self.alpha_range[1],
                    device=self.device,
                    use_fp16=self.use_fp16,
                    tile_size=effective_tile_size,
                    sigma=0.0,  # No blur for watermark
                    c_blend=1.0,  # Use primitive colors (1.0 = full color, 0.0 = use c parameter)
                    primitive_colors=self._primitive_colors,
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize SimpleTileRenderer: {e}")
                print(f"   Canvas size: {(H, W)}, Primitive shape: {self._primitive_cache.shape}")
                raise
        
        return self._renderer
    
    def _compute_position(self, H: int, W: int) -> Tuple[float, float]:
        """Compute (x, y) position based on position string or tuple."""
        if isinstance(self.position, tuple):
            return self.position
        
        # Compute overlay size
        r = min(H, W) * self.scale / 2.0  # radius
        
        # Padding from edges
        padding = r * 1.2
        
        position_map = {
            'center': (W / 2.0, H / 2.0),
            'bottom-right': (W - padding, H - padding),
            'bottom-left': (padding, H - padding),
            'top-right': (W - padding, padding),
            'top-left': (padding, padding),
        }
        
        if self.position not in position_map:
            raise ValueError(f"Unknown position: {self.position}. "
                           f"Use one of {list(position_map.keys())} or (x, y) tuple")
        
        return position_map[self.position]
    
    def _get_fixed_params(self, H: int, W: int):
        """Get or compute fixed rendering parameters."""
        if self._fixed_params is None or self._fixed_params['canvas_size'] != (H, W):
            x, y = self._compute_position(H, W)
            r = min(H, W) * self.scale / 2.0
            
            # Create parameter tensors (single instance)
            x_param = torch.tensor([x], device=self.device, dtype=torch.float32)
            y_param = torch.tensor([y], device=self.device, dtype=torch.float32)
            r_param = torch.tensor([r], device=self.device, dtype=torch.float32)
            theta_param = torch.tensor([0.0], device=self.device, dtype=torch.float32)
            
            # Opacity in logit space
            alpha_mean = (self.alpha_range[0] + self.alpha_range[1]) / 2.0
            v_logit = torch.logit(torch.tensor([alpha_mean]))
            v_param = v_logit.to(device=self.device, dtype=torch.float32)
            
            # Color modulation parameter
            # Note: c is (N, 3) shape - one RGB multiplier per primitive
            # primitive_colors (1, H, W, 3) is already passed to SimpleTileRenderer separately
            # Use white (1, 1, 1) to not modify the primitive colors
            c_param = torch.ones((1, 3), device=self.device, dtype=torch.float32)
            
            self._fixed_params = {
                'x': x_param,
                'y': y_param,
                'r': r_param,
                'theta': theta_param,
                'v': v_param,
                'c': c_param,
                'canvas_size': (H, W),
            }
        
        return self._fixed_params
    
    def apply(
        self,
        image: torch.Tensor,
        alpha: Optional[float] = None,
        position: Optional[Union[str, Tuple[float, float]]] = None,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply overlay to image tensor(s).
        
        Args:
            image: Image tensor in one of these formats:
                   - (B, C, H, W) - batch of images, range [-1, 1] or [0, 1]
                   - (C, H, W) - single image
                   - (H, W, C) - single image in HWC format
            alpha: Override opacity (0.0-1.0). If None, uses alpha_range mean
            position: Override position for this call
            return_mask: If True, also return the overlay mask
        
        Returns:
            Output image tensor in same format as input
            If return_mask=True, returns (output_image, mask)
        """
        # Store original format
        original_format, original_shape, original_dtype = self._parse_input(image)
        
        # Convert to (B, C, H, W) in [0, 1] range
        image_batch = self._to_batch_format(image)
        B, C, H, W = image_batch.shape
        
        # Override position if specified
        if position is not None:
            original_position = self.position
            self.position = position
            self._fixed_params = None  # Force recomputation
        
        # Get fixed parameters
        params = self._get_fixed_params(H, W)
        
        # Override alpha if specified
        if alpha is not None:
            v_logit = torch.logit(torch.tensor([alpha]))
            params['v'] = v_logit.to(device=self.device, dtype=torch.float32)
        
        # Get renderer
        renderer = self._get_renderer()
        
        # Render overlay for each image in batch
        output_batch = []
        masks = []
        
        for i in range(B):
            # Convert to (H, W, C) format for renderer
            img_hwc = image_batch[i].permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            
            # Render overlay (gradient tracking enabled)
            output_hwc = renderer.render_from_params(
                params['x'], params['y'], params['r'],
                params['theta'], params['v'], params['c'],
                sigma=0.0,
                I_bg=img_hwc,
            )
            
            # Convert back to (C, H, W)
            output_chw = output_hwc.permute(2, 0, 1)
            output_batch.append(output_chw)
            
            if return_mask:
                # Render mask (just the overlay on white background)
                white_bg = torch.ones_like(img_hwc)
                mask_hwc = renderer.render_from_params(
                    params['x'], params['y'], params['r'],
                    params['theta'], params['v'], params['c'],
                    sigma=0.0,
                    I_bg=white_bg,
                )
                # Extract alpha channel (difference from white)
                mask = 1.0 - (mask_hwc.mean(dim=-1, keepdim=True))  # (H, W, 1)
                masks.append(mask.squeeze(-1))
        
        # Stack batch
        result = torch.stack(output_batch, dim=0)  # (B, C, H, W)
        
        # Restore original position if overridden
        if position is not None:
            self.position = original_position
            self._fixed_params = None
        
        # Convert back to original format
        result = self._from_batch_format(result, original_format, original_shape, original_dtype)
        
        if return_mask:
            mask_result = torch.stack(masks, dim=0)  # (B, H, W)
            return result, mask_result
        
        return result
    
    def render(
        self,
        image: torch.Tensor,
        alpha: Optional[float] = None,
        position: Optional[Union[str, Tuple[float, float]]] = None,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Render overlay on image tensor(s). Alias for apply().
        
        This method provides a more intuitive name for DiffBMP rendering.
        
        Args:
            image: Image tensor in one of these formats:
                   - (B, C, H, W) - batch of images, range [-1, 1] or [0, 1]
                   - (C, H, W) - single image
                   - (H, W, C) - single image in HWC format
            alpha: Override opacity (0.0-1.0). If None, uses alpha_range mean
            position: Override position for this call
            return_mask: If True, also return the overlay mask
        
        Returns:
            Rendered image tensor in same format as input
            If return_mask=True, returns (rendered_image, mask)
        """
        return self.apply(image, alpha=alpha, position=position, return_mask=return_mask)
    
    def apply_with_optimization(
        self,
        image: torch.Tensor,
        target_image: Optional[torch.Tensor] = None,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        alpha: Optional[float] = None,
        position: Optional[Union[str, Tuple[float, float]]] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Apply overlay with gradient-based optimization.
        
        Args:
            image: Background image tensor
            target_image: Target image to optimize towards (default: rendered version)
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizer
            alpha: Override opacity
            position: Override position
            verbose: Print optimization progress
            
        Returns:
            Optimized output image
        """
        # Store original format
        original_format, original_shape, original_dtype = self._parse_input(image)
        
        # Convert to (B, C, H, W) in [0, 1] range
        image_batch = self._to_batch_format(image)
        B, C, H, W = image_batch.shape
        
        # Override position if specified
        if position is not None:
            original_position = self.position
            self.position = position
            self._fixed_params = None
        
        # Get fixed parameters
        params = self._get_fixed_params(H, W)
        
        # Override alpha if specified
        if alpha is not None:
            v_logit = torch.logit(torch.tensor([alpha]))
            params['v'] = v_logit.to(device=self.device, dtype=torch.float32)
        
        # Create parameters with gradient capability
        # Position, scale, rotation are FIXED (not in optimizer) but need gradients for chain
        x_opt = params['x'].clone().detach().requires_grad_(True)  # Fixed but needs grad
        y_opt = params['y'].clone().detach().requires_grad_(True)  # Fixed but needs grad
        r_opt = params['r'].clone().detach().requires_grad_(True)  # Fixed but needs grad
        theta_opt = params['theta'].clone().detach().requires_grad_(True)  # Fixed but needs grad
        
        # Opacity and color: start from random to find optimal values
        v_opt = (params['v'] + torch.randn_like(params['v']) * 1.0).clone().detach().requires_grad_(True)
        c_opt = (params['c'] + torch.randn_like(params['c']) * 0.3).clamp(0, 1).clone().detach().requires_grad_(True)
        
        if verbose:
            print(f"  📍 Fixed parameters:")
            print(f"     Position: ({x_opt.item():.1f}, {y_opt.item():.1f})")
            print(f"     Radius: {r_opt.item():.1f}")
            print(f"     Rotation: {theta_opt.item():.3f} rad")
            print(f"  🎨 Optimizing:")
            print(f"     Alpha (initial): {torch.sigmoid(v_opt).item():.3f} → target: {torch.sigmoid(params['v']).item():.3f}")
            print(f"     Color optimization enabled")
        
        # Setup optimizer (only for optimizable parameters: v and c)
        optimizer = torch.optim.Adam([v_opt, c_opt], lr=learning_rate)
        
        # Get renderer
        renderer = self._get_renderer()
        
        # Prepare background image (HWC format) - reused in loop  
        # Enable gradient for CUDA kernel, but won't optimize it
        img_hwc = image_batch[0].permute(1, 2, 0).contiguous()
        if not img_hwc.requires_grad:
            img_hwc = img_hwc.detach().requires_grad_(True)
        
        # Generate target if not provided
        if target_image is None:
            # Create target: input image with overlay fully applied
            with torch.no_grad():
                # Simply render with background
                target_hwc = renderer.render_from_params(
                    params['x'], params['y'], params['r'],
                    params['theta'], params['v'], params['c'],
                    sigma=0.0,
                    I_bg=img_hwc,
                )
                target_image = target_hwc.unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
        else:
            target_image = self._to_batch_format(target_image)
        
        if verbose:
            print(f"\n🔄 Starting optimization: {num_iterations} iterations")
        
        # Optimization loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Simply render with background - simpler and maintains gradient flow
            rendered_hwc = renderer.render_from_params(
                x_opt, y_opt, r_opt, theta_opt, v_opt, c_opt,
                sigma=0.0,
                I_bg=img_hwc,  # Use actual background directly
            )
            
            # Convert to (B, C, H, W) for loss computation
            rendered_chw = rendered_hwc.permute(2, 0, 1).unsqueeze(0)
            
            # Compute loss
            loss = F.mse_loss(rendered_chw, target_image)
            
            # Try to compute gradients using autograd
            try:
                grads = torch.autograd.grad(loss, [v_opt, c_opt], create_graph=False)
                v_opt.grad = grads[0]
                c_opt.grad = grads[1]
            except RuntimeError as e:
                # Fallback: Use numerical gradients when autograd fails
                if iteration == 0 and verbose:
                    print(f"  ⚠️ Using numerical gradients (autograd unavailable)")
                eps = 1e-3
                
                # Numerical gradient for v_opt
                with torch.no_grad():
                    # Store original loss value
                    loss_orig = loss.item() if hasattr(loss, 'item') else float(loss)
                    
                    # Compute gradient for v_opt
                    v_plus = v_opt + eps
                    rendered_plus = renderer.render_from_params(
                        x_opt, y_opt, r_opt, theta_opt, v_plus, c_opt,
                        sigma=0.0, I_bg=img_hwc
                    )
                    loss_plus = F.mse_loss(rendered_plus.permute(2,0,1).unsqueeze(0), target_image)
                    loss_plus_val = loss_plus.item() if hasattr(loss_plus, 'item') else float(loss_plus)
                    
                    # Ensure gradient has correct shape
                    grad_v = torch.tensor([(loss_plus_val - loss_orig) / eps], 
                                         device=v_opt.device, dtype=v_opt.dtype)
                    v_opt.grad = grad_v.reshape_as(v_opt)
                    
                    # Numerical gradient for c_opt (full computation)
                    grad_c = torch.zeros_like(c_opt)
                    for i in range(c_opt.shape[1]):  # For each color channel
                        c_plus = c_opt.clone()
                        c_plus[0, i] += eps
                        rendered_plus = renderer.render_from_params(
                            x_opt, y_opt, r_opt, theta_opt, v_opt, c_plus,
                            sigma=0.0, I_bg=img_hwc
                        )
                        loss_plus = F.mse_loss(rendered_plus.permute(2,0,1).unsqueeze(0), target_image)
                        loss_plus_val = loss_plus.item() if hasattr(loss_plus, 'item') else float(loss_plus)
                        grad_c[0, i] = (loss_plus_val - loss_orig) / eps
                    
                    c_opt.grad = grad_c
            
            # Optimization step
            optimizer.step()
            
            if verbose:
                print(f"  Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.10f}")
        
        if verbose:
            print(f"✓ Optimization complete! Final loss: {loss.item():.10f}\n")
        
        # Final rendering with optimized parameters (gradient tracking enabled)
        img_hwc_final = image_batch[0].permute(1, 2, 0).contiguous()
        final_hwc = renderer.render_from_params(
            x_opt, y_opt, r_opt, theta_opt, v_opt, c_opt,
            sigma=0.0,
            I_bg=img_hwc_final,
        )
        final_chw = final_hwc.permute(2, 0, 1)
        result = final_chw.unsqueeze(0)
        
        # Restore original position if overridden
        if position is not None:
            self.position = original_position
            self._fixed_params = None
        
        # Convert back to original format
        result = self._from_batch_format(result, original_format, original_shape, original_dtype)
        
        return result
    
    def render_with_optimization(
        self,
        image: torch.Tensor,
        target_image: Optional[torch.Tensor] = None,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        alpha: Optional[float] = None,
        position: Optional[Union[str, Tuple[float, float]]] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Render overlay with gradient-based optimization. Alias for apply_with_optimization().
        
        This method provides a more intuitive name for DiffBMP rendering.
        
        Args:
            image: Background image tensor
            target_image: Target image to optimize towards (default: rendered version)
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizer
            alpha: Override opacity
            position: Override position
            verbose: Print optimization progress
            
        Returns:
            Optimized rendered image
        """
        return self.apply_with_optimization(
            image, 
            target_image=target_image,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            alpha=alpha,
            position=position,
            verbose=verbose
        )
    
    def _parse_input(self, image: torch.Tensor):
        """Determine input format and properties."""
        shape = image.shape
        dtype = image.dtype
        
        if len(shape) == 4:  # (B, C, H, W)
            format_type = 'BCHW'
        elif len(shape) == 3:
            # Check if (C, H, W) or (H, W, C)
            if shape[0] in [1, 3, 4]:  # Likely (C, H, W)
                format_type = 'CHW'
            else:  # Likely (H, W, C)
                format_type = 'HWC'
        else:
            raise ValueError(f"Unsupported image shape: {shape}")
        
        return format_type, shape, dtype
    
    def _to_batch_format(self, image: torch.Tensor) -> torch.Tensor:
        """Convert to (B, C, H, W) in [0, 1] range."""
        # Determine format
        format_type, _, _ = self._parse_input(image)
        
        # Convert to (B, C, H, W)
        if format_type == 'BCHW':
            img = image
        elif format_type == 'CHW':
            img = image.unsqueeze(0)
        elif format_type == 'HWC':
            img = image.permute(2, 0, 1).unsqueeze(0)
        
        # Ensure [0, 1] range and track if we converted
        self._converted_from_neg1_pos1 = False
        if img.min() < -0.1:  # Likely [-1, 1]
            img = (img + 1.0) / 2.0
            self._converted_from_neg1_pos1 = True
        
        # Ensure on correct device
        img = img.to(self.device)
        
        return img
    
    def _from_batch_format(
        self,
        image: torch.Tensor,
        original_format: str,
        original_shape: tuple,
        original_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert from (B, C, H, W) back to original format."""
        # Handle batch dimension
        if original_format == 'CHW':
            img = image.squeeze(0)
        elif original_format == 'HWC':
            img = image.squeeze(0).permute(1, 2, 0)
        else:  # BCHW
            img = image
        
        # Restore value range if needed
        if hasattr(self, '_converted_from_neg1_pos1') and self._converted_from_neg1_pos1:
            # Convert back from [0, 1] to [-1, 1]
            img = (img * 2.0) - 1.0
        
        return img.to(dtype=original_dtype)
    
    def update_position(self, position: Union[str, Tuple[float, float]]):
        """Update overlay position."""
        self.position = position
        self._fixed_params = None
    
    def update_alpha(self, alpha: float):
        """Update overlay opacity."""
        alpha = np.clip(alpha, 0.0, 1.0)
        self.alpha_range = (alpha, alpha)
        self._fixed_params = None
    
    def clear_cache(self):
        """Clear all cached data."""
        self._primitive_cache = None
        self._primitive_colors = None
        self._renderer = None
        self._fixed_params = None
    
    def __repr__(self):
        return (f"DiffBMPRenderer(overlay='{self.overlay_path.name}', "
                f"canvas_size={self.canvas_size}, position='{self.position}', "
                f"scale={self.scale}, alpha_range={self.alpha_range})")

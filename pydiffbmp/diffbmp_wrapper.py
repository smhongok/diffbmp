"""
PyDiffBMP Functional API with Wrapper Class

Provides a class-based interface for differentiable rendering with bitmap primitives.
Parameters (x, y, r, theta, v, c) are passed as function arguments for flexibility.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional, Dict, Any
from pathlib import Path
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
from pydiffbmp.core.renderer.simple_tile_renderer import SimpleTileRenderer
from pydiffbmp.util.svg_loader import SVGLoader
from pydiffbmp.util.primitive_loader import PrimitiveLoader
from pydiffbmp.core.initializer.random_initializater import RandomInitializer
from pydiffbmp.core.initializer.svgsplat_initializater import StructureAwareInitializer
from pydiffbmp.core.initializer.designated_initializer import DesignatedInitializer
from pydiffbmp.core.preprocessing import Preprocessor


class DiffBMPWrapper:
    """
    Wrapper class for DiffBMP functional operations.
    Parameters (x, y, r, theta, v, c) are passed as function arguments for flexibility.
    
    Example:
        >>> wrapper = DiffBMPWrapper(device='cuda')
        >>> wrapper.load_primitive("heart.svg", size=128)
        >>> x, y, r, theta, v, c = wrapper.initialize_params(n_primitives=100, canvas_size=(512, 512))
        >>> rendered = wrapper.render(x, y, r, theta, v, c)
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize the DiffBMP wrapper.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.primitive = None
        self.renderer = None
        
        # Canvas size
        self.canvas_size = None
    
    def load_primitive(
        self,
        path: Union[str, Path],
        size: int = 128,
        bg_threshold: int = 250,
        radial_transparency: bool = False,
        resampling: str = 'NEAREST'
    ) -> 'DiffBMPWrapper':
        """
        Load a primitive (SVG, PNG, font, etc.) for rendering.
        
        Args:
            path: Path to primitive file
            size: Output size for the primitive bitmap
            bg_threshold: Threshold for background detection
            radial_transparency: Whether to apply radial transparency
            resampling: Resampling method for images
        
        Returns:
            self for method chaining
        """
        # Handle path resolution
        path_str = str(path)
        if not os.path.isabs(path_str):
            # Try different asset folders
            possible_paths = [
                path_str,
                os.path.join("assets/svg", path_str),
                os.path.join("assets/primitives", path_str),
                os.path.join("assets", path_str),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    path_str = p
                    break
        
        # Load primitive using PrimitiveLoader
        try:
            primitive_loader = PrimitiveLoader(
                primitive_paths=path_str,
                output_width=size,
                device=self.device,
                bg_threshold=bg_threshold,
                radial_transparency=radial_transparency,
                resampling=resampling
            )
            print(f"Loaded primitive: {path_str}")
        except Exception as e:
            print(f"PrimitiveLoader failed, trying SVGLoader: {e}")
            # Fallback to SVGLoader for SVG files
            primitive_loader = SVGLoader(
                svg_path=path_str,
                output_width=size,
                device=self.device
            )
        
        # Load bitmap tensor
        primitive_tensor = primitive_loader.load_alpha_bitmap()
        
        # Get primitive colors
        try:
            primitive_colors = primitive_loader.get_primitive_color_maps()
        except:
            # Fallback: create default colors
            if primitive_tensor.ndim == 3:
                num_primitives = primitive_tensor.shape[0]
            else:
                num_primitives = 1
            primitive_colors = torch.zeros(num_primitives, size, size, 3, device=self.device)
        
        # Store primitive data
        self.primitive = {
            'S': primitive_tensor,
            'colors': primitive_colors,
            'loader': primitive_loader
        }
        
        return self
    
    def initialize_params(
        self,
        n_primitives: int,
        canvas_size: Tuple[int, int],
        method: str = 'random',
        target_image: Optional[torch.Tensor] = None,
        radii_min: float = 2.0,
        radii_max: float = None,
        v_init_bias: float = 2.0,
        theta_init: Optional[float] = None,
        x: Optional[Union[float, list]] = None,
        y: Optional[Union[float, list]] = None,
        r: Optional[Union[float, list]] = None,
        theta: Optional[Union[float, list]] = None,
        v: Optional[Union[float, list]] = None,
        c: Optional[Union[list, list]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize rendering parameters.
        
        Args:
            n_primitives: Number of primitives to initialize
            canvas_size: (H, W) tuple for canvas dimensions
            method: Initialization method ('random', 'structure_aware', or 'designated')
            target_image: Target image tensor (H, W, 3), required for 'structure_aware'
            radii_min: Minimum radius for primitives
            radii_max: Maximum radius for primitives
            v_init_bias: Initial bias for visibility logits
            theta_init: Fixed initial rotation for all primitives
            x: Designated x positions (for 'designated' method)
            y: Designated y positions (for 'designated' method)
            r: Designated radii (for 'designated' method)
            theta: Designated rotations (for 'designated' method)
            v: Designated visibility values (for 'designated' method)
            c: Designated colors (for 'designated' method)
            **kwargs: Additional arguments for initializer
        
        Returns:
            Tuple of (x, y, r, theta, v, c) tensors with gradients enabled
        """
        self.canvas_size = canvas_size
        H, W = canvas_size
        
        # Set default radii_max
        if radii_max is None:
            radii_max = 0.1 * min(H, W)
        
        # Prepare initializer config
        init_config = {
            'N': n_primitives,
            'radii_min': radii_min,
            'radii_max': radii_max,
            'v_init_bias': v_init_bias,
            'theta_init': theta_init,
            'debug_mode': False,
            'detail_first': True,
        }
        
        # Add designated parameters if provided (for 'designated' method)
        if x is not None:
            init_config['x'] = x
        if y is not None:
            init_config['y'] = y
        if r is not None:
            init_config['r'] = r
        if theta is not None:
            init_config['theta'] = theta
        if v is not None:
            init_config['v'] = v
        if c is not None:
            init_config['c'] = c
        
        init_config.update(kwargs)
        
        # Create initializer
        if method == 'random':
            initializer = RandomInitializer(init_config)
        elif method == 'structure_aware':
            if target_image is None:
                raise ValueError("target_image is required for 'structure_aware' initialization")
            initializer = StructureAwareInitializer(init_config)
        elif method == 'designated':
            initializer = DesignatedInitializer(init_config)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        # Initialize target image if needed
        if target_image is not None:
            if not isinstance(target_image, torch.Tensor):
                target_image = torch.tensor(target_image, device=self.device)
            elif target_image.device != self.device:
                target_image = target_image.to(self.device)
        else:
            # Create dummy target for random initialization
            target_image = torch.zeros((H, W, 3), device=self.device)
        
        # Run initialization
        x, y, r, v, theta, c = initializer.initialize(
            I_target=target_image,
            target_binary_mask=None,
            I_bg=None,
            renderer=None,
            opt_conf=None
        )
        
        # Enable gradient tracking if requested
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        r = r.requires_grad_(True)
        theta = theta.requires_grad_(True)
        v = v.requires_grad_(True)
        c = c.requires_grad_(True)
        
        return x, y, r, theta, v, c
    
    def render(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        r: torch.Tensor,
        theta: torch.Tensor,
        v: torch.Tensor,
        c: torch.Tensor,
        background: Optional[Union[str, torch.Tensor]] = 'white',
        blur_sigma: float = 1.0,
        return_alpha: bool = False,
        tile_size: int = 32,
        alpha_upper_bound: float = 0.5,
        c_blend: float = 0.0,
        use_fp16: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Render the primitives using provided parameters.
        
        Args:
            x: X-position tensor (N,)
            y: Y-position tensor (N,)
            r: Radius tensor (N,)
            theta: Rotation tensor (N,)
            v: Visibility logits tensor (N,)
            c: Color logits tensor (N, 3)
            background: Background ('white', 'black', 'random') or image tensor (H, W, 3)
            blur_sigma: Gaussian blur sigma
            return_alpha: Whether to return alpha channel
            tile_size: Tile size for rendering
            alpha_upper_bound: Maximum alpha value
            c_blend: Blend factor between primitive color and parameter color
            use_fp16: Whether to use FP16 for memory efficiency
        
        Returns:
            rendered: (H, W, 3) RGB image tensor with gradients
            If return_alpha=True: (H, W, 4) RGBA tensor
        """
        if self.primitive is None:
            raise ValueError("Primitive not loaded. Call load_primitive() first.")
        if self.canvas_size is None:
            raise ValueError("Canvas size not set. Call initialize_params() or set canvas_size manually first.")
        
        H, W = self.canvas_size
        
        # Create or reuse renderer
        if self.renderer is None:
            # Use small tile_size to ensure parallel processing (avoids sequential path bugs)
            # Parallel path is required for proper gradient computation
            # For small images (<=64), use tile_size=12 to get >4 tiles and trigger parallel path
            effective_tile_size = 12 if min(H, W) <= 64 else tile_size
            
            self.renderer = SimpleTileRenderer(
                canvas_size=(H, W),
                S=self.primitive['S'],
                alpha_upper_bound=alpha_upper_bound,
                device=self.device,
                use_fp16=use_fp16,
                tile_size=effective_tile_size,
                sigma=blur_sigma,
                c_blend=c_blend,
                primitive_colors=self.primitive['colors'],
                output_path=None
            )
        
        # Handle background
        if isinstance(background, str):
            if background == 'white':
                I_bg = torch.ones((H, W, 3), device=self.device, dtype=torch.float32)
            elif background == 'black':
                I_bg = torch.zeros((H, W, 3), device=self.device, dtype=torch.float32)
            elif background == 'random':
                I_bg = torch.rand((H, W, 3), device=self.device, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown background type: {background}")
        elif isinstance(background, torch.Tensor):
            I_bg = background
            if I_bg.device != self.device:
                I_bg = I_bg.to(self.device)
        else:
            I_bg = None
        
        # Render with provided parameters
        rendered = self.renderer.render_from_params(
            x=x, y=y, r=r,
            theta=theta, v=v, c=c,
            return_alpha=return_alpha,
            I_bg=I_bg,
            sigma=0.0,
            is_final=False
        )
        
        if return_alpha:
            rendered, alpha = rendered
            return torch.cat([rendered, alpha.unsqueeze(-1)], dim=-1)
        else:
            return rendered

    def render_batch(
        self,
        params_list: list,
        background: Optional[Union[str, torch.Tensor]] = 'white',
        blur_sigma: float = 1.0,
        return_alpha: bool = False,
        tile_size: int = 32,
        alpha_upper_bound: float = 0.5,
        c_blend: float = 0.0,
        use_fp16: bool = False
    ) -> list:
        """
        Render multiple candidates in batch.
        
        This method optimizes batch rendering by reusing GPU resources across
        multiple render calls, reducing memory allocation overhead.
        
        Args:
            params_list: List of (x, y, r, theta, v, c) tuples, each representing
                        a candidate's parameters where tensors have shape (N,)
            background: Background ('white', 'black', 'random') or image tensor (H, W, 3)
            blur_sigma: Gaussian blur sigma
            return_alpha: Whether to return alpha channel
            tile_size: Tile size for rendering
            alpha_upper_bound: Maximum alpha value
            c_blend: Blend factor between primitive color and parameter color
            use_fp16: Whether to use FP16 for memory efficiency
        
        Returns:
            List of rendered images, each (H, W, 3) or (H, W, 4) if return_alpha=True
        """
        if self.primitive is None:
            raise ValueError("Primitive not loaded. Call load_primitive() first.")
        if self.canvas_size is None:
            raise ValueError("Canvas size not set. Call initialize_params() or set canvas_size manually first.")
        
        H, W = self.canvas_size
        
        # Pre-compute background tensor once (optimization: avoid repeated creation)
        if isinstance(background, str):
            if background == 'white':
                I_bg = torch.ones((H, W, 3), device=self.device, dtype=torch.float32)
            elif background == 'black':
                I_bg = torch.zeros((H, W, 3), device=self.device, dtype=torch.float32)
            elif background == 'random':
                I_bg = torch.rand((H, W, 3), device=self.device, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown background type: {background}")
        elif isinstance(background, torch.Tensor):
            I_bg = background
            if I_bg.device != self.device:
                I_bg = I_bg.to(self.device)
        else:
            I_bg = None
        
        # Try to use CUDA batch rendering if renderer is initialized
        if self.renderer is not None and hasattr(self.renderer, 'render_from_params_batch'):
            try:
                return self.renderer.render_from_params_batch(
                    params_list, 
                    return_alpha=return_alpha, 
                    I_bg=I_bg
                )
            except Exception as e:
                # Fallback to sequential on error
                print(f"CUDA batch rendering failed, falling back to sequential: {e}")
        
        # Fallback: sequential rendering using self.render()
        results = []
        for params in params_list:
            x, y, r, theta, v, c = params
            
            # Use self.render() to ensure renderer is initialized
            rendered = self.render(
                x, y, r, theta, v, c,
                background=I_bg if I_bg is not None else background,
                blur_sigma=blur_sigma,
                return_alpha=return_alpha,
                tile_size=tile_size,
                alpha_upper_bound=alpha_upper_bound,
                c_blend=c_blend,
                use_fp16=use_fp16
            )
            results.append(rendered)
        
        return results


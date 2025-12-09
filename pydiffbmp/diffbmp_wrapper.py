"""
PyDiffBMP Functional API with Wrapper Class

Provides a class-based interface for differentiable rendering with bitmap primitives.
Parameters are stored as class members to simplify function calls.
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
from pydiffbmp.core.preprocessing import Preprocessor


class DiffBMPWrapper:
    """
    Wrapper class for DiffBMP functional operations.
    Stores parameters as member variables to simplify API usage.
    
    Example:
        >>> wrapper = DiffBMPWrapper(device='cuda')
        >>> wrapper.load_primitive("heart.svg", size=128)
        >>> wrapper.initialize_params(n_primitives=100, canvas_size=(512, 512))
        >>> rendered = wrapper.render()
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
        
        # Rendering parameters (will be initialized)
        self.x = None
        self.y = None
        self.r = None
        self.theta = None
        self.v = None
        self.c = None
        
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
        **kwargs
    ) -> 'DiffBMPWrapper':
        """
        Initialize rendering parameters.
        
        Args:
            n_primitives: Number of primitives to initialize
            canvas_size: (H, W) tuple for canvas dimensions
            method: Initialization method ('random' or 'structure_aware')
            target_image: Target image tensor (H, W, 3), required for 'structure_aware'
            radii_min: Minimum radius for primitives
            radii_max: Maximum radius for primitives
            v_init_bias: Initial bias for visibility logits
            theta_init: Fixed initial rotation for all primitives
            **kwargs: Additional arguments for initializer
        
        Returns:
            self for method chaining
        """
        self.canvas_size = canvas_size
        H, W = canvas_size
        
        # Set default radii_max
        if radii_max is None:
            radii_max = 0.5 * min(H, W)
        
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
        init_config.update(kwargs)
        
        # Create initializer
        if method == 'random':
            initializer = RandomInitializer(init_config)
        elif method == 'structure_aware':
            if target_image is None:
                raise ValueError("target_image is required for 'structure_aware' initialization")
            initializer = StructureAwareInitializer(init_config)
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
        
        # Store as member variables with gradient tracking
        self.x = x.requires_grad_(True)
        self.y = y.requires_grad_(True)
        self.r = r.requires_grad_(True)
        self.theta = theta.requires_grad_(True)
        self.v = v.requires_grad_(True)
        self.c = c.requires_grad_(True)
        
        return self
    
    def render(
        self,
        background: Optional[Union[str, torch.Tensor]] = 'white',
        blur_sigma: float = 0.0,
        return_alpha: bool = False,
        tile_size: int = 32,
        alpha_upper_bound: float = 0.5,
        c_blend: float = 0.0,
        use_fp16: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Render the primitives using stored parameters.
        
        Args:
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
        if self.x is None:
            raise ValueError("Parameters not initialized. Call initialize_params() first.")
        if self.canvas_size is None:
            raise ValueError("Canvas size not set. Call initialize_params() first.")
        
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
        
        # Render with stored parameters
        rendered = self.renderer.render_from_params(
            x=self.x, y=self.y, r=self.r,
            theta=self.theta, v=self.v, c=self.c,
            return_alpha=return_alpha,
            I_bg=I_bg,
            sigma=1.0,
            is_final=False
        )
        
        if return_alpha:
            rendered, alpha = rendered
            return torch.cat([rendered, alpha.unsqueeze(-1)], dim=-1)
        else:
            return rendered
    
    def update_params(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None
    ) -> 'DiffBMPWrapper':
        """
        Update specific parameters while keeping others unchanged.
        
        Args:
            x, y: Position parameters
            r: Scale parameters
            theta: Rotation parameters
            v: Visibility logits
            c: Color logits
        
        Returns:
            self for method chaining
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if r is not None:
            self.r = r
        if theta is not None:
            self.theta = theta
        if v is not None:
            self.v = v
        if c is not None:
            self.c = c
        
        return self
    
    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all parameters as a tuple.
        
        Returns:
            Tuple of (x, y, r, theta, v, c) tensors
        """
        return self.x, self.y, self.r, self.theta, self.v, self.c
    
    def clone_params(self, requires_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Clone all parameters for optimization.
        
        Args:
            requires_grad: Whether cloned parameters should require gradients
        
        Returns:
            Tuple of cloned (x, y, r, theta, v, c) tensors
        """
        x_clone = self.x.clone().detach()
        y_clone = self.y.clone().detach()
        r_clone = self.r.clone().detach()
        theta_clone = self.theta.clone().detach()
        v_clone = self.v.clone().detach()
        c_clone = self.c.clone().detach()
        
        if requires_grad:
            x_clone.requires_grad_(True)
            y_clone.requires_grad_(True)
            r_clone.requires_grad_(True)
            theta_clone.requires_grad_(True)
            v_clone.requires_grad_(True)
            c_clone.requires_grad_(True)
        
        return x_clone, y_clone, r_clone, theta_clone, v_clone, c_clone

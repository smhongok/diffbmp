"""
PyTorch-style functional API for pydiffbmp

This module provides a simple, functional interface for differentiable rendering
with bitmap primitives. Designed to be similar to PyTorch's functional API.

Example:
    >>> import torch
    >>> import pydiffbmp
    >>> 
    >>> # Load primitive and target
    >>> primitive = pydiffbmp.load_primitive("heart.svg", size=128)
    >>> target = pydiffbmp.preprocess_image("target.png", target_size=512)
    >>> 
    >>> # Initialize parameters (requires_grad=True)
    >>> x, y, r, theta, v, c = pydiffbmp.initialize_params(
    ...     n_primitives=100,
    ...     canvas_size=(512, 512),
    ...     method='random'
    ... )
    >>> 
    >>> # Differentiable render
    >>> rendered = pydiffbmp.render(primitive, x, y, r, theta, v, c)
    >>> 
    >>> # User-defined loss and optimization
    >>> loss = torch.nn.functional.mse_loss(rendered, target)
    >>> loss.backward()
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

# Import internal components
from pydiffbmp.util.primitive_loader import PrimitiveLoader
from pydiffbmp.util.svg_loader import SVGLoader
from pydiffbmp.core.renderer.simple_tile_renderer import SimpleTileRenderer
from pydiffbmp.core.initializer.random_initializater import RandomInitializer
from pydiffbmp.core.initializer.svgsplat_initializater import StructureAwareInitializer
from pydiffbmp.core.preprocessing import Preprocessor


class _PrimitiveWrapper:
    """Internal wrapper to hold primitive data and renderer"""
    def __init__(self, primitive_tensor, primitive_loader, primitive_colors, device):
        self.S = primitive_tensor  # (num_primitives, H, W) or (H, W)
        self.loader = primitive_loader
        self.colors = primitive_colors  # (num_primitives, H, W, 3)
        self.device = device
        self.renderer = None  # Will be created lazily when needed


def load_primitive(
    path: Union[str, Path],
    size: int = 128,
    device: str = 'cuda',
    bg_threshold: int = 250,
    radial_transparency: bool = False,
    resampling: str = 'NEAREST'
) -> _PrimitiveWrapper:
    """
    Load a primitive (SVG, PNG, font, etc.) for rendering.
    
    Args:
        path: Path to primitive file (SVG, PNG, JPG, TTF, OTF)
              Can be absolute or relative to assets folder
        size: Output size for the primitive bitmap (default: 128)
        device: Device to use ('cuda' or 'cpu')
        bg_threshold: Threshold for background detection (default: 0.5)
        radial_transparency: Whether to apply radial transparency (default: False)
        resampling: Resampling method for images (default: 3, LANCZOS)
    
    Returns:
        Primitive wrapper object ready for rendering
        
    Example:
        >>> primitive = pydiffbmp.load_primitive("heart.svg", size=128)
        >>> primitive = pydiffbmp.load_primitive("logo.png", size=256)
    """
    device = torch.device(device)
    
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
            device=device,
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
            device=device
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
        primitive_colors = torch.zeros(num_primitives, size, size, 3, device=device)
    
    return _PrimitiveWrapper(primitive_tensor, primitive_loader, primitive_colors, device)


def initialize_params(
    n_primitives: int,
    canvas_size: Tuple[int, int],
    method: str = 'random',
    target_image: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    radii_min: float = 2.0,
    radii_max: float = 10.0,
    v_init_bias: float = 2.0,
    theta_init: Optional[float] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize rendering parameters with requires_grad=True.
    
    Args:
        n_primitives: Number of primitives to initialize
        canvas_size: (H, W) tuple for canvas dimensions
        method: Initialization method ('random' or 'structure_aware')
        target_image: Target image tensor (H, W, 3), required for 'structure_aware'
        device: Device to use ('cuda' or 'cpu')
        radii_min: Minimum radius for primitives (default: 2.0)
        radii_max: Maximum radius for primitives (default: 0.5 * min(H, W))
        v_init_bias: Initial bias for visibility logits (default: 2.0)
        theta_init: Fixed initial rotation for all primitives (default: None, random)
        **kwargs: Additional arguments for initializer
    
    Returns:
        Tuple of (x, y, r, theta, v, c) tensors, all with requires_grad=True
        - x, y: (N,) position parameters
        - r: (N,) scale parameters
        - theta: (N,) rotation parameters
        - v: (N,) visibility logits (use sigmoid to get alpha)
        - c: (N, 3) color logits (use sigmoid to get RGB)
        
    Example:
        >>> x, y, r, theta, v, c = pydiffbmp.initialize_params(
        ...     n_primitives=100,
        ...     canvas_size=(512, 512),
        ...     method='random'
        ... )
    """
    device = torch.device(device)
    H, W = canvas_size
    
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
        raise ValueError(f"Unknown initialization method: {method}. Use 'random' or 'structure_aware'")
    
    # Initialize target image if needed
    if target_image is not None:
        if not isinstance(target_image, torch.Tensor):
            target_image = torch.tensor(target_image, device=device)
        elif target_image.device != device:
            target_image = target_image.to(device)
    else:
        # Create dummy target for random initialization
        target_image = torch.zeros((H, W, 3), device=device)
    
    # Run initialization
    x, y, r, v, theta, c = initializer.initialize(
        I_target=target_image,
        target_binary_mask=None,
        I_bg=None,
        renderer=None,
        opt_conf=None
    )
    
    # Ensure all parameters have requires_grad=True
    x.requires_grad_(True)
    y.requires_grad_(True)
    r.requires_grad_(True)
    theta.requires_grad_(True)
    v.requires_grad_(True)
    c.requires_grad_(True)
    
    return x, y, r, theta, v, c


def render(
    primitive: _PrimitiveWrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    canvas_size: Optional[Tuple[int, int]] = None,
    background: str = 'white',
    blur_sigma: float = 1.0,
    return_alpha: bool = False,
    tile_size: int = 32,
    alpha_upper_bound: float = 0.5,
    c_blend: float = 0.0,
    use_fp16: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Differentiable rendering function.
    
    This is the core function that renders primitives with full gradient tracking.
    
    Args:
        primitive: Loaded primitive from load_primitive()
        x, y: (N,) position tensors
        r: (N,) scale tensor
        theta: (N,) rotation tensor
        v: (N,) visibility logits (sigmoid applied internally)
        c: (N, 3) color logits (sigmoid applied internally)
        canvas_size: (H, W) tuple. If None, inferred from first render
        background: Background color ('white', 'black', or 'random')
        blur_sigma: Gaussian blur sigma (default: 0.0, no blur)
        return_alpha: Whether to return alpha channel (default: False)
        tile_size: Tile size for rendering (default: 32)
        alpha_upper_bound: Maximum alpha value (default: 0.5)
        c_blend: Blend factor between primitive color and parameter color (default: 0.0)
        use_fp16: Whether to use FP16 for memory efficiency (default: False)
    
    Returns:
        rendered: (H, W, 3) RGB image tensor with gradients
        If return_alpha=True: (rendered, alpha) tuple
        
    Example:
        >>> rendered = pydiffbmp.render(
        ...     primitive, x, y, r, theta, v, c,
        ...     canvas_size=(512, 512)
        ... )
        >>> loss = F.mse_loss(rendered, target)
        >>> loss.backward()  # Gradients flow back to x, y, r, theta, v, c
    """
    # Infer canvas size if not provided
    if canvas_size is None:
        # Try to infer from parameters
        x_max = x.max().item() if x.numel() > 0 else 512
        y_max = y.max().item() if y.numel() > 0 else 512
        canvas_size = (int(y_max) + 100, int(x_max) + 100)
        print(f"Inferred canvas_size: {canvas_size}")
    
    H, W = canvas_size
    
    # Create or reuse renderer
    if primitive.renderer is None:
        # Create renderer with primitive
        primitive.renderer = SimpleTileRenderer(
            canvas_size=(H, W),
            S=primitive.S,
            alpha_upper_bound=alpha_upper_bound,
            device=primitive.device,
            use_fp16=use_fp16,
            tile_size=tile_size,
            sigma=blur_sigma,
            c_blend=c_blend,
            primitive_colors=primitive.colors,
            output_path=None
        )
    
    renderer = primitive.renderer
    
    # Create background
    if background == 'white':
        I_bg = torch.ones((H, W, 3), device=primitive.device, dtype=torch.float32)
    elif background == 'black':
        I_bg = torch.zeros((H, W, 3), device=primitive.device, dtype=torch.float32)
    elif background == 'random':
        I_bg = torch.rand((H, W, 3), device=primitive.device, dtype=torch.float32)
    else:
        I_bg = None
    
    # Render with gradients
    rendered = renderer.render_from_params(
        x=x, y=y, r=r, theta=theta, v=v, c=c,
        return_alpha=return_alpha,
        I_bg=I_bg,
        sigma=0.0,  # blur already applied in renderer initialization
        is_final=False
    )
    
    if return_alpha:
        rendered, alpha = rendered
        return torch.cat([rendered, alpha.unsqueeze(-1)], dim=-1)  # render_from_params returns (image, alpha) if return_alpha=True
    else:
        return rendered


def preprocess_image(
    image_path: Union[str, Path],
    target_size: int = 512,
    trim: bool = False,
    transform_mode: str = 'none',
    has_background: bool = True
) -> torch.Tensor:
    """
    Load and preprocess an image as target.
    
    Args:
        image_path: Path to image file
        target_size: Target width (height adjusted to maintain aspect ratio)
        trim: Whether to trim whitespace (default: False)
        transform_mode: Transformation mode ('none', 'flip', etc.)
        has_background: Whether image has background (default: True)
    
    Returns:
        image_tensor: (H, W, 3) tensor in range [0, 1]
        
    Example:
        >>> target = pydiffbmp.preprocess_image("target.png", target_size=512)
    """
    preprocessor = Preprocessor(
        final_width=target_size,
        trim=trim,
        FM_halftone=False,
        transform_mode=transform_mode
    )
    
    config = {
        'img_path': str(image_path),
        'final_width': target_size
    }
    
    if has_background:
        image_np = preprocessor.load_image_8bit_color(config).astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
    else:
        image_np, mask_np = preprocessor.load_image_8bit_color_opacity(config)
        image_np = image_np.astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
    
    return image_tensor


# Export all public functions
__all__ = [
    'load_primitive',
    'initialize_params',
    'render',
    'preprocess_image',
]

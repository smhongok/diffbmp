import torch
from torch.autograd import Function
import os
import sys
import ctypes
                    
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEBUG_MODE = False

# Import regular FP32 CUDA extension
try:
    from .cuda_tile_rasterizer import _C
    CUDA_AVAILABLE = True
except ImportError:
    # Try alternative import path
    try:
        import cuda_tile_rasterizer._C as _C
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False

# # Import FP16 CUDA extension
try:    
    from .cuda_tile_rasterizer_fp16 import _C as _C_fp16
    CUDA_FP16_AVAILABLE = True
except ImportError:
    # Try alternative import path
    try:
        import cuda_tile_rasterizer_fp16._C as _C_fp16
        CUDA_FP16_AVAILABLE = True
    except ImportError:
        CUDA_FP16_AVAILABLE = False

class TileRasterizer:
    """
    Class-based tile rasterizer that manages global memory between forward and backward passes.
    This ensures accurate gradient computation for transmit-over compositing.
    """
    def __init__(self, image_height, image_width, tile_size=16, sigma=0.0, alpha_upper_bound=1.0, max_prims_per_pixel=500, num_primitives=None):
        self.image_height = image_height
        self.image_width = image_width
        self.tile_size = tile_size
        self.sigma = sigma
        self.alpha_upper_bound = alpha_upper_bound
        self.max_prims_per_pixel = max_prims_per_pixel
        self.num_primitives = num_primitives
        
        # Initialize global rasterizer in C++
        if CUDA_AVAILABLE:
            _C.init_tile_rasterizer(image_height, image_width, tile_size, sigma, alpha_upper_bound, max_prims_per_pixel, num_primitives)
    
    def __call__(self, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_conf, tile_primitive_mapping=None):
        """
        Forward pass using the class-based rasterizer with dynamic tile-primitive mapping
        """
        
        return TileRasterizerFunction.apply(
            means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_conf, tile_primitive_mapping,
            self.image_height, self.image_width, self.tile_size, self.sigma, True  # use_class=True
        )

class TileRasterizerFunction(Function):
    @staticmethod
    def forward(ctx, means2D, radii, rotations, opacities, colors, 
                primitive_templates, global_bmp_sel, lr_conf, tile_primitive_mapping, image_height, image_width, tile_size, sigma, use_class=False):
        
        # Call CUDA forward
        if use_class and CUDA_AVAILABLE:
            if DEBUG_MODE:
                print("Using class-based approach")
            # Convert global_bmp_sel to int32 to match CUDA kernel expectation
            global_bmp_sel_int32 = global_bmp_sel.to(dtype=torch.int32)
            # Save for backward (save the int32 version)
            ctx.save_for_backward(means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, lr_conf)
            ctx.image_height = image_height
            ctx.image_width = image_width
            ctx.tile_size = tile_size
            ctx.sigma = sigma
            ctx.use_class = use_class
            ctx.tile_primitive_mapping = tile_primitive_mapping
            # Use class-based approach with global memory management
            out_color, out_alpha = _C.rasterize_tiles_class(
                means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, tile_primitive_mapping)
        else:
            # Save for backward (original approach)
            global_bmp_sel_int32 = global_bmp_sel.to(dtype=torch.int32)
            ctx.save_for_backward(means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, lr_conf)
            ctx.image_height = image_height
            ctx.image_width = image_width
            ctx.tile_size = tile_size
            ctx.sigma = sigma
            ctx.use_class = use_class
            ctx.tile_primitive_mapping = tile_primitive_mapping
            # Use original approach
            out_color, out_alpha = _C.rasterize_tiles(
                means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, tile_primitive_mapping, 
                image_height, image_width, tile_size, sigma)
        
        return out_color, out_alpha
    
    @staticmethod
    def backward(ctx, grad_out_color, grad_out_alpha):
        means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_conf = ctx.saved_tensors
        
        # Create learning rate tensor from config
        lr_config = torch.tensor([
            lr_conf[0].item(),
            lr_conf[1].item(),
            lr_conf[2].item(),
            lr_conf[3].item(),
            lr_conf[4].item(),
            lr_conf[5].item(),
            lr_conf[6].item()
        ], dtype=torch.float32, device=means2D.device)
        
        # Call CUDA backward
        if ctx.use_class and CUDA_AVAILABLE:
            # Use class-based approach with shared global memory (global_bmp_sel is already int32)
            grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = _C.rasterize_tiles_backward_class(
                grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config)
        else:
            # Use original approach
            grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = _C.rasterize_tiles_backward(
                grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config,
                ctx.image_height, ctx.image_width, ctx.tile_size, ctx.sigma)
                
        return grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors, None, None, None, None, None, None, None, None, None

class CudaTileRasterizeFunction(Function):
    @staticmethod
    def forward(ctx, means2D, radii, rotations, opacities, colors, 
                primitive_templates, global_bmp_sel, lr_conf, tile_primitive_mapping, image_height, image_width, tile_size, sigma):
        
        # Save for backward
        global_bmp_sel_int32 = global_bmp_sel.to(dtype=torch.int32)
        ctx.save_for_backward(means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, lr_conf)
        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.tile_size = tile_size
        ctx.sigma = sigma
        ctx.tile_primitive_mapping = tile_primitive_mapping
        # Call CUDA forward using class-based function that supports global_bmp_sel
        out_color, out_alpha = _C.rasterize_tiles_class(
            means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, tile_primitive_mapping)
        
        return out_color, out_alpha
    
    @staticmethod
    def backward(ctx, grad_out_color, grad_out_alpha):
        means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_conf = ctx.saved_tensors
        
        # Create learning rate tensor from config
        lr_config = torch.tensor([
            lr_conf[0].item(),
            lr_conf[1].item(),
            lr_conf[2].item(),
            lr_conf[3].item(),
            lr_conf[4].item(),
            lr_conf[5].item(),
            lr_conf[6].item()
        ], dtype=torch.float32, device=means2D.device)
        
        # Call CUDA backward
        grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = \
            _C.rasterize_tiles_backward(
                grad_out_color, grad_out_alpha,
                means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config,
                ctx.image_height, ctx.image_width, ctx.tile_size, ctx.sigma)
        
        return grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors, None, None, None, None, None, None, None, None

# class CudaTileRasterizeFunctionFP16(Function):
#     """
#     PyTorch autograd Function for FP16 CUDA tile rasterization.
#     """
    
#     @staticmethod
#     def forward(ctx, means2D, radii, rotations, opacities, colors, 
#                 primitive_templates, image_height, image_width, tile_size, sigma):
        
#         # if not CUDA_FP16_AVAILABLE:
#         #     raise RuntimeError("FP16 CUDA extension not available")
        
#         # Save for backward
#         ctx.save_for_backward(means2D, radii, rotations, opacities, colors, primitive_templates)
#         ctx.image_height = image_height
#         ctx.image_width = image_width
#         ctx.tile_size = tile_size
#         ctx.sigma = sigma
        
#         # Call FP16 CUDA forward
#         out_color, out_alpha = _C_fp16.rasterize_tiles_fp16(
#             means2D, radii, rotations, opacities, colors, primitive_templates,
#             image_height, image_width, tile_size, sigma)
        
#         return out_color, out_alpha
    
#     @staticmethod
#     def backward(ctx, grad_out_color, grad_out_alpha):
#         # if not CUDA_FP16_AVAILABLE:
#         #     raise RuntimeError("FP16 CUDA extension not available")
        
#         means2D, radii, rotations, opacities, colors, primitive_templates = ctx.saved_tensors
        
#         # Call FP16 CUDA backward
#         grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = \
#             _C_fp16.rasterize_tiles_backward_fp16(
#                 grad_out_color, grad_out_alpha,
#                 means2D, radii, rotations, opacities, colors, primitive_templates,
#                 ctx.image_height, ctx.image_width, ctx.tile_size, ctx.sigma)
        
#         return grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors, None, None, None, None, None

def rasterize_tiles(means2D, radii, rotations, opacities, colors, 
                   primitive_templates, global_bmp_sel, lr_conf, tile_primitive_mapping, image_height, image_width, tile_size, sigma=0.0):
    """
    CUDA-accelerated tile-based rasterization of 2D primitives.
    
    Args:
        means2D: (N, 2) tensor of primitive centers
        radii: (N,) tensor of primitive radii
        rotations: (N,) tensor of primitive rotations
        opacities: (N,) tensor of primitive opacities (logits)
        colors: (N, 3) tensor of primitive colors (logits)
        primitive_templates: (num_templates, H, W) tensor of primitive templates
        global_bmp_sel: (N,) tensor of template selection indices
        image_height: height of output image
        image_width: width of output image
        tile_size: size of tiles for processing
        sigma: smoothing parameter
        lr_conf: dictionary containing learning rate configuration
        
    Returns:
        tuple: (out_color, out_alpha) tensors
    """
        
    return CudaTileRasterizeFunction.apply(
        means2D, radii, rotations, opacities, colors, 
        primitive_templates, global_bmp_sel, lr_conf, tile_primitive_mapping, image_height, image_width, tile_size, sigma
    )

# def rasterize_tiles_fp16(means2D, radii, rotations, opacities, colors, 
#                         primitive_templates, image_height, image_width, tile_size, sigma=0.0):
#     """
#     FP16 CUDA-accelerated tile-based rasterization of 2D primitives.
    
#     Args:
#         means2D: (N, 2) tensor of primitive centers
#         radii: (N,) tensor of primitive radii
#         rotations: (N,) tensor of primitive rotations
#         opacities: (N,) tensor of primitive opacities (logits)
#         colors: (N, 3) tensor of primitive colors (logits)
#         primitive_templates: (T, H, W) tensor of primitive templates
#         image_height: int, output image height
#         image_width: int, output image width
#         tile_size: int, size of tiles for processing
#         sigma: float, Gaussian blur sigma
        
#     Returns:
#         out_color: (H, W, 3) tensor of rendered image
#         out_alpha: (H, W) tensor of alpha channel
#     """
#     return CudaTileRasterizeFunctionFP16.apply(
#         means2D, radii, rotations, opacities, colors, primitive_templates,
#         image_height, image_width, tile_size, sigma)

def compute_per_pixel_gradients(means2D, radii, rotations, opacities, colors, 
                               primitive_templates, global_bmp_sel, target_image, pixels_per_tile):
    """
    Compute per-pixel gradient magnitudes for primitives.
    
    Args:
        means2D: (N, 2) tensor of primitive centers
        radii: (N,) tensor of primitive radii
        rotations: (N,) tensor of primitive rotations
        opacities: (N,) tensor of primitive opacities (logits)
        colors: (N, 3) tensor of primitive colors
        primitive_templates: (num_templates, H, W) tensor of primitive templates
        global_bmp_sel: (N,) tensor of template selection indices
        target_image: (H, W, 3) target image for gradient computation
        pixels_per_tile: number of pixels to sample per tile
        
    Returns:
        gradient_magnitudes: (H, W) tensor of per-pixel gradient magnitudes
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    return _C.compute_per_pixel_gradients(
        means2D, radii, rotations, opacities, colors,
        primitive_templates, global_bmp_sel, target_image, pixels_per_tile
    )

# __all__ = ['rasterize_tiles', 'rasterize_tiles_fp16', 'CUDA_AVAILABLE', 'CUDA_FP16_AVAILABLE']
__all__ = ['rasterize_tiles', 'TileRasterizer', 'TileRasterizerFunction', 'CudaTileRasterizeFunction', 'compute_per_pixel_gradients', 'CUDA_AVAILABLE']
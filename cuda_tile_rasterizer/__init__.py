import torch
from torch.autograd import Function
try:
    import cuda_tile_rasterizer._C as _C
except ImportError:
    # Try alternative import path
    try:
        from . import _C
    except ImportError:
        # Try direct import from current directory
        import _C

class CudaTileRasterizeFunction(Function):
    @staticmethod
    def forward(ctx, means2D, radii, rotations, opacities, colors, 
                primitive_templates, image_height, image_width, tile_size, sigma):
        
        # Save for backward
        ctx.save_for_backward(means2D, radii, rotations, opacities, colors, primitive_templates)
        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.tile_size = tile_size
        ctx.sigma = sigma
        
        # Call CUDA forward
        out_color, out_alpha = _C.rasterize_tiles(
            means2D, radii, rotations, opacities, colors, primitive_templates,
            image_height, image_width, tile_size, sigma)
        
        return out_color, out_alpha
    
    @staticmethod
    def backward(ctx, grad_out_color, grad_out_alpha):
        means2D, radii, rotations, opacities, colors, primitive_templates = ctx.saved_tensors
        
        # Call CUDA backward
        grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = \
            _C.rasterize_tiles_backward(
                grad_out_color, grad_out_alpha,
                means2D, radii, rotations, opacities, colors, primitive_templates,
                ctx.image_height, ctx.image_width, ctx.tile_size, ctx.sigma)
        
        return grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors, None, None, None, None, None

def rasterize_tiles(means2D, radii, rotations, opacities, colors, 
                   primitive_templates, image_height, image_width, tile_size, sigma=0.0):
    """
    CUDA-accelerated tile-based rasterization of 2D primitives.
    
    Args:
        means2D: (N, 2) tensor of primitive centers
        radii: (N,) tensor of primitive radii
        rotations: (N,) tensor of primitive rotations
        opacities: (N,) tensor of primitive opacities (logits)
        colors: (N, 3) tensor of primitive colors (logits)
        primitive_templates: (T, H, W) tensor of primitive templates
        image_height: int, output image height
        image_width: int, output image width
        tile_size: int, size of tiles for processing
        sigma: float, Gaussian blur sigma
        
    Returns:
        out_color: (H, W, 3) tensor of rendered image
        out_alpha: (H, W) tensor of alpha channel
    """
    return CudaTileRasterizeFunction.apply(
        means2D, radii, rotations, opacities, colors, primitive_templates,
        image_height, image_width, tile_size, sigma)

__all__ = ['rasterize_tiles']
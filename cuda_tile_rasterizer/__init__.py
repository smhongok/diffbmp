import torch
from torch.autograd import Function
import os
import sys
from torch.amp import custom_bwd, custom_fwd
import ctypes
                    
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEBUG_MODE = False

# Import regular FP32 CUDA extension
try:
    from .cuda_tile_rasterizer import _C
    CUDA_AVAILABLE = True
    print("FP32 CUDA extension loaded successfully!")
except ImportError:
    # Try alternative import path
    print("Warning: FP32 CUDA extension not available, falling back to FP32")
    try:
        import cuda_tile_rasterizer._C as _C
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False

# # Import FP16 CUDA extension
try:    
    from .cuda_tile_rasterizer_fp16 import _C_fp16
    CUDA_FP16_AVAILABLE = True
    print("FP16 CUDA extension loaded successfully!")
except ImportError:
    # Try alternative import path
    print("Warning: FP16 CUDA extension not available, falling back to FP32")
    try:
        import cuda_tile_rasterizer_fp16._C_fp16 as _C_fp16
        CUDA_FP16_AVAILABLE = True
        print("FP16 CUDA extension loaded successfully!")
    except ImportError:
        CUDA_FP16_AVAILABLE = False
        print("Warning: FP16 CUDA extension not available, falling back to FP32")

class TileRasterizer:
    """
    Class-based tile rasterizer that manages global memory between forward and backward passes.
    This ensures accurate gradient computation for transmit-over compositing.
    """
    def __init__(self, image_height, image_width, tile_size=16, sigma=0.0, alpha_upper_bound=1.0, max_prims_per_pixel=500, num_primitives=None, use_fp16=False):
        self.image_height = image_height
        self.image_width = image_width
        self.tile_size = tile_size
        self.sigma = sigma
        self.alpha_upper_bound = alpha_upper_bound
        self.max_prims_per_pixel = max_prims_per_pixel
        self.num_primitives = num_primitives
        self.use_fp16 = use_fp16

        if self.use_fp16:
            self.apply_fn = TileRasterizerFunctionFP16.apply
            if CUDA_FP16_AVAILABLE:
                _C_fp16.init_tile_rasterizer_fp16(image_height, image_width, tile_size, sigma, alpha_upper_bound, max_prims_per_pixel, num_primitives)
        else:
            self.apply_fn = TileRasterizerFunction.apply
            if CUDA_AVAILABLE:
                _C.init_tile_rasterizer(image_height, image_width, tile_size, sigma, alpha_upper_bound, max_prims_per_pixel, num_primitives)
    
    def __call__(self, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_conf, tile_primitive_mapping=None):
        """
        Forward pass using the class-based rasterizer with dynamic tile-primitive mapping
        """
        
        return self.apply_fn(
            means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_conf, tile_primitive_mapping,
            self.image_height, self.image_width, self.tile_size, self.sigma, True
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
            out_color, out_alpha = _C.rasterize_tiles_class(means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, tile_primitive_mapping)
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
        
        # Debug: Check if alpha gradient is None
        if DEBUG_MODE:
            print(f"Alpha gradient is None: {grad_out_alpha is None}")
            if grad_out_alpha is not None:
                print(f"Alpha gradient shape: {grad_out_alpha.shape}, mean: {grad_out_alpha.mean().item()}")
            
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
        # Use class-based approach with shared global memory (global_bmp_sel is already int32)
        if ctx.use_class and CUDA_AVAILABLE:
            grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = _C.rasterize_tiles_backward_class(
                grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config)
        else:
            # Use original approach
            grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = _C.rasterize_tiles_backward(
                grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config,
                ctx.image_height, ctx.image_width, ctx.tile_size, ctx.sigma)
                
        return grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors, None, None, None, None, None, None, None, None, None

class TileRasterizerFunctionFP16(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
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
            out_color, out_alpha = _C_fp16.rasterize_tiles_class_fp16(means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, tile_primitive_mapping)
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
            out_color, out_alpha = _C_fp16.rasterize_tiles_fp16(
                means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel_int32, tile_primitive_mapping, 
                image_height, image_width, tile_size, sigma)
            
        return out_color, out_alpha
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_out_color, grad_out_alpha):
        means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_conf = ctx.saved_tensors
        
        # Debug: Check if alpha gradient is None
        if DEBUG_MODE:
            print(f"Alpha gradient is None: {grad_out_alpha is None}")
            if grad_out_alpha is not None:
                print(f"Alpha gradient shape: {grad_out_alpha.shape}, mean: {grad_out_alpha.mean().item()}")
            
        # Create learning rate tensor from config
        lr_config = torch.tensor([
            lr_conf[0].item(),
            lr_conf[1].item(),
            lr_conf[2].item(),
            lr_conf[3].item(),
            lr_conf[4].item(),
            lr_conf[5].item(),
            lr_conf[6].item()
        ], dtype=torch.float16, device=means2D.device)
        
        # Call CUDA backward
        # Use class-based approach with shared global memory (global_bmp_sel is already int32)
        if ctx.use_class and CUDA_AVAILABLE:
            grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = _C_fp16.rasterize_tiles_backward_class_fp16(
                grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config)
        else:
            # Use original approach
            grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors = _C_fp16.rasterize_tiles_backward_fp16(
                grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config,
                ctx.image_height, ctx.image_width, ctx.tile_size, ctx.sigma)
                
        return grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors, None, None, None, None, None, None, None, None, None

#__all__ = ['rasterize_tiles', 'TileRasterizer', 'TileRasterizerFunction', 'CudaTileRasterizeFunction', 'CUDA_AVAILABLE']
__all__ = ['TileRasterizer', 'TileRasterizerFunction', 'TileRasterizerFunctionFP16', 'CUDA_AVAILABLE']
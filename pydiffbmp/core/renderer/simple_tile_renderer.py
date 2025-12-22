import torch
import torch.nn.functional as F
from typing import Any, Tuple, List, Optional
from pydiffbmp.core.renderer.vector_renderer import VectorRenderer
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from pydiffbmp.util.loss_functions import LossComposer
import pynvml as nvml
from pydiffbmp.util.constants import MAX_PRIMS_PER_PIXEL

DEBUG_MODE = False
DEBUG_MODE_DETAIL = False
DEBUG_MODE_SAVE = False

# Try to import CUDA extension, fallback to PyTorch if not available
try:
    import sys
    import os
    # Add CUDA extension path to sys.path
    cuda_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cuda_tile_rasterizer',  'cuda_tile_rasterizer')
    cuda_fp16_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cuda_tile_rasterizer',  'cuda_tile_rasterizer_fp16')
    print(f"cuda_tile_rasterizer path: {cuda_path}")
    print(f"cuda_tile_rasterizer_fp16 path: {cuda_fp16_path}")
    if cuda_path not in sys.path:
        sys.path.insert(0, cuda_path)
    if cuda_fp16_path not in sys.path:
        sys.path.insert(0, cuda_fp16_path)
    from cuda_tile_rasterizer import TileRasterizer, print_cuda_timing_stats, print_cuda_timing_stats_fp16
    CUDA_AVAILABLE = True
    print("CUDA tile rasterizer loaded successfully!")
            
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"CUDA tile rasterizer not available, using PyTorch fallback: {e}")

# CUDA_AVAILABLE=False
# CUDA_AVAILABLE_FP16=False

def _get_running_procs(handle, kind):
    # kind: "Compute" or "Graphics"
    for name in [f"nvmlDeviceGet{kind}RunningProcesses_v3",
                 f"nvmlDeviceGet{kind}RunningProcesses_v2",
                 f"nvmlDeviceGet{kind}RunningProcesses"]:
        fn = getattr(nvml, name, None)
        if fn:
            try:
                return fn(handle)
            except nvml.NVMLError:
                pass
    return []

def vram_used_by_pid(pid=None):
    """Return per-GPU and total VRAM (bytes) held by the PID as reported by NVML/nvidia-smi."""
    if pid is None:
        pid = os.getpid()

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        # If env not set, assume physical GPU 0
        target = "0"
    else:
        target = vis.split(",")[0].strip()

    nvml.nvmlInit()
    try:
        h = nvml.nvmlDeviceGetHandleByIndex(int(target))
        used = 0
        # NVML exposes compute/graphics processes separately → sum both
        for procs in (_get_running_procs(h, "Compute"), _get_running_procs(h, "Graphics")):
            for p in procs or []:
                if int(p.pid) == int(pid):
                    # usedGpuMemory is in bytes (can be -1 for underflow/UNKNOWN)
                    if getattr(p, "usedGpuMemory", 0) and p.usedGpuMemory > 0:
                        used += p.usedGpuMemory
        return used
    finally:
        nvml.nvmlShutdown()


class SimpleTileRenderer(VectorRenderer):
    """
    Memory-efficient tile-based renderer that dynamically selects primitives per tile.
    Only processes primitives that actually affect each tile.
    """
    
    def __init__(self, canvas_size: Tuple[int, int], S: torch.Tensor, 
                 tile_size: int = 32, max_prims_per_pixel: int = None, **kwargs):
        """
        Initialize the tile renderer.
        
        Args:
            canvas_size: Tuple of (height, width) for the output canvas
            S: Primitive shapes tensor
            tile_size: Size of each tile (default: 32)
            max_prims_per_pixel: Maximum number of primitives per pixel (default: MAX_PRIMS_PER_PIXEL constant)
            **kwargs: Additional arguments passed to VectorRenderer
        """
        print("="*10,"Initializing SimpleTileRenderer...","="*10)
        super().__init__(canvas_size, S, **kwargs)
        self.tile_size = tile_size
        
        # Calculate tile grid dimensions
        self.tiles_h = (self.H + tile_size - 1) // tile_size
        self.tiles_w = (self.W + tile_size - 1) // tile_size
        
        self.cuda_rasterizer = None
        
        # Compute bounding boxes for each primitive in self.S
        self.primitive_bboxes = self._compute_primitive_bboxes()
        
        self.max_prims_per_pixel = max_prims_per_pixel if max_prims_per_pixel is not None else MAX_PRIMS_PER_PIXEL
        
        # PyTorch timing variables
        self.pytorch_forward_time = 0.0
        self.pytorch_backward_time = 0.0
        self.pytorch_forward_count = 0
        self.pytorch_backward_count = 0

    def _clamp_params_inplace(self, x, y, r):
        # Same policy as VectorRenderer: r ∈ [2, min(H,W)//4]
        r_max = int(min(self.H, self.W) // 4)
        with torch.no_grad():
            x.clamp_(0.0, float(self.W - 1))
            y.clamp_(0.0, float(self.H - 1))
            r.clamp_(2.0, float(r_max))
    
    def _compute_primitive_bboxes(self):
        """
        Compute minimal bounding boxes for each primitive in self.S.
        
        Returns:
            List of (min_u, max_u, min_v, max_v) for each primitive in normalized coordinates
        """
        bboxes = []
        
        if self.S.dim() == 2:  # Single primitive (H, W)
            primitives = [self.S]
        elif self.S.dim() == 3:  # Multiple primitives (p, H, W)
            primitives = [self.S[i] for i in range(self.S.shape[0])]
        else:
            raise ValueError(f"Unsupported self.S shape: {self.S.shape}")
        
        for primitive in primitives:
            # Find non-zero regions
            nonzero_mask = primitive > 1e-6  # Small threshold for numerical stability
            
            if not nonzero_mask.any():
                # Empty primitive
                bboxes.append((-1, 1, -1, 1))  # Full range as fallback
                continue
            
            # Get coordinates of non-zero pixels
            H, W = primitive.shape
            v_coords, u_coords = torch.where(nonzero_mask)
            
            # Convert to normalized coordinates [-1, 1]
            u_norm = (u_coords.float() / (W - 1)) * 2 - 1  # [0, W-1] -> [-1, 1]
            v_norm = (v_coords.float() / (H - 1)) * 2 - 1  # [0, H-1] -> [-1, 1]
            
            # Compute bounding box
            min_u = u_norm.min().item()
            max_u = u_norm.max().item()
            min_v = v_norm.min().item()
            max_v = v_norm.max().item()
            
            bboxes.append((min_u, max_u, min_v, max_v))
        
        return bboxes

    def render_from_params(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, 
                           theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                           return_alpha: bool = False, I_bg: torch.Tensor = None, 
                           sigma: float = 0.0, lr_conf: dict = None, is_final: bool = False) -> torch.Tensor:
        """
        Memory-efficient tile-based rendering.
        
        Args:
            x, y: (N,) primitive positions
            r: (N,) primitive scales
            theta: (N,) primitive rotations
            v: (N,) visibility logits  
            c: (N, 3) RGB logits
            return_alpha: Whether to return alpha channel
            I_bg: Background image
            sigma: Gaussian blur std
            
        Returns:
            Rendered image tensor
        """
        
        N = x.shape[0]
        # Pre-compute global primitive template selection (before tile processing)
        
        if self.S_blurred is not None:
            # Ensure self.S_blurred has dimension 3 for consistent processing
            if self.S_blurred.dim() == 2:  # Single template [H, W] -> [1, H, W]
                self.S_blurred = self.S_blurred.unsqueeze(0)  # Add batch dimension
                if DEBUG_MODE:
                    print("    🔧 Converted single template to batch format: [H, W] -> [1, H, W]")

            if self.S_blurred.dim() == 3:  # Multiple primitive templates [p, H, W]
                p = self.S_blurred.size(0)
                global_bmp_sel = torch.arange(N, device=self.device, dtype=torch.long) % p
                global_bmp_sel = global_bmp_sel.flip(0)  # Same as original VectorRenderer
                if DEBUG_MODE:
                    print(f"    📊 Using {p} templates, global_bmp_sel shape: {global_bmp_sel.shape}")
            else:
                global_bmp_sel = None  # Single template case
                print("    ⚠️ Unexpected self.S_blurred dimension, using None for global_bmp_sel")

        # Ensure self.S has dimension 3 for consistent processing
        else:
            if self.S.dim() == 2:  # Single template [H, W] -> [1, H, W]
                self.S = self.S.unsqueeze(0)  # Add batch dimension
                if DEBUG_MODE:
                    print("    🔧 Converted single template to batch format: [H, W] -> [1, H, W]")

            if self.S.dim() == 3:  # Multiple primitive templates [p, H, W]
                p = self.S.size(0)
                global_bmp_sel = torch.arange(N, device=self.device, dtype=torch.long) % p
                global_bmp_sel = global_bmp_sel.flip(0)  # Same as original VectorRenderer
                if DEBUG_MODE:
                    print(f"    📊 Using {p} templates, global_bmp_sel shape: {global_bmp_sel.shape}")
            else:
                global_bmp_sel = None  # Single template case
                print("    ⚠️ Unexpected self.S dimension, using None for global_bmp_sel")
        
        # Initialize output canvas with background color
        # This ensures tiles without primitives display I_bg instead of black
        if self.use_fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        if I_bg is not None:
            # Only clone/convert if necessary to avoid unnecessary memory operations
            if I_bg.device != self.device or I_bg.dtype != dtype:
                output = I_bg.to(device=self.device, dtype=dtype).clone()
            else:
                output = I_bg.clone()
        else:
            output = torch.zeros((self.H, self.W, 3), device=self.device, dtype=dtype)
        
        # Choose processing method based on tile count and device
        total_tiles = self.tiles_h * self.tiles_w
        use_parallel = total_tiles > 4 and torch.cuda.is_available()  # Parallel for larger tile counts
        
        self._forward_compute_time_accum = 0.0
        
        if use_parallel:
            output = self._process_tiles_parallel(x, y, r, theta, v, c, sigma, I_bg, global_bmp_sel, output, lr_conf, is_final=is_final, return_alpha=return_alpha)
        else:
            output = self._process_tiles_sequential(x, y, r, theta, v, c, sigma, I_bg, global_bmp_sel, output, lr_conf, is_final=is_final)
        
        # Update PyTorch forward timing statistics (core compute only)
        self.pytorch_forward_time += self._forward_compute_time_accum
        self.pytorch_forward_count += 1
        
        if return_alpha:
            output, alpha = output
            return output, alpha
        else:
            return output
    
    def _process_tiles_sequential(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                 theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                                 sigma: float, I_bg: torch.Tensor,
                                 global_bmp_sel: torch.Tensor, output: torch.Tensor, lr_conf: dict,
                                 is_final: bool = False,
                                 return_alpha: bool = False) -> torch.Tensor:
        """Sequential tile processing (original method)."""
        if return_alpha:
            alpha = torch.zeros((self.H, self.W), device=self.device, dtype=output.dtype)
        
        for tile_y in range(self.tiles_h):
            for tile_x in range(self.tiles_w):
                # Calculate tile boundaries
                y_start = tile_y * self.tile_size
                y_end = min(y_start + self.tile_size, self.H)
                x_start = tile_x * self.tile_size  
                x_end = min(x_start + self.tile_size, self.W)
                
                # Find primitives that affect this tile
                tile_primitive_indices = self._unified_primitive_assignment(
                    x, y, r, theta, x_start, x_end, y_start, y_end
                )
                
                # Skip if no primitives affect this tile
                if len(tile_primitive_indices) == 0:
                    continue
                
                # Render this tile with selected primitives only
                tile_result = self._render_tile(
                    x, y, r, theta, v, c, c_blend, tile_primitive_indices,
                    x_start, x_end, y_start, y_end, sigma, I_bg,
                    global_bmp_sel=global_bmp_sel, is_final=is_final,
                    return_alpha=return_alpha
                )
                
                # Place result in output canvas
                if return_alpha:
                    tile_result, tile_alpha = tile_result
                    output[y_start:y_end, x_start:x_end] = tile_result
                    alpha[y_start:y_end, x_start:x_end] = tile_alpha
                else:
                    output[y_start:y_end, x_start:x_end] = tile_result
        
        if return_alpha:
            return output, alpha
        else:
            return output
    
    def _process_tiles_parallel(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                               theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                               sigma: float, I_bg: torch.Tensor, global_bmp_sel: torch.Tensor, output: torch.Tensor, lr_conf: dict, is_final: bool = False,
                               return_alpha: bool = False) -> torch.Tensor:
        """True vectorized tile processing using PyTorch operations."""
        
        # Pre-compute all tile boundaries
        total_tiles = self.tiles_h * self.tiles_w
        tile_y_coords = torch.arange(self.tiles_h, device=self.device).repeat_interleave(self.tiles_w)
        tile_x_coords = torch.arange(self.tiles_w, device=self.device).repeat(self.tiles_h)
        
        y_starts = tile_y_coords * self.tile_size
        y_ends = torch.clamp(y_starts + self.tile_size, max=self.H)
        x_starts = tile_x_coords * self.tile_size
        x_ends = torch.clamp(x_starts + self.tile_size, max=self.W)
        
        # Vectorized primitive-to-tile assignment
        primitive_tile_masks = self._unified_primitive_assignment(
            x, y, r, theta, x_starts, x_ends, y_starts, y_ends
        )
        
        # Pre-filter primitives per tile for better performance
        tile_primitive_indices = []
        for tile_idx in range(total_tiles):
            tile_mask = primitive_tile_masks[tile_idx]
            if tile_mask.any():
                indices = torch.nonzero(tile_mask, as_tuple=True)[0]
                tile_primitive_indices.append(indices)
            else:
                tile_primitive_indices.append(torch.empty(0, dtype=torch.long, device=self.device))
        
        # Try true parallel CUDA processing for all tiles at once
        if CUDA_AVAILABLE:
            try:
                # Prepare all data for single CUDA call
                result = self._cuda_process_all_tiles(
                    x, y, r, theta, v, c, sigma, I_bg,
                    global_bmp_sel, primitive_tile_masks,
                    x_starts, x_ends, y_starts, y_ends, lr_conf, is_final=is_final,
                    return_alpha=return_alpha
                )

                if result is not None:
                    if return_alpha:
                        result, alpha = result
                        return result, alpha
                    else:
                        return result
                else:
                    print("  ⚠️ CUDA kernel call returned None, falling back...")
            except Exception as e:
                print(f"  ❌ CUDA kernel call failed: {e}")
                # Fallback to sequential processing
                pass
        
        # Fallback: Process tiles with primitives (sequential)
        if return_alpha:
            alpha = torch.zeros((self.H, self.W), device=self.device, dtype=output.dtype)
        
        for tile_idx in range(total_tiles):
            # Get primitives affecting this tile
            tile_mask = primitive_tile_masks[tile_idx]
            if not tile_mask.any():
                continue
                
            primitive_indices = torch.nonzero(tile_mask, as_tuple=True)[0].tolist()
            
            # Render this tile
            tile_result = self._render_tile(
                x, y, r, theta, v, c, primitive_indices,
                x_starts[tile_idx].item(), x_ends[tile_idx].item(),
                y_starts[tile_idx].item(), y_ends[tile_idx].item(),
                sigma, I_bg, global_bmp_sel=global_bmp_sel, is_final=is_final,
                return_alpha=return_alpha
            )
            
            # Place result in output canvas
            y_start, y_end = y_starts[tile_idx].item(), y_ends[tile_idx].item()
            x_start, x_end = x_starts[tile_idx].item(), x_ends[tile_idx].item()
            
            if return_alpha:
                tile_result, tile_alpha = tile_result
                output[y_start:y_end, x_start:x_end] = tile_result
                alpha[y_start:y_end, x_start:x_end] = tile_alpha
            else:
                output[y_start:y_end, x_start:x_end] = tile_result
        
        if return_alpha:
            return output, alpha
        else:
            return output
    
    def _cuda_process_all_tiles(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                               theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                               sigma: float, I_bg: torch.Tensor,
                               global_bmp_sel: torch.Tensor, primitive_tile_masks: torch.Tensor,
                               x_starts: torch.Tensor, x_ends: torch.Tensor,
                               y_starts: torch.Tensor, y_ends: torch.Tensor, lr_conf: dict, is_final: bool = False,
                               return_alpha: bool = False) -> torch.Tensor:

        try:
            # Prepare input tensors for CUDA kernel
            means2D = torch.stack([x, y], dim=1)  # (N, 2)
            radii = r  # (N,)
            rotations = theta  # (N,)
            opacities = v  # (N,)
            colors = c  # (N, 3)
            
            # Use instance variables for c_o and c_blend
            if self.c_o is not None:
                # Use primitive-specific color maps based on global_bmp_sel
                # self.c_o shape: (num_primitives, H, W, 3)
                # global_bmp_sel shape: (N,)
                # We need to select the color map for each primitive
                c_o = self.c_o[global_bmp_sel]  # (N, H, W, 3) - select color maps for each primitive
            else:
                print(f"❌ Debug - self.c_o is None")
                return None

            c_blend = self.c_blend  # Use config value

            if self.S_blurred is None:
                primitive_templates = self.S  # Use original templates for optimization

                # Use existing cuda_rasterizer or create new one if needed
                if self.cuda_rasterizer is None:
                    print(f"    🔧 Creating new TileRasterizer for optimization with {len(radii)} primitives...")
                    self.cuda_rasterizer = TileRasterizer(
                        self.H, self.W, self.tile_size, sigma, 
                        self.alpha_upper_bound, self.max_prims_per_pixel, len(radii),
                        use_fp16=self.use_fp16
                    )

            elif self.S_blurred is not None and not is_final:
                primitive_templates = self.S_blurred  # Use blurred templates for optimization
                
                # Use existing cuda_rasterizer or create new one if needed
                if self.cuda_rasterizer is None:
                    print(f"    🔧 Creating new TileRasterizer for optimization with {len(radii)} primitives...")
                    self.cuda_rasterizer = TileRasterizer(
                        self.H, self.W, self.tile_size, sigma, 
                        self.alpha_upper_bound, self.max_prims_per_pixel, len(radii),
                        use_fp16=self.use_fp16
                    )

            # Get primitive templates based on is_final flag
            else:
                primitive_templates = self.S  # Use original templates for final rendering
                primitive_templates = primitive_templates.unsqueeze(0) if primitive_templates.dim() == 2 else primitive_templates
                if DEBUG_MODE:
                    print(f"    🎯 Final rendering: Using original templates with shape: {primitive_templates.shape}")
                    print(f"    🔍 Debug - self.S.dim(): {self.S.dim()}")
                    print(f"    🔍 Debug - self.S.shape: {self.S.shape}")
                    print(f"    🔍 Debug - self.S.dtype: {self.S.dtype}")
                    print(f"    🔍 Debug - self.S.device: {self.S.device}")


            if DEBUG_MODE:
                print(f"    🎯 Calling CUDA rasterizer with {len(radii)} primitives, {self.H}x{self.W} image...")
            

            # Convert existing primitive_tile_masks to tile_primitive_mapping format
            tile_primitive_mapping = self._convert_masks_to_mapping(
                primitive_tile_masks, x_starts, x_ends, y_starts, y_ends
            )
            
            if lr_conf is None:
                lr_config_tensor = torch.tensor([
                    0.1,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ], dtype=torch.float32, device=means2D.device)
            else:
                # Convert lr_conf dict to tensor (constants already merged in config)
                lr_config_tensor = torch.tensor([
                    lr_conf.get('default', 0.1),
                    lr_conf['gain_x'],
                    lr_conf['gain_y'],
                    lr_conf['gain_r'],
                    lr_conf['gain_v'],
                    lr_conf['gain_theta'],
                    lr_conf.get('gain_c', 1.0)
                ], dtype=torch.float16 if self.use_fp16 else torch.float32, device=means2D.device)
            

            if self.cuda_rasterizer is None:
                torch.cuda.empty_cache()  # Clear GPU memory
                self.cuda_rasterizer = TileRasterizer(
                    self.H, self.W, self.tile_size, sigma, 
                    self.alpha_upper_bound, self.max_prims_per_pixel, len(radii),
                    use_fp16=self.use_fp16
                )

            # Call CUDA rasterizer for all tiles at once
            # Use FP16 version if available and use_fp16 is True
            # Use TileRasterizer class-based version (already created above)
            cuda_color, cuda_alpha = self.cuda_rasterizer(
                means2D, radii, rotations, opacities, colors, c_o,
                primitive_templates, global_bmp_sel, c_blend,
                lr_config_tensor, tile_primitive_mapping
            )
            
            if DEBUG_MODE:
                print("    ✅ CUDA rasterizer call completed!")
                print(f"    📊 Output data ranges:")
                print(f"      cuda_color: [{cuda_color.min():.4f}, {cuda_color.max():.4f}]")
                print(f"      cuda_alpha: [{cuda_alpha.min():.4f}, {cuda_alpha.max():.4f}]")
                
                # Debug: Check if output is all zeros or all ones
                if cuda_color.min() == cuda_color.max():
                    print(f"    ⚠️ WARNING: cuda_color is uniform value: {cuda_color.min():.4f}")
                if cuda_alpha.min() == cuda_alpha.max():
                    print(f"    ⚠️ WARNING: cuda_alpha is uniform value: {cuda_alpha.min():.4f}")
                
                # Debug: Print sample primitive values
                print(f"    📊 Sample primitive values:")
                num_primitives = min(5, len(x))
                for i in range(num_primitives):
                    print(f"      Primitive {i}: x={x[i]:.4f}, y={y[i]:.4f}, r={r[i]:.4f}, v={v[i]:.4f}, theta={theta[i]:.4f}, c={c[i].tolist()}")
                
            # Handle background
            if I_bg is None:
                result = cuda_color
            else:
                result = cuda_color + (1 - cuda_alpha.unsqueeze(-1)) * I_bg

            if return_alpha:
                return result, cuda_alpha
            return result
            
        except Exception as e:
            print(f"    ❌ CUDA kernel call failed with exception: {e}")
            import traceback
            print(f"    📋 Exception traceback:")
            traceback.print_exc()
            # Return None to trigger fallback
            return None
    
    def _convert_masks_to_mapping(self, primitive_tile_masks: torch.Tensor,
                                 x_starts: torch.Tensor, x_ends: torch.Tensor,
                                 y_starts: torch.Tensor, y_ends: torch.Tensor) -> dict:
        """
        Convert primitive_tile_masks to tile_primitive_mapping format for CUDA.
        
        Args:
            primitive_tile_masks: (num_tiles, num_primitives) boolean tensor
            x_starts, x_ends, y_starts, y_ends: tile boundaries
            
        Returns:
            tile_primitive_mapping: dict with tile_offsets and tile_indices
        """
        num_tiles, num_primitives = primitive_tile_masks.shape
        
        # Calculate tile offsets (cumulative primitive counts)
        tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.int32, device=primitive_tile_masks.device)
        tile_indices = []
        
        for tile_idx in range(num_tiles):
            # Get primitives for this tile
            tile_mask = primitive_tile_masks[tile_idx]
            if tile_mask.any():
                indices = torch.nonzero(tile_mask, as_tuple=True)[0].to(torch.int32)
                tile_offsets[tile_idx + 1] = tile_offsets[tile_idx] + len(indices)
                tile_indices.append(indices)
            else:
                tile_offsets[tile_idx + 1] = tile_offsets[tile_idx]
                tile_indices.append(torch.empty(0, dtype=torch.int32, device=primitive_tile_masks.device))
        
        # Concatenate all tile indices
        if tile_indices:
            tile_indices = torch.cat(tile_indices)
        else:
            tile_indices = torch.empty(0, dtype=torch.int32, device=primitive_tile_masks.device)
        
        # Convert to tensor format for C++: [tile_offsets_size, tile_indices_size, tile_offsets..., tile_indices...]
        tile_offsets_size = len(tile_offsets)
        tile_indices_size = len(tile_indices)
        
        # Create concatenated tensor
        mapping_tensor = torch.cat([
            torch.tensor([tile_offsets_size, tile_indices_size], dtype=torch.int32, device=primitive_tile_masks.device),
            tile_offsets,
            tile_indices
        ])
        
        if DEBUG_MODE:
            # Validation logging
            print(f"    📊 Tile mapping validation:")
            print(f"      - Total tiles: {num_tiles}")
            print(f"      - Total primitive indices: {len(tile_indices)}")
            print(f"      - Tile offsets size: {len(tile_offsets)}")
            
            # Print first few tile offsets
            print(f"      - First 5 tile offsets: {tile_offsets[:5].tolist()}")
            if len(tile_offsets) > 5:
                print(f"      - Last 5 tile offsets: {tile_offsets[-5:].tolist()}")
            
            # Print first few primitive indices
            if len(tile_indices) > 0:
                print(f"      - First 10 primitive indices: {tile_indices[:10].tolist()}")
                
                # Verify indices are unique and in correct range
                unique_indices = torch.unique(tile_indices)
                print(f"      - Unique indices count: {len(unique_indices)} (should be <= total primitives)")
                print(f"      - Indices range: [{tile_indices.min()}, {tile_indices.max()}]")
                
                # Check for any duplicate indices (should not happen)
                if len(unique_indices) != len(tile_indices):
                    print(f"      ⚠️ WARNING: Duplicate indices found! {len(tile_indices)} total, {len(unique_indices)} unique")
                else:
                    print(f"      ✅ All indices are unique")
            
            # Calculate and print primitive distribution per tile
            print(f"      - Primitive distribution per tile:  (from python)")
            for tile_idx in range(min(10, num_tiles)):
                start_idx = tile_offsets[tile_idx].item()
                end_idx = tile_offsets[tile_idx + 1].item()
                num_prims = end_idx - start_idx
                print(f"        Tile {tile_idx}: {num_prims} primitives (indices {start_idx}-{end_idx-1})")
            
            if num_tiles > 10:
                print(f"        ... (showing first 10 tiles only)")
            
            # Verify total primitive count
            total_mapped_prims = tile_offsets[-1].item()
            print(f"      - Total mapped primitives: {total_mapped_prims} (should match indices_size: {len(tile_indices)})")
            
            if total_mapped_prims != len(tile_indices):
                print(f"      ⚠️ WARNING: Total mapped primitives ({total_mapped_prims}) != indices_size ({len(tile_indices)})")
            
            # Calculate average primitives per tile and duplication factor
            avg_prims_per_tile = total_mapped_prims / num_tiles if num_tiles > 0 else 0
            duplication_factor = total_mapped_prims / num_primitives if num_primitives > 0 else 0
            
            print(f"      - Average primitives per tile: {avg_prims_per_tile:.1f}")
            print(f"      - Duplication factor: {duplication_factor:.2f}x (should be reasonable, typically 1.5-3.0x)")
            
            if duplication_factor > 5.0:
                print(f"      ⚠️ WARNING: High duplication factor ({duplication_factor:.2f}x) - too many primitives per tile!")
            elif duplication_factor < 1.0:
                print(f"      ⚠️ WARNING: Low duplication factor ({duplication_factor:.2f}x) - some tiles may be empty!")
            else:
                print(f"      ✅ Duplication factor is reasonable")
            
            print(f"    📊 Tile mapping validation completed.")
        
        return mapping_tensor
    
    def render(self, cached_masks: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
               return_alpha: bool = False, I_bg: torch.Tensor = None) -> torch.Tensor:
        """
        VectorRenderer-compatible render method that uses pre-computed cached_masks.
        This method provides compatibility with the existing optimization pipeline.
        
        Args:
            cached_masks: (N, H, W) pre-computed soft masks
            v: (N,) visibility logits
            c: (N, 3) RGB logits
            return_alpha: Whether to return alpha channel
            I_bg: Background image
            
        Returns:
            Rendered image tensor
        """
        # Use parent class's render method for compatibility
        return super().render(cached_masks, v, c, return_alpha, I_bg)

    def _get_background_for_render(self, bg_type:str, export:bool = False) -> torch.Tensor:
        """
        Get background image tensor based on bg_type.
        Supports: white, black, random, and all rainbow colors (red, orange, yellow, green, blue, indigo, violet)
        
        Args:
            bg_type: Background type string
            export: If True, "random" background will be converted to white for consistent export
        """
        if bg_type == "white":
            I_bg = torch.ones((self.H, self.W, 3), device=self.device)
        elif bg_type == "black":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
        elif bg_type == "random":
            if export:
                # Use white background for exports when bg_type is random
                I_bg = torch.ones((self.H, self.W, 3), device=self.device)
            else:
                I_bg = torch.rand((self.H, self.W, 3), device=self.device)
        elif bg_type == "red":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 0] = 1.0
        elif bg_type == "orange":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 0] = 1.0
            I_bg[:, :, 1] = 0.647  # RGB(255, 165, 0)
        elif bg_type == "yellow":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 0] = 1.0
            I_bg[:, :, 1] = 1.0
        elif bg_type == "green":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 1] = 1.0
        elif bg_type == "blue":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 2] = 1.0
        elif bg_type == "indigo":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 0] = 0.294
            I_bg[:, :, 2] = 0.510  # RGB(75, 0, 130)
        elif bg_type == "violet" or bg_type == "purple":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 0] = 0.580
            I_bg[:, :, 2] = 0.827  # RGB(148, 0, 211)
        elif bg_type == "gray" or bg_type == "grey":
            I_bg = torch.full((self.H, self.W, 3), fill_value=0.5, device=self.device)
        elif bg_type == "pink":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 0] = 1.0
            I_bg[:, :, 1] = 0.753
            I_bg[:, :, 2] = 0.796  # RGB(255, 192, 203)
        elif bg_type == "cyan":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 1] = 1.0
            I_bg[:, :, 2] = 1.0
        elif bg_type == "magenta":
            I_bg = torch.zeros((self.H, self.W, 3), device=self.device)
            I_bg[:, :, 0] = 1.0
            I_bg[:, :, 2] = 1.0
        else:
            raise ValueError(f"Unsupported bg_type: {bg_type}. Supported types: white, black, random, red, orange, yellow, green, blue, indigo, violet/purple, gray/grey, pink, cyan, magenta")
        return I_bg
    
    def _optimize_parameters_whole(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                  v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                  target_image: torch.Tensor, opt_conf: dict,
                                  target_binary_mask: Optional[torch.Tensor] = None,
                                  initializer: Optional[Any] = None
                                  ):
        """
        Override optimization to use tile-based rendering instead of cached_masks.
        This is the core difference from VectorRenderer - we render directly from parameters.
        """
        try:
            from torch.amp import GradScaler, autocast
            AUTOCAST_NEW_API = True  # PyTorch 2.0+: use device_type='cuda'
        except ImportError:
            from torch.cuda.amp import GradScaler, autocast
            AUTOCAST_NEW_API = False  # PyTorch 1.x: use autocast() directly
        from tqdm import tqdm
        import datetime
        import os
        
        is_no_bg_mode = target_image.shape[2] == 4
        
        # Initialize loss composer from config
        loss_config = opt_conf.get("loss_config", {"type": "mse"})
        self.loss_composer = LossComposer(loss_config, device=self.device)
        print(f"Using loss configuration: {loss_config}")
        
        # Get optimization parameters from config
        num_iterations = opt_conf.get("num_iterations", 100)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Mixed-precision scaler (only used if use_fp16 is True)
        # PyTorch 2.0+: GradScaler('cuda'), PyTorch 1.x: GradScaler()
        if self.use_fp16:
            scaler = GradScaler('cuda') if AUTOCAST_NEW_API else GradScaler()
        else:
            scaler = None
        
        # Pre-calculate configurations
        blur_sigma = opt_conf.get("blur_sigma", 1.0)
        
        # Create output directory for saving images if it doesn't exist
        save_image_intervals = range(0, num_iterations, 10)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create optimizer (constants already merged in config)
        optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf['gain_x']},
            {'params': y, 'lr': lr*lr_conf['gain_y']},
            {'params': r, 'lr': lr*lr_conf['gain_r']},
            {'params': v, 'lr': lr*lr_conf['gain_v'] * (1000.0 / x.numel())},
            {'params': theta, 'lr': lr*lr_conf['gain_theta']},
            {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
        ])
        
        # Create scheduler if decay is enabled (constants already merged in config)
        do_decay = opt_conf.get("do_decay", False)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=opt_conf['decay_rate']) if do_decay else None
        
        # For Gaussian blur transition if enabled
        do_gaussian_blur = opt_conf.get("do_gaussian_blur", False)
        do_adapt_gaussian_blur = opt_conf.get("do_adapt_gaussian_blur", False)
        sigma_start = opt_conf.get("blur_sigma_start", 2.0)
        sigma_end = opt_conf.get("blur_sigma_end", 0.0)
        prune_conf = opt_conf.get("pruning", {})
        do_pruning = prune_conf.get("do_pruning", False)

        print(f"Starting tile-based optimization, {num_iterations} iterations... self.use_fp16: {self.use_fp16}")

        # Adjust lr_conf['gain_v'] to include scaling as in vector_renderer.py
        lr_conf['gain_v'] = lr_conf.get('gain_v', 1.0) * (1000.0 / x.numel())
        
        # Initialize MP4 recording if requested
        record_mp4 = opt_conf.get("record_optimization", False)
        if record_mp4:
            optimization_history = {
                'x': [],
                'y': [],
                'r': [],
                'v': [],
                'theta': [],
                'c': []
            }
            print("🎬 Recording optimization process for MP4 export...")
        
        # Optimization loop
        for iteration in tqdm(range(num_iterations), desc="Optimizing"):
            I_bg = self._get_background_for_render(opt_conf.get("bg_color", "white"))
            optimizer.zero_grad()

            # Pruning(Restart)
            if do_pruning and iteration !=0:
                prune_conf = opt_conf.get("pruning", {})
                prune_iterations = prune_conf.get("prune_iterations", 5)
                no_prune_last_iterations = prune_conf.get("no_prune_last_iterations", 10)
                no_prune_warmup_iterations = prune_conf.get("no_prune_warmup_iterations", 20)
                if iteration % prune_iterations == 0 and iteration < num_iterations - no_prune_last_iterations and iteration > no_prune_warmup_iterations:
                    self.do_prune(x,y,r,v,theta,c,prune_conf,initializer,target_image)         

            # Adaptive Gaussian blur
            if do_adapt_gaussian_blur:
                progress = iteration / num_iterations
                current_sigma = sigma_start * (1 - progress) + sigma_end * progress
            elif do_gaussian_blur:
                current_sigma = blur_sigma
            else:
                current_sigma = 0.0
            
            # Use tile-based rendering directly from parameters
            if self.use_fp16:
                autocast_ctx = autocast(device_type='cuda') if AUTOCAST_NEW_API else autocast()
                with autocast_ctx:
                    if is_no_bg_mode:
                        rendered, rendered_alpha = self.render_from_params(
                            x, y, r, theta, v, c, sigma=current_sigma, I_bg=I_bg, lr_conf=lr_conf, return_alpha=True
                        )
                    else:
                        rendered = self.render_from_params(
                            x, y, r, theta, v, c, sigma=current_sigma, I_bg=I_bg, lr_conf=lr_conf
                        )
                        rendered_alpha = None
                    
                    # Compute loss
                    loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c, 
                            rendered_alpha=rendered_alpha)
                    
                # Scale the loss and call backward
                start_backward_time = time.time()
                scaler.scale(loss).backward()
                end_backward_time = time.time()
                
                # Update PyTorch backward timing statistics
                backward_time_ms = (end_backward_time - start_backward_time) * 1000.0
                self.pytorch_backward_time += backward_time_ms
                self.pytorch_backward_count += 1
                
                scaler.step(optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()

            else:
                if is_no_bg_mode:
                    rendered, rendered_alpha = self.render_from_params(
                        x, y, r, theta, v, c, sigma=current_sigma, I_bg=I_bg, lr_conf=lr_conf, return_alpha=True
                    )
                else:
                    rendered = self.render_from_params(
                        x, y, r, theta, v, c, sigma=current_sigma, I_bg=I_bg, lr_conf=lr_conf
                    )
                    rendered_alpha = None
                
                # Compute loss
                loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c, 
                                         rendered_alpha=rendered_alpha)
                
                # Backward pass
                start_backward_time = time.time()
                loss.backward()
                end_backward_time = time.time()
                
                # Update PyTorch backward timing statistics
                backward_time_ms = (end_backward_time - start_backward_time) * 1000.0
                self.pytorch_backward_time += backward_time_ms
                self.pytorch_backward_count += 1
                
                if DEBUG_MODE_DETAIL:
                    print(f"\n🔍 Gradient Analysis (First 10 Primitives) - Iteration {iteration}")
                    print("=" * 80)
                    
                    # Check if gradients exist
                    if x.grad is not None:
                        print("📊 X Gradients (First 10):")
                        for i in range(min(10, len(x))):
                            print(f"  Primitive {i}: {x.grad[i].item():.6f}")
                    
                    if y.grad is not None:
                        print("📊 Y Gradients (First 10):")
                        for i in range(min(10, len(y))):
                            print(f"  Primitive {i}: {y.grad[i].item():.6f}")
                    
                    if r.grad is not None:
                        print("📊 R Gradients (First 10):")
                        for i in range(min(10, len(r))):
                            print(f"  Primitive {i}: {r.grad[i].item():.6f}")
                    
                    if theta.grad is not None:
                        print("📊 Theta Gradients (First 10):")
                        for i in range(min(10, len(theta))):
                            print(f"  Primitive {i}: {theta.grad[i].item():.6f}")
                    
                    if v.grad is not None:
                        print("📊 V (Opacity) Gradients (First 10):")
                        for i in range(min(10, len(v))):
                            print(f"  Primitive {i}: {v.grad[i].item():.6f}")
                    
                    if c.grad is not None:
                        print("📊 C (Color) Gradients (First 10):")
                        for i in range(min(10, len(c))):
                            print(f"  Primitive {i}: R={c.grad[i,0].item():.6f}, G={c.grad[i,1].item():.6f}, B={c.grad[i,2].item():.6f}")
                    
                    print("=" * 80)

                # Debug: Check gradients after backward
                if DEBUG_MODE and iteration % 10 == 0:
                    print(f"  Debug - x.grad: {x.grad.abs().mean().item() if x.grad is not None else 'None'}")
                    print(f"  Debug - y.grad: {y.grad.abs().mean().item() if y.grad is not None else 'None'}")
                    print(f"  Debug - r.grad: {r.grad.abs().mean().item() if r.grad is not None else 'None'}")
                
                optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            self._clamp_params_inplace(x, y, r)

            # Record parameters for MP4 if requested
            if record_mp4:
                optimization_history['x'].append(x.detach().clone())
                optimization_history['y'].append(y.detach().clone())
                optimization_history['r'].append(r.detach().clone())
                optimization_history['v'].append(v.detach().clone())
                optimization_history['theta'].append(theta.detach().clone())
                optimization_history['c'].append(c.detach().clone())

            if DEBUG_MODE:
                print(f"    📊 Input data ranges: iteration {iteration}")
                print(f"      x: [{x.min():.4f}, {x.max():.4f}]")
                print(f"      y: [{y.min():.4f}, {y.max():.4f}]")
                print(f"      r: [{r.min():.4f}, {r.max():.4f}]")
                print(f"      v: [{v.min():.4f}, {v.max():.4f}]")
                print(f"      theta: [{theta.min():.4f}, {theta.max():.4f}]")
                print(f"      c: [{c.min():.4f}, {c.max():.4f}]")
            
            # Log progress with loss components
            if iteration % 10 == 0:
                # Get loss components for detailed logging
                _, loss_components = self.compute_loss(
                    rendered, target_image, x, y, r, v, theta, c, 
                    rendered_alpha=rendered_alpha,
                    epoch=iteration,
                    return_components=True
                )
                
                # Format loss components string
                components_str = ", ".join([f"{name}={val:.6f}" for name, val in loss_components.items()])
                print(f"Iteration {iteration}: Total={loss.item():.6f} ({components_str})")
                  
            # Save intermediate images
            if iteration in save_image_intervals and DEBUG_MODE_SAVE:
                img_path = os.path.join(self.output_path, f"tile_iter_{iteration:04d}_{timestamp}.png")
                if rendered_alpha is not None:
                    rendered_to_save = torch.cat([rendered, rendered_alpha.unsqueeze(-1)], dim=-1)
                else:
                    rendered_to_save = rendered
                self.save_image_tensor(rendered_to_save, img_path)
        
        print(f"Tile-based optimization completed. Final loss: {loss.item():.6f}")
        
        # Export MP4 if recording was enabled
        if record_mp4:
            self._export_optimization_mp4(
                optimization_history,
                I_bg=self._get_background_for_render(opt_conf.get("bg_color", "white"), export=True),
                timestamp=timestamp,
                opt_conf=opt_conf
            )
        
        used = vram_used_by_pid()
        mb = lambda b: b / (1024**2)
        gb = lambda b: b / (1024**3)
        
        # Print CUDA timing statistics
        if CUDA_AVAILABLE:
            print("\n" + "="*60)
            print("CUDA Performance Statistics:")
            print("="*60)
            if self.use_fp16:
                print_cuda_timing_stats_fp16()
            else:
                print_cuda_timing_stats()
            print("-"*60)
            print(f"VRAM used: {mb(used):.0f} MiB, {gb(used):.3f} GB")
            print("="*60)
        else:
            # Print PyTorch timing statistics
            print("\n" + "="*60)
            print("PyTorch Performance Statistics:")
            print("="*60)
            print("Forward Pass:")
            print(f"  Total time: {self.pytorch_forward_time:.2f} ms")
            print(f"  Iterations: {self.pytorch_forward_count}")
            if self.pytorch_forward_count > 0:
                print(f"  Average time per iteration: {self.pytorch_forward_time / self.pytorch_forward_count:.2f} ms")
            
            print("Backward Pass:")
            print(f"  Total time: {self.pytorch_backward_time:.2f} ms")
            print(f"  Iterations: {self.pytorch_backward_count}")
            if self.pytorch_backward_count > 0:
                print(f"  Average time per iteration: {self.pytorch_backward_time / self.pytorch_backward_count:.2f} ms")
            
            print("Combined:")
            print(f"  Total time: {self.pytorch_forward_time + self.pytorch_backward_time:.2f} ms")
            print(f"  Total iterations: {self.pytorch_forward_count + self.pytorch_backward_count}")
            if (self.pytorch_forward_count + self.pytorch_backward_count) > 0:
                avg_time = (self.pytorch_forward_time + self.pytorch_backward_time) / (self.pytorch_forward_count + self.pytorch_backward_count)
                print(f"  Average time per iteration: {avg_time:.2f} ms")
            print("-"*60)
            print(f"VRAM used: {mb(used):.0f} MiB, {gb(used):.3f} GB")
            print("="*60)
        
        return x, y, r, v, theta, c

    def do_prune(self, x , y , r, v, theta, c, prune_conf, initializer, target_image) -> None:
        """
        Prune the low-opacity primitives and re-initialize them using the initializer.
        
        Args:
            x, y, r, v, theta, c: Primitive parameters
            prune_conf: Pruning configuration
            initializer: StructureAwareInitializer instance with adjusted_pts and sampled_variances
            target_binary_mask: Binary mask for target regions
            target_image: Target image tensor for color sampling
        """
        print("Re-initializing low-opacity primitives...")
        initializer.decrease_max_radius += 1
        prune_threshold = prune_conf.get("prune_threshold", 0.3)
        # Find primitives with low opacity
        opacity = torch.sigmoid(v)
        low_opacity_mask = opacity < prune_threshold
        num_to_prune = low_opacity_mask.sum().item()

        if num_to_prune == 0:
            print("No primitives to prune.")
            return

        print(f"Pruning {num_to_prune} low-opacity primitives (threshold: {prune_threshold})")
        prune_indices = torch.where(low_opacity_mask)[0]
        try:
            # Use initializer's reinitialize_subset method
            new_x, new_y, new_r, new_v, new_theta, new_c = initializer.reinitialize_subset(
                num_to_prune, target_image[:,:,:3], x.device
            )
            # Update pruned primitives
            with torch.no_grad():
                x.data[prune_indices] = new_x
                y.data[prune_indices] = new_y
                r.data[prune_indices] = new_r
                v.data[prune_indices] = new_v
                theta.data[prune_indices] = new_theta
                c.data[prune_indices] = new_c
            print(f"✓ Re-initialized {num_to_prune} primitives using StructureAwareInitializer")
        except Exception as e:
            print(f"Warning: Reinitialize failed ({e}). Using fallback.")
            self._fallback_random_init(x, y, r, v, theta, c, prune_indices, num_to_prune)

    def _fallback_random_init(self, x, y, r, v, theta, c, indices, num_points):
        """Fallback random initialization for pruned primitives."""
        with torch.no_grad():
            x.data[indices] = torch.rand(num_points, device=x.device) * self.W
            y.data[indices] = torch.rand(num_points, device=y.device) * self.H
            r.data[indices] = torch.rand(num_points, device=r.device) * (min(self.H, self.W) / 20) + 2
            v.data[indices] = torch.full((num_points,), -2.0, device=v.device)
            theta.data[indices] = torch.rand(num_points, device=theta.device) * 2 * np.pi
            c.data[indices] = torch.rand(num_points, 3, device=c.device)
        print(f"✓ Fallback: Randomly re-initialized {num_points} primitives")


    def save_image_tensor(self, image_tensor: torch.Tensor, path: str):
        """Save image tensor to file."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert to numpy and ensure correct range [0, 1]
        img_np = image_tensor.detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1).astype(np.float32)
        
        # Save using matplotlib
        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # compute_loss is inherited from VectorRenderer
         
    def _render_tile(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                     theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                     primitive_indices: List[int], x_start: int, x_end: int,
                     y_start: int, y_end: int, sigma: float,
                     I_bg: torch.Tensor, global_bmp_sel: torch.Tensor = None, 
                     is_final: bool = False, return_alpha: bool = False) -> torch.Tensor:
        """
        Render a single tile with only the selected primitives.
        
        Args:
            x, y, r, theta: Full primitive parameters
            v, c: Full visibility and color parameters
            primitive_indices: Indices of primitives that affect this tile
            x_start, x_end, y_start, y_end: Tile boundaries
            sigma: Gaussian blur std
            I_bg: Background image
            
        Returns:
            Rendered tile (tile_h, tile_w, 3)
        """
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        if len(primitive_indices) == 0:
            # Empty tile
            if self.use_fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32
            return torch.zeros((tile_h, tile_w, 3), device=self.device, dtype=dtype)
        
        # Select only relevant primitives
        indices = torch.tensor(primitive_indices, device=self.device)
        tile_x = x[indices]
        tile_y = y[indices]
        tile_r = r[indices]
        tile_theta = theta[indices]
        tile_v = v[indices]
        tile_c = c[indices]
        if self.c_o is not None:
            # Use primitive-specific color maps based on global_bmp_sel for this tile
            tile_global_bmp_sel = global_bmp_sel[indices]  # (num_primitives_in_tile,)
            tile_c_o = self.c_o[tile_global_bmp_sel]  # (num_primitives_in_tile, H, W, 3) - select color maps for each primitive
        else:
            print(f"❌ Debug - self.c_o is None")
            return None
        tile_c_blend = torch.tensor(self.c_blend, device=self.device, dtype=torch.float16 if self.use_fp16 else torch.float32)  # Convert to tensor
        
        # Create coordinate grid for this tile
        tile_X = self.X[:, y_start:y_end, x_start:x_end]  # (1, tile_h, tile_w)
        tile_Y = self.Y[:, y_start:y_end, x_start:x_end]  # (1, tile_h, tile_w)
        tile_h, tile_w = tile_X.shape[1], tile_X.shape[2]
        
        # Try CUDA acceleration if available
        if CUDA_AVAILABLE and len(primitive_indices) > 0:
            try:
                # Prepare data for CUDA - ensure all tensors are float32
                means2D = torch.stack([tile_x, tile_y], dim=1).float()  # (N, 2)
                radii = tile_r.float()  # (N,)
                rotations = tile_theta.float()  # (N,)
                opacities = tile_v.float()  # (N,) logits
                colors = tile_c.float()  # (N, 3) logits
                
                # Get primitive templates for selected primitives based on is_final flag
                if is_final or self.S_blurred is None:
                    # Use original templates for final rendering
                    if global_bmp_sel is not None:
                        selected_templates = self.S[global_bmp_sel[primitive_indices]].float()  # (N, H, W)
                    else:
                        selected_templates = self.S[primitive_indices].float()  # (N, H, W)
                else:
                    # Use blurred templates for optimization
                    if global_bmp_sel is not None:
                        selected_templates = self.S_blurred[global_bmp_sel[primitive_indices]].float()  # (N, H, W)
                    else:
                        selected_templates = self.S_blurred[primitive_indices].float()  # (N, H, W)
                
                # Note: This is fallback PyTorch implementation since cuda_rasterize_tiles is not available
                # Will fall through to PyTorch implementation below
                raise NotImplementedError("CUDA single tile rendering not implemented")
                
            except Exception as e:
                # Fallback to PyTorch
                _start_time = time.time()
                tile_masks = self._generate_tile_masks(
                    tile_x, tile_y, tile_r, tile_theta, tile_X, tile_Y, sigma,
                    global_primitive_indices=primitive_indices,
                    global_bmp_sel=global_bmp_sel, is_final=is_final
                )
                
                # Convert logits to actual values
                alpha = torch.sigmoid(tile_v) * self.alpha_upper_bound
                rgb = torch.sigmoid(tile_c)
                
                # Apply c_o and c_blend
                c_o_sigmoid = torch.sigmoid(tile_c_o)
                c_blend_sigmoid = torch.sigmoid(tile_c_blend)
                rgb = rgb * (1 - c_blend_sigmoid) + c_o_sigmoid * c_blend_sigmoid
                
                # Apply alpha to masks
                a = tile_masks * alpha.view(-1, 1, 1)
                
                # Create premultiplied colors
                m = a.unsqueeze(-1) * rgb.view(-1, 1, 1, 3)
                
                # Composite using parent's function
                comp_m, comp_a = self._transmit_over(m, a)
                _end_time = time.time()
                self._forward_compute_time_accum += (_end_time - _start_time) * 1000.0
        else:
            # Generate masks for selected primitives in this tile
            _start_time = time.time()
            tile_masks = self._generate_tile_masks(
                tile_x, tile_y, tile_r, tile_theta, tile_X, tile_Y, sigma,
                global_primitive_indices=primitive_indices,
                global_bmp_sel=global_bmp_sel, is_final=is_final
            )
            
            # Convert logits to actual values
            alpha = torch.sigmoid(tile_v) * self.alpha_upper_bound
            rgb = torch.sigmoid(tile_c)
            
            # Apply c_o and c_blend
            c_o_sigmoid = torch.sigmoid(tile_c_o)
            # Reshape c_o_sigmoid to match rgb shape: (num_primitives_in_tile, 3)
            # Take the center pixel or average of each primitive's color map
            h_center, w_center = tile_c_o.shape[1] // 2, tile_c_o.shape[2] // 2
            c_o_sigmoid_center = c_o_sigmoid[:, h_center, w_center, :]  # (num_primitives_in_tile, 3)
            
            c_blend_sigmoid = torch.sigmoid(tile_c_blend).item()  # Convert to scalar
            rgb = rgb * (1 - c_blend_sigmoid) + c_o_sigmoid_center * c_blend_sigmoid
            
            # Apply alpha to masks
            a = tile_masks * alpha.view(-1, 1, 1)
            
            # Create premultiplied colors
            m = a.unsqueeze(-1) * rgb.view(-1, 1, 1, 3)
            
            # Composite using parent's function
            comp_m, comp_a = self._transmit_over(m, a)
            _end_time = time.time()
            self._forward_compute_time_accum += (_end_time - _start_time) * 1000.0
        
        # Handle background
        if I_bg is None:
            result = comp_m
        else:
            bg_tile = I_bg[y_start:y_end, x_start:x_end]
            result = comp_m + (1 - comp_a.unsqueeze(-1)) * bg_tile
        
        if return_alpha:
            return result, comp_a
        else:
            return result
    
    def _generate_tile_masks(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                            theta: torch.Tensor, tile_X: torch.Tensor, tile_Y: torch.Tensor,
                            sigma: float, global_primitive_indices: List[int] = None,
                            global_bmp_sel: torch.Tensor = None, is_final: bool = False) -> torch.Tensor:
        """
        Generate masks for primitives within a tile using actual self.S primitives.
        Based on _batched_soft_rasterize logic but for tile regions only.
        
        Args:
            x, y, r, theta: Selected primitive parameters
            tile_X, tile_Y: Coordinate grids for this tile (1, tile_h, tile_w)
            sigma: Gaussian blur std
            
        Returns:
            Tile masks (num_selected, tile_h, tile_w)
        """
        from contextlib import nullcontext
        try:
            from torch.amp import autocast
            AUTOCAST_NEW_API = True
        except ImportError:
            from torch.cuda.amp import autocast
            AUTOCAST_NEW_API = False
        from pydiffbmp.util.utils import gaussian_blur
        
        num_primitives = x.shape[0]
        tile_h, tile_w = tile_X.shape[1], tile_X.shape[2]
        
        if num_primitives == 0:
            if self.use_fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32
            return torch.zeros((0, tile_h, tile_w), device=self.device, dtype=dtype)
        
        autocast_ctx = autocast(device_type='cuda') if AUTOCAST_NEW_API else autocast()
        context = autocast_ctx if self.use_fp16 else nullcontext()
        with context:
            target_dtype = torch.float16 if self.use_fp16 else torch.float32
            
            # Prepare the bitmap(s) - same logic as _batched_soft_rasterize
            if sigma > 0.0:
                bmp = self.S.unsqueeze(0)  # -> [1, p, H, W] or [1, H, W]
                bmp = gaussian_blur(bmp, sigma)
                bmp_image = bmp.squeeze(0).to(dtype=target_dtype).contiguous()
            else:
                bmp_image = self.S.to(dtype=target_dtype)
            
            # Expand tile coordinates
            X_exp = tile_X.expand(num_primitives, tile_h, tile_w)
            Y_exp = tile_Y.expand(num_primitives, tile_h, tile_w)
            x_exp = x.view(num_primitives, 1, 1).expand(num_primitives, tile_h, tile_w)
            y_exp = y.view(num_primitives, 1, 1).expand(num_primitives, tile_h, tile_w)
            r_exp = r.view(num_primitives, 1, 1).expand(num_primitives, tile_h, tile_w)
            
            # Normalize and rotate positions
            pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            R_inv = torch.zeros(num_primitives, 2, 2, device=self.device)
            R_inv[:, 0, 0] = cos_t; R_inv[:, 0, 1] = sin_t
            R_inv[:, 1, 0] = -sin_t; R_inv[:, 1, 1] = cos_t
            uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
            grid = uv.permute(0, 2, 3, 1)  # (num_primitives, tile_h, tile_w, 2)
            
            # Build bmp_exp: one bitmap per instance, cycling through p if provided
            if bmp_image.dim() == 2:
                # single primitive template [H, W]
                bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(num_primitives, 1, -1, -1).contiguous()
            elif bmp_image.dim() == 3:
                # sequence of p templates [p, H, W]
                if global_bmp_sel is not None and global_primitive_indices is not None:
                    # Use global template selection - map global indices to templates
                    tile_bmp_indices = [global_bmp_sel[i].item() for i in global_primitive_indices]
                    tile_bmp_indices = torch.tensor(tile_bmp_indices, device=self.device, dtype=torch.long)
                    bmp_sel = bmp_image[tile_bmp_indices, :, :]  # [num_primitives, H, W]
                    bmp_exp = bmp_sel.unsqueeze(1).contiguous()  # [num_primitives, 1, H, W]
                else:
                    # Fallback: use local modulo (old behavior)
                    p = bmp_image.size(0)
                    idx = torch.arange(num_primitives, device=self.device, dtype=torch.long) % p
                    idx = idx.flip(0)  # Same as original
                    bmp_sel = bmp_image[idx, :, :]  # [num_primitives, H, W]
                    bmp_exp = bmp_sel.unsqueeze(1).contiguous()  # [num_primitives, 1, H, W]
            else:
                raise ValueError(f"Unsupported self.S shape: {bmp_image.shape}")
            
            # Sample masks via grid_sample
            sampled = F.grid_sample(bmp_exp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            
            # Return single-channel masks
            return sampled.squeeze(1)  # (num_primitives, tile_h, tile_w)


    def _export_optimization_mp4(self, 
                                 optimization_history: dict,
                                 I_bg: torch.Tensor,
                                 timestamp: str,
                                 opt_conf: dict) -> None:
        """
        Export MP4 video showing the optimization process.
        
        Args:
            optimization_history: Dictionary containing lists of x, y, r, v, theta, c for each iteration
            I_bg: Background image tensor
            timestamp: Timestamp string for filename
            opt_conf: Optimization configuration dictionary
            post_conf: Postprocessing configuration dictionary
        """
        import cv2
        import os
        
        num_frames = len(optimization_history['x'])
        fps = opt_conf.get("mp4_fps")
        
        print(f"\n{'='*80}")
        print(f"🎬 Exporting optimization MP4: {num_frames} frames at {fps} FPS")
        print(f"{'='*80}")
        
        # Create output path
        video_path = os.path.join(self.output_path, f'output_{timestamp}.mp4')
        
        # Direct MP4 output with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (self.W, self.H))
        
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        
        try:
            # Render each frame from recorded parameters
            last_frame = None
            for frame_idx in range(num_frames):
                if frame_idx % max(1, num_frames // 20) == 0:
                    print(f"Rendering frame {frame_idx+1}/{num_frames}...")
                
                # Get parameters for this iteration
                x = optimization_history['x'][frame_idx]
                y = optimization_history['y'][frame_idx]
                r = optimization_history['r'][frame_idx]
                v = optimization_history['v'][frame_idx]
                theta = optimization_history['theta'][frame_idx]
                c = optimization_history['c'][frame_idx]
                
                # Render frame with current parameters
                with torch.no_grad():
                    frame = self.render_from_params(
                        x, y, r, theta, v, c,
                        sigma=0.0, I_bg=I_bg
                    )
                
                self._write_frame(writer, frame)
                last_frame = frame
                
                # Clear GPU memory periodically
                if frame_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Hold the last frame for 1 second
            if last_frame is not None:
                hold_frames = fps  # 1 second worth of frames
                print(f"Adding {hold_frames} hold frames (1 second)...")
                for _ in range(hold_frames):
                    self._write_frame(writer, last_frame)
            
            print(f"✅ Optimization video saved to: {video_path}")
            print(f"{'='*80}\n")
            
        finally:
            writer.release()
    
    def _write_frame(self, writer, frame_tensor: torch.Tensor) -> None:
        """
        Convert tensor frame to BGR format and write to video.
        
        Args:
            writer: OpenCV VideoWriter object
            frame_tensor: RGB frame tensor (H, W, 3) in range [0, 1]
        """
        import cv2
        import numpy as np
        
        # Convert to numpy and scale to [0, 255]
        frame_np = frame_tensor.detach().cpu().numpy()
        frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        writer.write(frame_bgr)
                

if __name__ == "__main__":
    # Test the SimpleTileRenderer
    print("Testing SimpleTileRenderer...")
    
    import numpy as np
    
    # Create test primitives (X shape and hollow rectangle)
    def create_x_primitive(size=128, thickness=0.1):
        """Create an X-shaped primitive that shows rotation clearly"""
        y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
        
        # Create two diagonal lines
        line1 = np.abs(y - x) < thickness  # Main diagonal
        line2 = np.abs(y + x) < thickness  # Anti-diagonal
        
        x_shape = (line1 | line2).astype(np.float32)
        return torch.tensor(x_shape, dtype=torch.float32)
    
    def create_hollow_rect_primitive(size=128, outer_size=0.8, thickness=0.15):
        """Create a hollow rectangle that shows rotation and scaling"""
        y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
        
        # Outer rectangle
        outer = (np.abs(x) <= outer_size) & (np.abs(y) <= outer_size)
        
        # Inner rectangle (hollow part)
        inner_size = outer_size - thickness
        inner = (np.abs(x) <= inner_size) & (np.abs(y) <= inner_size)
        
        hollow_rect = (outer & ~inner).astype(np.float32)
        return torch.tensor(hollow_rect, dtype=torch.float32)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test primitives
    x_primitive = create_x_primitive().to(device)
    hollow_rect_primitive = create_hollow_rect_primitive().to(device)
    
    # Stack primitives for variety
    primitives = torch.stack([x_primitive, hollow_rect_primitive], dim=0)  # Shape: (2, 128, 128)
    print(f"Primitives shape: {primitives.shape}")
    print(f"  - X primitive: {x_primitive.shape}")
    print(f"  - Hollow rectangle: {hollow_rect_primitive.shape}")
    
    # Create renderer
    canvas_size = (194, 256)
    tile_size = 32
    renderer = SimpleTileRenderer(
        canvas_size=canvas_size,
        tile_size=tile_size,
        S=primitives,  # Use both primitives: (2, 128, 128)
        device=device
    )
    
    print(f"Canvas size: {canvas_size}")
    print(f"Tile size: {tile_size}")
    print(f"Tiles grid: {renderer.tiles_h} x {renderer.tiles_w}")
    print(f"Primitive bboxes: {renderer.primitive_bboxes}")
    
    # Create test parameters
    N = 50  # Number of primitives
    x = torch.rand(N, device=device) * canvas_size[1]  # Random x positions
    y = torch.rand(N, device=device) * canvas_size[0]  # Random y positions
    r = torch.rand(N, device=device) * 28 + 2  # Random scales 2-30
    theta = torch.rand(N, device=device) * 2 * np.pi  # Random rotations
    v = torch.rand(N, device=device) * 2 - 1  # Random visibility logits
    c = torch.rand(N, 3, device=device) * 2 - 1  # Random color logits
    
    print(f"\nTest parameters:")
    print(f"  N primitives: {N}")
    print(f"  x range: [{x.min():.1f}, {x.max():.1f}]")
    print(f"  y range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  r range: [{r.min():.1f}, {r.max():.1f}]")
    
    # Test rendering
    print("\nRendering...")
    try:
        lr_conf = {
            'default': 0.1,
            'gain_x': 1.0,
            'gain_y': 1.0,
            'gain_r': 1.0,
            'gain_v': 1.0,
            'gain_theta': 1.0,
            'gain_c': 1.0
        }

        result = renderer.render_from_params(x, y, r, theta, v, c, sigma=1.0, lr_conf=lr_conf)
        print(f"Success! Result shape: {result.shape}")
        print(f"Result range: [{result.min():.3f}, {result.max():.3f}]")
        
        # Visualization
        print("\n🎨 Creating visualizations...")
        import matplotlib.pyplot as plt
        import os
        
        # Create output directory
        output_dir = "simple_tile_renderer_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save rendered result
        plt.figure(figsize=(8, 8))
        result_np = result.detach().cpu().numpy()
        plt.imshow(result_np)
        plt.title(f"SimpleTileRenderer Result ({N} primitives)")
        plt.axis('off')
        plt.tight_layout()
        result_path = os.path.join(output_dir, "rendered_result.png")
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  🖼️ Saved rendered result: {result_path}")
        
        # 2. Create tile-primitive distribution visualization
        print("\n🗺️ Analyzing tile-primitive distribution...")
        tile_primitive_counts = np.zeros((renderer.tiles_h, renderer.tiles_w))
        
        for tile_y in range(renderer.tiles_h):
            for tile_x in range(renderer.tiles_w):
                y_start = tile_y * tile_size
                y_end = min(y_start + tile_size, canvas_size[0])
                x_start = tile_x * tile_size
                x_end = min(x_start + tile_size, canvas_size[1])
                
                tile_primitives = renderer._unified_primitive_assignment(
                    x, y, r, theta, x_start, x_end, y_start, y_end
                )
                tile_primitive_counts[tile_y, tile_x] = len(tile_primitives)
        
        # Plot tile-primitive distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Primitive positions with tile grid
        ax1.imshow(result_np, alpha=0.3)
        
        # Draw tile grid
        for i in range(renderer.tiles_w + 1):
            ax1.axvline(x=i * tile_size - 0.5, color='red', alpha=0.5, linewidth=1)
        for i in range(renderer.tiles_h + 1):
            ax1.axhline(y=i * tile_size - 0.5, color='red', alpha=0.5, linewidth=1)
        
        # Plot primitive positions
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        
        scatter = ax1.scatter(x_np, y_np, s=r_np*2, c=range(N), 
                            cmap='tab10', alpha=0.8, edgecolors='black')
        ax1.set_xlim(0, canvas_size[1])
        ax1.set_ylim(canvas_size[0], 0)  # Flip y-axis
        ax1.set_title(f"Primitive Positions & Tile Grid ({tile_size}x{tile_size})")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        
        # Add primitive indices as labels
        for i in range(N):
            ax1.annotate(str(i), (x_np[i], y_np[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='white', weight='bold')
        
        # Right: Tile-primitive count heatmap
        im = ax2.imshow(tile_primitive_counts, cmap='viridis', interpolation='nearest')
        ax2.set_title("Primitives per Tile")
        ax2.set_xlabel("Tile X")
        ax2.set_ylabel("Tile Y")
        
        # Add text annotations
        for i in range(renderer.tiles_h):
            for j in range(renderer.tiles_w):
                count = int(tile_primitive_counts[i, j])
                color = 'white' if count > tile_primitive_counts.max() / 2 else 'black'
                ax2.text(j, i, str(count), ha='center', va='center', 
                        color=color, fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='Number of Primitives')
        plt.tight_layout()
        
        distribution_path = os.path.join(output_dir, "tile_primitive_distribution.png")
        plt.savefig(distribution_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  🗺️ Saved distribution analysis: {distribution_path}")
        
        # 3. Create primitive detail visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create a detailed view of primitives
        colors = plt.cm.tab10(np.linspace(0, 1, N))
        
        for i in range(N):
            # Draw primitive bounding box (transformed)
            prim_idx = i % len(renderer.primitive_bboxes)
            min_u, max_u, min_v, max_v = renderer.primitive_bboxes[prim_idx]
            
            cos_t = np.cos(theta[i].item())
            sin_t = np.sin(theta[i].item())
            scale = r[i].item()
            
            # Bounding box corners
            corners_u = np.array([min_u, max_u, max_u, min_u, min_u])
            corners_v = np.array([min_v, min_v, max_v, max_v, min_v])
            
            # Transform to world coordinates
            world_x = x[i].item() + scale * (corners_u * cos_t - corners_v * sin_t)
            world_y = y[i].item() + scale * (corners_u * sin_t + corners_v * cos_t)
            
            ax.plot(world_x, world_y, color=colors[i], linewidth=2, alpha=0.7)
            ax.fill(world_x, world_y, color=colors[i], alpha=0.2)
            
            # Center point
            ax.scatter(x[i].item(), y[i].item(), color=colors[i], s=50, 
                      edgecolors='black', zorder=5)
            ax.annotate(f'{i}', (x[i].item(), y[i].item()), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, weight='bold')
        
        ax.set_xlim(0, canvas_size[1])
        ax.set_ylim(canvas_size[0], 0)
        ax.set_title("Primitive Bounding Boxes (Transformed)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        
        primitive_path = os.path.join(output_dir, "primitive_bboxes.png")
        plt.savefig(primitive_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  🔲 Saved primitive bboxes: {primitive_path}")
        
        # Test tile primitive selection for a few tiles
        print("\n🔍 Testing tile primitive selection...")
        test_tiles = [(0, 0), (2, 2), (4, 4)]
        for tile_y, tile_x in test_tiles:
            y_start = tile_y * tile_size
            y_end = min(y_start + tile_size, canvas_size[0])
            x_start = tile_x * tile_size
            x_end = min(x_start + tile_size, canvas_size[1])
            
            tile_primitives = renderer._unified_primitive_assignment(
                x, y, r, theta, x_start, x_end, y_start, y_end
            )
            print(f"  Tile ({tile_y},{tile_x}) [{x_start}-{x_end}, {y_start}-{y_end}]: {len(tile_primitives)} primitives {tile_primitives}")
        
        print(f"\n📁 All visualizations saved to: {output_dir}/")
        print("\n✅ SimpleTileRenderer test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        import traceback
        traceback.print_exc()

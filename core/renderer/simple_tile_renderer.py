import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .vector_renderer import VectorRenderer
from util.utils import gaussian_blur
import numpy as np
import cv2

DEBUG_MODE = False
DEBUG_MODE_DETAIL = False

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
    from cuda_tile_rasterizer import TileRasterizer
    CUDA_AVAILABLE = True
    print("CUDA tile rasterizer loaded successfully!")
            
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"CUDA tile rasterizer not available, using PyTorch fallback: {e}")


class SimpleTileRenderer(VectorRenderer):
    """
    Memory-efficient tile-based renderer that dynamically selects primitives per tile.
    Only processes primitives that actually affect each tile.
    """
    
    def __init__(self, canvas_size: Tuple[int, int], S: torch.Tensor, 
                 tile_size: int = 32, **kwargs):
        """
        Initialize the tile renderer.
        
        Args:
            canvas_size: Tuple of (height, width) for the output canvas
            S: Primitive shapes tensor
            tile_size: Size of each tile (default: 32)
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

    def _clamp_params_inplace(self, x, y, r):
        # VectorRenderer와 동일 정책: r ∈ [2, min(H,W)//4]
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
        
        # Initialize output canvas
        if self.use_fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        output = torch.zeros((self.H, self.W, 3), device=self.device, dtype=dtype)
        
        # Choose processing method based on tile count and device
        total_tiles = self.tiles_h * self.tiles_w
        use_parallel = total_tiles > 4 and torch.cuda.is_available()  # Parallel for larger tile counts
        
        if use_parallel:
            output = self._process_tiles_parallel(x, y, r, theta, v, c, sigma, I_bg, global_bmp_sel, output, lr_conf, is_final=is_final, return_alpha=return_alpha)
        else:
            output = self._process_tiles_sequential(x, y, r, theta, v, c, sigma, I_bg, global_bmp_sel, output, lr_conf, is_final=is_final)
        
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
                    x, y, r, theta, v, c, tile_primitive_indices,
                    x_start, x_end, y_start, y_end, sigma, I_bg,
                    global_bmp_sel=global_bmp_sel, is_final=is_final,
                    return_alpha=return_alpha
                )
                
                # Place result in output canvas
                output[y_start:y_end, x_start:x_end] = tile_result
        
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
                sigma, I_bg, global_bmp_sel=global_bmp_sel
            )
            
            # Place result in output canvas
            y_start, y_end = y_starts[tile_idx].item(), y_ends[tile_idx].item()
            x_start, x_end = x_starts[tile_idx].item(), x_ends[tile_idx].item()
            output[y_start:y_end, x_start:x_end] = tile_result
        
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


            if self.S_blurred is None:
                primitive_templates = self.S  # Use original templates for optimization

                # Use existing cuda_rasterizer or create new one if needed
                if self.cuda_rasterizer is None:
                    print(f"    🔧 Creating new TileRasterizer for optimization with {len(radii)} primitives...")
                    self.cuda_rasterizer = TileRasterizer(
                        self.H, self.W, self.tile_size, sigma, 
                        self.alpha_upper_bound, 300, len(radii),
                        use_fp16=self.use_fp16
                    )

            elif self.S_blurred is not None and not is_final:
                primitive_templates = self.S_blurred  # Use blurred templates for optimization
                
                # Use existing cuda_rasterizer or create new one if needed
                if self.cuda_rasterizer is None:
                    print(f"    🔧 Creating new TileRasterizer for optimization with {len(radii)} primitives...")
                    self.cuda_rasterizer = TileRasterizer(
                        self.H, self.W, self.tile_size, sigma, 
                        self.alpha_upper_bound, 300, len(radii),
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
                
                # Free existing cuda_rasterizer and create new one for final rendering
                if self.cuda_rasterizer is not None:
                    print(f"    🔧 Freeing existing CUDA rasterizer for final rendering...")
                    del self.cuda_rasterizer
                    self.cuda_rasterizer = None
                    torch.cuda.empty_cache()  # Clear GPU memory
                
                print(f"    🔧 Creating new TileRasterizer for final rendering with {len(radii)} primitives...")
                self.cuda_rasterizer = TileRasterizer(
                    self.H, self.W, self.tile_size, sigma, 
                    self.alpha_upper_bound, 300, len(radii),
                    use_fp16=self.use_fp16
                )


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
                # Convert lr_conf dict to tensor
                lr_config_tensor = torch.tensor([
                    lr_conf.get('default', 0.1),
                    lr_conf.get('gain_x', 1.0),
                    lr_conf.get('gain_y', 1.0),
                    lr_conf.get('gain_r', 1.0),
                    lr_conf.get('gain_v', 1.0),
                    lr_conf.get('gain_theta', 1.0),
                    lr_conf.get('gain_c', 1.0)
                ], dtype=torch.float16 if self.use_fp16 else torch.float32, device=means2D.device)
                
            # Call CUDA rasterizer for all tiles at once
            # Use FP16 version if available and use_fp16 is True
            # Use TileRasterizer class-based version (already created above)
            cuda_color, cuda_alpha = self.cuda_rasterizer(
                means2D, radii, rotations, opacities, colors,
                primitive_templates, global_bmp_sel,
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
    
    def _optimize_parameters_whole(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                  v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                  target_image: torch.Tensor, opt_conf: dict,
                                  target_binary_mask: Optional[torch.Tensor] = None,
                                  adjusted_pts: Optional[torch.Tensor] = None
                                  ):
        """
        Override optimization to use tile-based rendering instead of cached_masks.
        This is the core difference from VectorRenderer - we render directly from parameters.
        """
        from torch.amp import GradScaler, autocast
        from tqdm import tqdm
        import datetime
        import os
        
        is_no_bg_mode = target_image.shape[2] == 4
        
        # Get optimization parameters from config
        num_iterations = opt_conf.get("num_iterations", 100)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Mixed-precision scaler (only used if use_fp16 is True)
        scaler = GradScaler('cuda') if self.use_fp16 else None
        
        # Pre-calculate configurations
        blur_sigma = opt_conf.get("blur_sigma", 1.0)
        
        # Create output directory for saving images if it doesn't exist
        save_image_intervals = [1, 5, 10, 20, 50, 100]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
            {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
            {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0 / x.numel())},
            {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
            {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
        ])
        
        # Create scheduler if decay is enabled
        do_decay = opt_conf.get("do_decay", False)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=opt_conf.get("decay_rate", 0.99)) if do_decay else None
        
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
        
        # Optimization loop
        for iteration in tqdm(range(num_iterations), desc="Optimizing"):
            optimizer.zero_grad()

            # Pruning(Restart)
            if do_pruning and iteration !=0:
                prune_conf = opt_conf.get("pruning", {})
                prune_iterations = prune_conf.get("prune_iterations", 5)
                no_prune_last_iterations = prune_conf.get("no_prune_last_iterations", 10)
                no_prune_warmup_iterations = prune_conf.get("no_prune_warmup_iterations", 20)
                if iteration % prune_iterations == 0 and iteration < num_iterations - no_prune_last_iterations and iteration > no_prune_warmup_iterations:
                    print("[DEBUG] Pruning...")
                    self.do_prune(x,y,r,v,theta,c,prune_conf,adjusted_pts,target_binary_mask,target_image)      

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
                with autocast('cuda'):
                    if is_no_bg_mode:
                        rendered, rendered_alpha = self.render_from_params(
                            x, y, r, theta, v, c, sigma=current_sigma, I_bg=None, lr_conf=lr_conf, return_alpha=True
                        )
                    else:
                        white_bg = torch.ones((self.H, self.W, 3), device=self.device)
                        rendered = self.render_from_params(
                            x, y, r, theta, v, c, sigma=current_sigma, I_bg=white_bg, lr_conf=lr_conf
                        )
                        rendered_alpha = None
                    
                    # Compute loss
                    loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c, 
                            rendered_alpha = rendered_alpha,
                            loss_w_conf=opt_conf.get("loss_weights", None))
                    
                # Scale the loss and call backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()

            else:
                if is_no_bg_mode:
                    rendered, rendered_alpha = self.render_from_params(
                        x, y, r, theta, v, c, sigma=current_sigma, I_bg=None, lr_conf=lr_conf, return_alpha=True
                    )
                else:
                    white_bg = torch.ones((self.H, self.W, 3), device=self.device)
                    rendered = self.render_from_params(
                        x, y, r, theta, v, c, sigma=current_sigma, I_bg=white_bg, lr_conf=lr_conf
                    )
                    rendered_alpha = None
                
                # Compute loss
                loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c, 
                                         rendered_alpha = rendered_alpha,
                                         loss_w_conf=opt_conf.get("loss_weights", None))
                
                # Backward pass
                loss.backward()
                
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

            if DEBUG_MODE:
                print(f"    📊 Input data ranges: iteration {iteration}")
                print(f"      x: [{x.min():.4f}, {x.max():.4f}]")
                print(f"      y: [{y.min():.4f}, {y.max():.4f}]")
                print(f"      r: [{r.min():.4f}, {r.max():.4f}]")
                print(f"      v: [{v.min():.4f}, {v.max():.4f}]")
                print(f"      theta: [{theta.min():.4f}, {theta.max():.4f}]")
                print(f"      c: [{c.min():.4f}, {c.max():.4f}]")
            
            # Log progress
            if iteration % 10 == 0 or iteration in save_image_intervals or iteration < 3:
                print(f"Iteration {iteration}: Loss = {loss.item():.6f}")
                
                # Save intermediate images
                if iteration in save_image_intervals and DEBUG_MODE:
                    img_path = os.path.join(self.output_path, f"tile_iter_{iteration:04d}_{timestamp}.png")
                    self.save_image_tensor(rendered, img_path)
        
        print(f"Tile-based optimization completed. Final loss: {loss.item():.6f}")
        return x, y, r, v, theta, c

    def do_prune(self, x , y , r, v, theta, c, prune_conf, adjusted_pts, target_binary_mask, target_image) -> None:
        """
        Prune the low-opacity primitives and re-initialize them using adjusted_pts.
        
        Args:
            x, y, r, v, theta, c: Primitive parameters
            prune_conf: Pruning configuration
            adjusted_pts: Pre-computed adjusted points from initialization (numpy array of shape [N, 2])
            target_binary_mask: Binary mask for target regions
            target_image: Target image tensor for color sampling
        """
        prune_threshold = prune_conf.get("prune_threshold", 0.3)
        
        # Calculate opacity from visibility parameters
        opacity = torch.sigmoid(v)
        
        # Find primitives with low opacity
        low_opacity_mask = opacity < prune_threshold
        num_to_prune = low_opacity_mask.sum().item()
        
        if num_to_prune == 0:
            print("No primitives to prune.")
            return
            
        print(f"Pruning {num_to_prune} low-opacity primitives (threshold: {prune_threshold})")
        
        # Get indices of primitives to prune
        prune_indices = torch.where(low_opacity_mask)[0]
        
        # 1. Position initialization: Random sampling from adjusted_pts
        if adjusted_pts is not None and len(adjusted_pts) > 0:
            # Convert to numpy if tensor
            adjusted_pts_np = adjusted_pts.cpu().numpy() if isinstance(adjusted_pts, torch.Tensor) else adjusted_pts
            num_adjusted_pts = len(adjusted_pts_np)
            
            # Random sampling from adjusted_pts (with replacement if needed)
            sample_indices = np.random.choice(num_adjusted_pts, num_to_prune, replace=True)
            new_positions = adjusted_pts_np[sample_indices]
            print(f"Sampled {num_to_prune} positions from {num_adjusted_pts} adjusted points")
        else:
            # Fallback to random positions within canvas bounds
            new_positions = np.random.rand(num_to_prune, 2) * np.array([self.W, self.H])
            print("No adjusted_pts available, using random positions")
        
        # Re-initialize pruned primitives
        with torch.no_grad():
            # 1. Update positions (x, y)
            x.data[prune_indices] = torch.tensor(new_positions[:, 0], 
                                               dtype=x.dtype, device=x.device)
            y.data[prune_indices] = torch.tensor(new_positions[:, 1], 
                                               dtype=y.dtype, device=y.device)
            
            # 2. Visibility initialization (same as svgsplat_initializater.py)
            v.data[prune_indices] = torch.full((num_to_prune,), -2.0, device=v.device)
            
            # 3. Rotation initialization (same as svgsplat_initializater.py)
            theta.data[prune_indices] = torch.rand(num_to_prune, device=theta.device) * 2 * np.pi
            
            # 4. Color initialization (same as svgsplat_initializater.py)
            if target_image is not None:
                I_np = target_image.detach().cpu().numpy()
                
                if I_np.ndim == 3:
                    H, W, C = I_np.shape
                    # Handle 4-channel (RGBA) images by taking only RGB channels
                    if C == 4:
                        I_np = I_np[:, :, :3]  # Remove alpha channel
                    I_color = I_np  # (H, W, 3)
                else:
                    # Grayscale case
                    H, W = I_np.shape
                    I_color = np.stack([I_np] * 3, axis=2)  # Convert to RGB
                
                # Sample pixel colors at new positions (same as svgsplat_initializater.py)
                idx_x = np.clip(np.round(new_positions[:, 0]).astype(int), 0, W - 1)
                idx_y = np.clip(np.round(new_positions[:, 1]).astype(int), 0, H - 1)
                c_init = I_color[idx_y, idx_x]  # (num_to_prune, 3)
                
                # Add slight noise to diversify parameters (same as svgsplat_initializater.py)
                c_init += np.random.normal(0.0, 0.02, c_init.shape)
                c_init = np.clip(c_init, 0.0, 1.0)  # Safely clip values
                
                c.data[prune_indices] = torch.tensor(c_init, dtype=c.dtype, device=c.device)
            else:
                # Fallback: use diverse random colors
                new_colors = torch.rand(num_to_prune, 3, device=c.device) * 0.8 + 0.1
                c.data[prune_indices] = new_colors
            
            # 5. Radius initialization: distance-based only (distance_factor = 1.0)
            if target_image is not None:
                I_np = target_image.detach().cpu().numpy()
                if I_np.ndim == 3:
                    # Convert to grayscale for edge detection
                    I_gray = np.mean(I_np, axis=2)
                else:
                    I_gray = I_np.copy()
                
                # Ensure correct format for edge detection
                if I_gray.dtype != np.uint8:
                    I_gray = (I_gray * 255).astype(np.uint8)
                
                # Same radius calculation as svgsplat_initializater.py
                H, W = I_gray.shape
                max_radius = min(H, W) / 4
                
                # Get min_radius from base initializer (similar to svgsplat_initializater.py)
                # Assuming radii_min is available, otherwise use default
                min_radius = getattr(self, 'radii_min', 2)
                
                # Calculate Canny edges and distance transform (same as svgsplat_initializater.py)
                edges = cv2.Canny(I_gray, 100, 200)
                inverted_edges = cv2.bitwise_not(edges)
                distance_map = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
                
                # Sample distance at each new position
                idx_x = np.clip(np.round(new_positions[:, 0]).astype(int), 0, W - 1)
                idx_y = np.clip(np.round(new_positions[:, 1]).astype(int), 0, H - 1)
                point_distances = distance_map[idx_y, idx_x]
                
                # Normalize distances (0 = on edge, 1 = farthest from edge)
                max_distance = np.max(distance_map) if np.max(distance_map) > 0 else 1.0
                normalized_distances = point_distances / max_distance
                
                # Pure distance-based radius (distance_factor = 1.0)
                distance_based_radius = min_radius + normalized_distances * (max_radius - min_radius)
                
                # Add noise for variety (same as svgsplat_initializater.py)
                noise_scale = 0.01
                noise = np.random.normal(0, noise_scale * (max_radius - min_radius), num_to_prune)
                r_np = np.clip(distance_based_radius + noise, min_radius, max_radius)
                
                r.data[prune_indices] = torch.tensor(r_np, dtype=r.dtype, device=r.device)
            else:
                # Fallback: random radius between min and max
                min_radius = min(self.H, self.W) / 200
                max_radius = min(self.H, self.W) / 20
                new_radii = min_radius + torch.rand(num_to_prune, device=r.device) * (max_radius - min_radius)
                r.data[prune_indices] = new_radii
        
        print(f"Re-initialized {num_to_prune} primitives using svgsplat_initializater.py initialization method")
    
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
    
    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor,
                    rendered_alpha: torch.Tensor = None,
                    loss_w_conf = None,
                    epoch = None) -> torch.Tensor:
        """
        Compute MSE loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3) or (H, W, 4) if it has an alpha channel
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            rendered_alpha: Optional alpha channel tensor (H, W) if available
            
        Returns:
            MSE loss value
        """
        # If target has an alpha channel, use it as a mask for loss calculation
        # In this case, we only compute loss for pixels where alpha > 0 (foreground pixels)
        # and include both RGB and alpha channel in the loss calculation
        if target.shape[2] == 4:
            assert rendered_alpha is not None, "Rendered alpha channel must be provided when target has an alpha channel."
            assert loss_w_conf is not None, "Loss weights must be provided when target has an alpha channel."

            color_loss_weight = loss_w_conf.get('color_loss_weight', 1.0)
            alpha_loss_weight = loss_w_conf.get('alpha_loss_weight', 1.0)

            # Handle rendered_alpha shape: could be (H, W) or (H, W, 1)
            if rendered_alpha.dim() == 3 and rendered_alpha.shape[2] == 1:
                rendered_alpha = rendered_alpha.squeeze(-1)  # (H, W, 1) -> (H, W)

            # Extract alpha channel once and create mask
            target_alpha = target[:, :, 3]    # Shape: (H, W)
            alpha_mask = target_alpha > 0     # Shape: (H, W), boolean mask
            
            # Only compute loss for pixels where alpha > 0
            if alpha_mask.any():
                # Extract RGB channels
                target_rgb = target[:, :, :3]     # Shape: (H, W, 3)
                
                # Ensure consistent precision before masking to avoid unnecessary conversions
                if self.use_fp16:
                    if target_rgb.dtype == torch.float32:
                        rendered = rendered.float()
                        rendered_alpha = rendered_alpha.float()
                    elif rendered.dtype == torch.float16 and target_rgb.dtype != torch.float16:
                        target_rgb = target_rgb.half()
                        target_alpha = target_alpha.half()
                else:
                    rendered = rendered.float()
                    rendered_alpha = rendered_alpha.float()
                    target_rgb = target_rgb.float()
                    target_alpha = target_alpha.float()
                
                # Apply mask to all tensors
                rendered_masked = rendered[alpha_mask]              # Shape: (N_valid, 3)
                rendered_alpha_masked = rendered_alpha[alpha_mask]  # Shape: (N_valid,)
                target_rgb_masked = target_rgb[alpha_mask]          # Shape: (N_valid, 3)
                target_alpha_masked = target_alpha[alpha_mask]      # Shape: (N_valid,)
                
                # Combine RGB and alpha channels for single MSE computation
                # Concatenate along last dimension: RGB (3) + Alpha (1) = 4 channels
                rendered_combined = torch.cat([rendered_masked, rendered_alpha_masked.unsqueeze(-1)], dim=-1)  # (N_valid, 4)
                target_combined = torch.cat([target_rgb_masked, target_alpha_masked.unsqueeze(-1)], dim=-1)    # (N_valid, 4)
                
                color_loss = F.mse_loss(rendered_combined, target_combined)
            else:
                # If no valid pixels, return zero loss
                color_loss = torch.tensor(0.0, device=rendered.device, requires_grad=True)
            
            alpha_loss = F.mse_loss(rendered_alpha, target_alpha)

            print(f"color loss : {color_loss.item()}")
            print(f"alpha loss : {alpha_loss.item()}")

            return color_loss_weight*color_loss + alpha_loss_weight*alpha_loss

        else:
            # Original behavior for 3-channel targets
            # Ensure tensors are in consistent precision
            if self.use_fp16:
                if target.dtype == torch.float32:
                    rendered = rendered.float()
                elif rendered.dtype == torch.float16 and target.dtype != torch.float16:
                    target = target.half()
            else:
                rendered = rendered.float()
                target = target.float()


            return F.mse_loss(rendered, target)    
         
    def _render_tile(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                     theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                     primitive_indices: List[int], x_start: int, x_end: int,
                     y_start: int, y_end: int, sigma: float,
                     I_bg: torch.Tensor, global_bmp_sel: torch.Tensor = None, 
                     is_final: bool = False) -> torch.Tensor:
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
        
        # Create coordinate grid for this tile
        tile_X = self.X[:, y_start:y_end, x_start:x_end]  # (1, tile_h, tile_w)
        tile_Y = self.Y[:, y_start:y_end, x_start:x_end]  # (1, tile_h, tile_w)
        
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
                tile_masks = self._generate_tile_masks(
                    tile_x, tile_y, tile_r, tile_theta, tile_X, tile_Y, sigma,
                    global_primitive_indices=primitive_indices,
                    global_bmp_sel=global_bmp_sel, is_final=is_final
                )
                
                # Convert logits to actual values
                alpha = torch.sigmoid(tile_v) * self.alpha_upper_bound
                rgb = torch.sigmoid(tile_c)
                
                # Apply alpha to masks
                a = tile_masks * alpha.view(-1, 1, 1)
                
                # Create premultiplied colors
                m = a.unsqueeze(-1) * rgb.view(-1, 1, 1, 3)
                
                # Composite using parent's function
                comp_m, comp_a = self._transmit_over(m, a)
        else:
            # Generate masks for selected primitives in this tile
            tile_masks = self._generate_tile_masks(
                tile_x, tile_y, tile_r, tile_theta, tile_X, tile_Y, sigma,
                global_primitive_indices=primitive_indices,
                global_bmp_sel=global_bmp_sel, is_final=is_final
            )
            
            # Convert logits to actual values
            alpha = torch.sigmoid(tile_v) * self.alpha_upper_bound
            rgb = torch.sigmoid(tile_c)
            
            # Apply alpha to masks
            a = tile_masks * alpha.view(-1, 1, 1)
            
            # Create premultiplied colors
            m = a.unsqueeze(-1) * rgb.view(-1, 1, 1, 3)
            
            # Composite using parent's function
            comp_m, comp_a = self._transmit_over(m, a)
        
        # Handle background
        if I_bg is None:
            result = comp_m
        else:
            bg_tile = I_bg[y_start:y_end, x_start:x_end]
            result = comp_m + (1 - comp_a.unsqueeze(-1)) * bg_tile
        
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
        from torch.amp import autocast
        from util.utils import gaussian_blur
        
        num_primitives = x.shape[0]
        tile_h, tile_w = tile_X.shape[1], tile_X.shape[2]
        
        if num_primitives == 0:
            if self.use_fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32
            return torch.zeros((0, tile_h, tile_w), device=self.device, dtype=dtype)
        
        context = autocast('cuda') if self.use_fp16 else nullcontext()
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


    def render_export_mp4(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          r: torch.Tensor,
                          theta: torch.Tensor,
                          v: torch.Tensor,
                          c: torch.Tensor,
                          video_path: str,
                          fps: int = 60) -> None:
        """
        Export MP4 video showing progressive primitive addition.
        
        Args:
            x, y, r, theta, v, c: Primitive parameters
            video_path: Output video file path
            fps: Frames per second
        """
        import cv2
        import tempfile
        import subprocess
        import os
        import numpy as np
        
        N = len(x)
        print(f"Generating MP4 with {N} primitives at {self.W}x{self.H}...")
        
        # Direct MP4 output with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (self.W, self.H))
        
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        
        try:
            # Create white background frame
            white_bg = torch.ones((self.H, self.W, 3), device=x.device, dtype=torch.float32)
            
            # Write initial white frame
            self._write_frame(writer, white_bg)
            
            # Process primitives incrementally
            for frame_idx in range(N):
                if frame_idx % max(1, N // 20) == 0:
                    print(f"Processing frame {frame_idx+1}/{N}...")
                
                # Render frame with primitives 0 to frame_idx
                frame = self.render_from_params(
                    x[:frame_idx+1], y[:frame_idx+1], r[:frame_idx+1],
                    theta[:frame_idx+1], v[:frame_idx+1], c[:frame_idx+1],
                    sigma=0.0, I_bg=white_bg
                )
                
                self._write_frame(writer, frame)
                
                # Clear GPU memory periodically
                if frame_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            print(f"Video saved to {video_path}")
            
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

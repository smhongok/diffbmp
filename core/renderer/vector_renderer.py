from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch import nn
import torch.utils.checkpoint as cp
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
from util.utils import gaussian_blur, make_batch_indices
import os
import gc
import pkg_resources
from collections import defaultdict
import datetime
from PIL import Image
import tempfile
import subprocess
import glob
import wandb

class VectorRenderer:
    """
    A class for rendering vector graphics using differentiable primitives.
    This class handles the core rendering functionality including mask generation,
    alpha compositing, and parameter optimization.
    """
    def __init__(self, 
                 canvas_size: Tuple[int, int],
                 S: torch.Tensor,
                 alpha_upper_bound: float = 0.5,
                 device: str = 'cuda',
                 use_fp16: bool = False,
                 gamma: float = 1.0,
                 output_path: str = None,
                 tile_size: int = 32):
        """
        Initialize the vector renderer.
        
        Args:
            canvas_size: Tuple of (height, width) for the output canvas
            alpha_upper_bound: Maximum alpha value for rendering (default: 0.5)
            device: Device to use for computation ('cuda' or 'cpu')
            use_fp16: Whether to use half precision (FP16) for memory efficiency
        """
        self.H, self.W = canvas_size
        self.alpha_upper_bound = alpha_upper_bound
        self.device = device
        self.use_checkpointing = False
        self.use_fp16 = use_fp16
        self.gamma = gamma
        self.output_path = output_path
        
        # Tile rendering parameters
        self.tile_size = tile_size
        self.tiles_h = (self.H + tile_size - 1) // tile_size
        self.tiles_w = (self.W + tile_size - 1) // tile_size
        # Convert S to appropriate precision during initialization
        if self.use_fp16:
            self.S = S.to(dtype=torch.float16)
        else:
            self.S = S
        
        # Pre-compute pixel coordinates
        self.X, self.Y = self._create_coordinate_grid()
        
        # Compute bounding boxes for each primitive in self.S (for tile rendering)
        self.primitive_bboxes = self._compute_primitive_bboxes()
        
        # Initialize video creation tracking
        self.saved_frames = []
        
        # Initialize wandb conditionally
        wandb_mode = os.environ.get('WANDB_MODE', 'online')
        if wandb_mode.lower() != 'disabled':
            try:
                wandb.login(key="08d57452958261449694f652099a45203ab23a2e")
                wandb.init(entity="svgsplat",project="svgsplat", name=f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                self.wandb_enabled = True
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
                self.wandb_enabled = False
        else:
            print("wandb disabled via WANDB_MODE environment variable")
            self.wandb_enabled = False
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpointing = True
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_checkpointing = False
    
    def _create_coordinate_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create the coordinate grid for rendering."""
        if self.use_fp16:
            X, Y = torch.meshgrid(
                torch.arange(self.W, device=self.device, dtype=torch.float16),
                torch.arange(self.H, device=self.device, dtype=torch.float16),
                indexing='xy'
            )
        else:
            X, Y = torch.meshgrid(
                torch.arange(self.W, device=self.device),
                torch.arange(self.H, device=self.device),
                indexing='xy'
            )
        return X.unsqueeze(0), Y.unsqueeze(0)  # (1, H, W)
    
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
                           no_background: bool = False, sigma: float = 0.0) -> torch.Tensor:
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
            no_background: Whether to use no background
            sigma: Gaussian blur std
            
        Returns:
            Rendered image tensor
        """
        N = x.shape[0]
        
        # Pre-compute global primitive template selection (before tile processing)
        if self.S.dim() == 3:  # Multiple primitive templates [p, H, W]
            p = self.S.size(0)
            global_bmp_sel = torch.arange(N, device=self.device, dtype=torch.long) % p
            global_bmp_sel = global_bmp_sel.flip(0)  # Same as original VectorRenderer
        else:
            global_bmp_sel = None  # Single template case
        
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
            output = self._process_tiles_parallel(x, y, r, theta, v, c, sigma, I_bg, no_background, global_bmp_sel, output)
        else:
            output = self._process_tiles_sequential(x, y, r, theta, v, c, sigma, I_bg, no_background, global_bmp_sel, output)
                
        if return_alpha:
            # For simplicity, return dummy alpha for now
            alpha = torch.ones((self.H, self.W), device=self.device, dtype=dtype)
            return output, alpha
        
        return output
    
    def _process_tiles_sequential(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                             theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                             sigma: float, I_bg: torch.Tensor, no_background: bool,
                             global_bmp_sel: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Sequential tile processing using unified primitive assignment."""
        
        # Pre-compute all tile boundaries
        total_tiles = self.tiles_h * self.tiles_w
        tile_y_coords = torch.arange(self.tiles_h, device=self.device).repeat_interleave(self.tiles_w)
        tile_x_coords = torch.arange(self.tiles_w, device=self.device).repeat(self.tiles_h)
        
        y_starts = tile_y_coords * self.tile_size
        y_ends = torch.clamp(y_starts + self.tile_size, max=self.H)
        x_starts = tile_x_coords * self.tile_size
        x_ends = torch.clamp(x_starts + self.tile_size, max=self.W)
        
        # Unified vectorized primitive-to-tile assignment with accurate bounding boxes
        primitive_tile_masks = self._unified_primitive_assignment(
            x, y, r, theta, x_starts, x_ends, y_starts, y_ends
        )
        
        # Process tiles sequentially
        for tile_idx in range(total_tiles):
            # Get primitives affecting this tile
            tile_mask = primitive_tile_masks[tile_idx]
            if not tile_mask.any():
                continue
                
            tile_primitive_indices = torch.nonzero(tile_mask, as_tuple=True)[0].tolist()
            
            # Render this tile with selected primitives only
            tile_result = self._render_tile(
                x, y, r, theta, v, c, tile_primitive_indices,
                x_starts[tile_idx].item(), x_ends[tile_idx].item(),
                y_starts[tile_idx].item(), y_ends[tile_idx].item(),
                sigma, I_bg, no_background, global_bmp_sel=global_bmp_sel
            )
            
            # Place result in output canvas
            y_start, y_end = y_starts[tile_idx].item(), y_ends[tile_idx].item()
            x_start, x_end = x_starts[tile_idx].item(), x_ends[tile_idx].item()
            output[y_start:y_end, x_start:x_end] = tile_result
        
        return output
    
    def _process_tiles_parallel(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                               theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                               sigma: float, I_bg: torch.Tensor, no_background: bool,
                               global_bmp_sel: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """True vectorized tile processing using PyTorch operations."""
        
        # Pre-compute all tile boundaries
        total_tiles = self.tiles_h * self.tiles_w
        tile_y_coords = torch.arange(self.tiles_h, device=self.device).repeat_interleave(self.tiles_w)
        tile_x_coords = torch.arange(self.tiles_w, device=self.device).repeat(self.tiles_h)
        
        y_starts = tile_y_coords * self.tile_size
        y_ends = torch.clamp(y_starts + self.tile_size, max=self.H)
        x_starts = tile_x_coords * self.tile_size
        x_ends = torch.clamp(x_starts + self.tile_size, max=self.W)
        
        # Unified vectorized primitive-to-tile assignment with accurate bounding boxes
        primitive_tile_masks = self._unified_primitive_assignment(
            x, y, r, theta, x_starts, x_ends, y_starts, y_ends
        )
        
        # Process tiles with primitives
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
                sigma, I_bg, no_background, global_bmp_sel=global_bmp_sel
            )
            
            # Place result in output canvas
            y_start, y_end = y_starts[tile_idx].item(), y_ends[tile_idx].item()
            x_start, x_end = x_starts[tile_idx].item(), x_ends[tile_idx].item()
            output[y_start:y_end, x_start:x_end] = tile_result
        
        return output
    


    def _unified_primitive_assignment(self, x: torch.Tensor, y: torch.Tensor, 
                                    r: torch.Tensor, theta: torch.Tensor,
                                    x_starts: torch.Tensor, x_ends: torch.Tensor,
                                    y_starts: torch.Tensor, y_ends: torch.Tensor) -> torch.Tensor:
        """
        Unified vectorized computation of which primitives affect which tiles.
        Combines the speed of vectorized operations with the accuracy of precise bounding boxes.
        
        Args:
            x, y: (N,) primitive positions
            r: (N,) primitive scales  
            theta: (N,) primitive rotations
            x_starts, x_ends, y_starts, y_ends: (total_tiles,) tile boundaries
            
        Returns:
            torch.Tensor: (total_tiles, N) boolean mask where [i, j] indicates 
                         if primitive j affects tile i
        """
        N = len(x)
        total_tiles = len(x_starts)
        p = len(self.primitive_bboxes)
        
        # Pre-compute trigonometric values for all primitives
        cos_theta = torch.cos(theta)  # (N,)
        sin_theta = torch.sin(theta)  # (N,)
        
        # Get primitive indices (cycling through available primitives)
        prim_indices = torch.arange(N, device=self.device) % p  # (N,)
        
        # Extract bounding boxes for all primitives
        # Convert list of tuples to tensor for vectorized operations
        bbox_tensor = torch.tensor(self.primitive_bboxes, device=self.device)  # (p, 4)
        selected_bboxes = bbox_tensor[prim_indices]  # (N, 4)
        
        min_u = selected_bboxes[:, 0]  # (N,)
        max_u = selected_bboxes[:, 1]  # (N,)
        min_v = selected_bboxes[:, 2]  # (N,)
        max_v = selected_bboxes[:, 3]  # (N,)
        
        # Create all corner combinations for bounding boxes
        corners_u = torch.stack([min_u, max_u, min_u, max_u], dim=1)  # (N, 4)
        corners_v = torch.stack([min_v, min_v, max_v, max_v], dim=1)  # (N, 4)
        
        # Transform all corners to world coordinates (vectorized)
        # Broadcasting: (N, 1) * (N, 4) -> (N, 4)
        world_x = x.unsqueeze(1) + r.unsqueeze(1) * (
            corners_u * cos_theta.unsqueeze(1) - corners_v * sin_theta.unsqueeze(1)
        )  # (N, 4)
        world_y = y.unsqueeze(1) + r.unsqueeze(1) * (
            corners_u * sin_theta.unsqueeze(1) + corners_v * cos_theta.unsqueeze(1)
        )  # (N, 4)
        
        # Compute bounding box extents for each primitive
        bbox_x_min = world_x.min(dim=1)[0]  # (N,)
        bbox_x_max = world_x.max(dim=1)[0]  # (N,)
        bbox_y_min = world_y.min(dim=1)[0]  # (N,)
        bbox_y_max = world_y.max(dim=1)[0]  # (N,)
        
        # Expand dimensions for broadcasting: (N, 1) and (1, total_tiles)
        bbox_x_min_exp = bbox_x_min.unsqueeze(1)  # (N, 1)
        bbox_x_max_exp = bbox_x_max.unsqueeze(1)  # (N, 1)
        bbox_y_min_exp = bbox_y_min.unsqueeze(1)  # (N, 1)
        bbox_y_max_exp = bbox_y_max.unsqueeze(1)  # (N, 1)
        
        x_starts_exp = x_starts.unsqueeze(0)  # (1, total_tiles)
        x_ends_exp = x_ends.unsqueeze(0)      # (1, total_tiles)
        y_starts_exp = y_starts.unsqueeze(0)  # (1, total_tiles)
        y_ends_exp = y_ends.unsqueeze(0)      # (1, total_tiles)
        
        # Vectorized intersection test for all primitive-tile pairs
        # Intersection condition: not (bbox_max < tile_min or bbox_min > tile_max)
        x_intersect = ~((bbox_x_max_exp < x_starts_exp) | (bbox_x_min_exp > x_ends_exp))
        y_intersect = ~((bbox_y_max_exp < y_starts_exp) | (bbox_y_min_exp > y_ends_exp))
        
        # Both x and y must intersect
        intersections = x_intersect & y_intersect  # (N, total_tiles)
        
        # Transpose to get (total_tiles, N) - each row is primitives affecting that tile
        return intersections.t()
    

    
    def _render_tile(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                     theta: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                     primitive_indices: List[int], x_start: int, x_end: int,
                     y_start: int, y_end: int, sigma: float,
                     I_bg: torch.Tensor, no_background: bool, 
                     global_bmp_sel: torch.Tensor = None) -> torch.Tensor:
        """
        Render a single tile with only the selected primitives.
        
        Args:
            x, y, r, theta: Full primitive parameters
            v, c: Full visibility and color parameters
            primitive_indices: Indices of primitives that affect this tile
            x_start, x_end, y_start, y_end: Tile boundaries
            sigma: Gaussian blur std
            I_bg: Background image
            no_background: Whether to use no background
            
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
        
        # Generate masks for selected primitives in this tile
        tile_masks = self._generate_tile_masks(
            tile_x, tile_y, tile_r, tile_theta, tile_X, tile_Y, sigma,
            global_primitive_indices=primitive_indices,
            global_bmp_sel=global_bmp_sel
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
        if no_background:
            result = comp_m
        else:
            if I_bg is not None:
                bg_tile = I_bg[y_start:y_end, x_start:x_end]
            else:
                bg_tile = torch.ones((tile_h, tile_w, 3), device=self.device, 
                                   dtype=comp_m.dtype)
            
            result = comp_m + (1 - comp_a.unsqueeze(-1)) * bg_tile
        
        return result
    
    def _generate_tile_masks(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                            theta: torch.Tensor, tile_X: torch.Tensor, tile_Y: torch.Tensor,
                            sigma: float, global_primitive_indices: List[int] = None,
                            global_bmp_sel: torch.Tensor = None) -> torch.Tensor:
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
    
    def _batched_soft_rasterize(self,
                               x: torch.Tensor,
                               y: torch.Tensor,
                               r: torch.Tensor,
                               theta: torch.Tensor,
                               sigma: float = 0.0) -> torch.Tensor:
        """
        Generate soft masks for each primitive, processing in smaller chunks to save memory.

        Now supports a sequence of p different primitives in self.S of shape (p, H, W),
        assigning them periodically to the B instances.

        Args:
            x, y:       [N] position coordinates for N shapes
            r:          [N] scales
            theta:      [N] rotations
            sigma:      Gaussian blur std
        Returns:
            masks:     [N, H, W]  (one soft mask per shape)
        """
        context = autocast('cuda') if self.use_fp16 else nullcontext()
        with context:
            B = len(x)
            _, H, W = self.X.shape
            target_dtype = torch.float16 if self.use_fp16 else torch.float32

            # Prepare the bitmap(s): either single template or a sequence of p templates
            if sigma > 0.0:
                bmp = self.S.unsqueeze(0)       # -> [1, p, H, W] or [1, H, W]
                bmp = gaussian_blur(bmp, sigma)
                bmp_image = bmp.squeeze(0).to(dtype=target_dtype).contiguous()
            else:
                bmp_image = self.S.to(dtype=target_dtype)

            # Expand coordinates and parameters
            X_exp = self.X.expand(B, H, W)
            Y_exp = self.Y.expand(B, H, W)
            x_exp = x.view(B, 1, 1).expand(B, H, W)
            y_exp = y.view(B, 1, 1).expand(B, H, W)
            r_exp = r.view(B, 1, 1).expand(B, H, W)

            # Normalize and rotate positions
            pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            R_inv = torch.zeros(B, 2, 2, device=self.device)
            R_inv[:, 0, 0] = cos_t; R_inv[:, 0, 1] = sin_t
            R_inv[:, 1, 0] = -sin_t; R_inv[:, 1, 1] = cos_t
            uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
            grid = uv.permute(0, 2, 3, 1)  # (B, H, W, 2)

            # Build bmp_exp: one bitmap per instance, cycling through p if provided
            if bmp_image.dim() == 2:
                # single primitive template [H, W]
                bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
            elif bmp_image.dim() == 3:
                # sequence of p templates [p, H, W]
                p = bmp_image.size(0)
                # periodic assignment
                idx = torch.arange(B, device=self.device, dtype=torch.long) % p  # [B]
                idx = idx.flip(0)
                bmp_sel = bmp_image[idx, :, :]            # [B, H, W]
                bmp_exp = bmp_sel.unsqueeze(1).contiguous()  # [B, 1, H, W]
            else:
                raise ValueError(f"Unsupported self.S shape: {bmp_image.shape}")

            # Sample masks via grid_sample (with optional checkpointing)
            if self.use_checkpointing:
                def grid_fn(img, g):
                    return F.grid_sample(img, g, mode='bilinear', padding_mode='zeros', align_corners=True)
                sampled = cp.checkpoint(grid_fn, bmp_exp, grid, use_reentrant=False)
            else:
                sampled = F.grid_sample(bmp_exp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            # Return single-channel masks
            return sampled.squeeze(1)  # (B, H, W)
    
    def _transmit_over(self, m: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        
        """
        Vectorized weighted-sum alpha compositing (Porter–Duff “Over”).
        Inputs:
        m: (N, H, W, 3) premultiplied colors
        a: (N, H, W) alphas
        Outputs:
        comp_m: (H, W, 3) composited RGB before blending white background
        comp_a: (H, W)   composited alpha
        """
        # 1. Compute transmittance T[k] = prod_{i<k}(1 - a[i]) for k=0..N
        #    T has shape (N+1, H, W), with T[0] = 1
        alpha_inv = 1.0 - a                               # (N, H, W)
        T_partial = torch.cumprod(alpha_inv, dim=0)       # (N, H, W)
        ones = torch.ones_like(T_partial[0:1])            # (1, H, W)
        T = torch.cat([ones, T_partial], dim=0)           # (N+1, H, W)

        # 2. Composite alpha: A_out = 1 - T[N]
        comp_a = 1.0 - T[-1]                               # (H, W)

        # 3. Composite color before blending white background:
        #    C_out = sum_{k=0..N-1} (m[k] * T[k]) + T[N]*white
        #    Here m[k] already = a[k] * c[k], so weight = T[k]
        #    Weighted sum is sum_{k=0..N-1} (m[k] * T[k])
        weights = T[:-1].unsqueeze(-1)                     # (N, H, W, 1)
        weighted = m * weights                            # (N, H, W, 3)
        comp_m = weighted.sum(dim=0)                     # (H, W, 3)

        return comp_m, comp_a

        # as a lab intern who edited this code without deep knowledge about computing, 
        # carfully suggest this method because
        # 1. Similar PSNR, increased SSIM, VIF, and lowered LPIPS compared to the original method
        # 2. significant decrease of total cost time


    
    def _get_checkpoint_kwargs(self):
        """
        Returns the correct checkpoint keyword arguments based on the PyTorch version.
        Older versions don't support use_reentrant.
        """
        # Check PyTorch version
        torch_version = pkg_resources.get_distribution("torch").version
        major, minor = map(int, torch_version.split('.')[:2])
        
        # use_reentrant supported only in PyTorch 1.12 and later
        if (major > 1) or (major == 1 and minor >= 12):
            return {"use_reentrant": False}
        else:
            # No such option in earlier versions
            return {}
    
    def _safe_checkpoint(self, func, *tensors):
        """
        Wrapper function to safely perform checkpointing regardless of PyTorch version
        """
        kwargs = self._get_checkpoint_kwargs()
        return torch.utils.checkpoint.checkpoint(func, *tensors, **kwargs)
    
    def render(
            self,
            cached_masks: torch.Tensor,
            v: torch.Tensor,
            c: torch.Tensor,
            return_alpha: bool = False,
            I_bg: Optional[torch.Tensor] = None,
            no_background: bool = False
        ):
        """
        Render the final image (optionally alpha).

        Args:
            cached_masks : (N, H, W)   – pre-computed soft masks
            v            : (N,)        – visibility logits
            c            : (N, 3)      – RGB logits
            return_alpha : If True  → (rgb, alpha) returned
                        If False → rgb only returned
            I_bg         : Optional background tensor (H, W, 3). If None, a white background will be used
                          unless no_background=True.
            no_background: If True, no background compositing is performed (transparent output).

        Returns
        -------
        - rgb  : (H, W, 3)  (always)
        - alpha: (H, W, 1)  (optional, when return_alpha=True)
        """
        context = autocast('cuda') if self.use_fp16 else nullcontext()
        
        with context:    
            target_dtype = torch.float16 if self.use_fp16 else torch.float32
            cached_masks = cached_masks
            v = v
            c = c
            N = v.shape[0]

            # 1. per-primitive alpha & color
            v_alpha = self.alpha_upper_bound * torch.sigmoid(v).view(N, 1, 1)
            a = v_alpha * cached_masks                     # (N, H, W)
            c_eff = torch.sigmoid(c).view(N, 1, 1, 3)      # (N, 1, 1, 3)
            
            # Create color tensor with minimal memory overhead
            m = a.unsqueeze(-1) * c_eff                    # (N, H, W, 3)

            # 2. Porter–Duff reduction (tree)
            if self.use_checkpointing:
                comp_m, comp_a = self._safe_checkpoint(
                    lambda mm, aa: self._transmit_over(mm, aa),
                    m, a
                )
            else:
                comp_m, comp_a = self._transmit_over(m, a)    
            
            # Free large tensors as soon as possible if in FP16 mode
            if self.use_fp16:
                del m, a, v_alpha, c_eff
            
            if return_alpha:
                # (H, W) → (H, W, 1) to match for broadcasting
                return comp_m, comp_a.unsqueeze(-1)

            # 3. Composite with background
            if no_background:
                # No background compositing - return transparent result
                final = comp_m
            else:
                # Use provided background or default white background
                if I_bg is None:
                    I_bg = torch.ones_like(comp_m)
                final = comp_m + (1.0 - comp_a).unsqueeze(-1) * I_bg
            
            # Free temporary tensors if in FP16 mode
            if self.use_fp16:
                del comp_m, comp_a
                if I_bg is not None:
                    del I_bg
            
            return final
    
    def render_export_mp4(
        self,
        cached_masks: torch.Tensor,  # (N, H, W)
        v: torch.Tensor,             # (N,)
        c: torch.Tensor,             # (N, 3)
        video_path: str,
        fps: int = 60
    ):
        """
        Sequential-over compositing 으로 primitive를 한 장씩 쌓아가며
        중간 이미지를 MP4로 바로 기록.

        Args:
        cached_masks: (N, H, W)   – soft masks
        v           : (N,)        – visibility logits
        c           : (N, 3)      – RGB logits
        video_path  : 저장될 MP4 경로
        fps         : 프레임레이트
        """
        context = autocast('cuda') if self.use_fp16 else nullcontext()

        with context:
            # 1.1 per-primitive alpha & color
            v_alpha = self.alpha_upper_bound * torch.sigmoid(v).view(-1, 1, 1)    # (N,1,1)
            a_all   = v_alpha * cached_masks                                       # (N,H,W)
            c_eff   = torch.sigmoid(c).view(-1, 1, 1, 3)                           # (N,1,1,3)
            m_all   = a_all.unsqueeze(-1) * c_eff                                  # (N,H,W,3)

        # --- 2. 비디오 초기화를 위한 첫 프레임 계산 ---
        N, H, W = a_all.shape
        
        # 임시 파일 생성 (OpenCV 중간 결과용)
        temp_video = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
                temp_video = tmp.name

            # sequential over: comp_m, comp_a 초기값 (맨 아래층 = 첫 프리미티브)
            comp_m = m_all[0]   # (H,W,3)
            comp_a = a_all[0]   # (H,W)
            # 흰 배경과 합성
            first = comp_m + (1.0 - comp_a).unsqueeze(-1)
            first_np = (first.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
            # OpenCV는 BGR 순서
            first_bgr = first_np[..., ::-1]

            # 임시 비디오 작성 (무손실 코덱 사용)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 무손실 중간 포맷
            writer = cv2.VideoWriter(temp_video, fourcc, fps, (W, H))

            if not writer.isOpened():
                raise RuntimeError("Failed to open video writer")

            # 첫 프레임 기록
            writer.write(first_bgr)

            # --- 3. 나머지 프리미티브 순차 합성 & 기록 ---
            for i in range(1, N):
                with context:
                    # comp = comp + (1 - comp_a) * next
                    comp_m = comp_m + (1.0 - comp_a).unsqueeze(-1) * m_all[i]
                    comp_a = comp_a + (1.0 - comp_a) * a_all[i]

                # 흰 배경과 합성
                frame = comp_m + (1.0 - comp_a).unsqueeze(-1)
                frame_np = (frame.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
                frame_bgr = frame_np[..., ::-1]
                writer.write(frame_bgr)

            writer.release()

            # --- 4. FFmpeg으로 웹 호환 H.264 MP4 변환 ---
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # 덮어쓰기 허용
                '-i', temp_video,
                
                # 비디오 코덱 설정 (웹 호환)
                '-c:v', 'libx264',
                '-profile:v', 'baseline',  # 최대 브라우저 호환성
                '-level', '3.1',
                '-crf', '23',              # 품질 (18-28, 낮을수록 고품질)
                
                # 픽셀 포맷 (필수)
                '-pix_fmt', 'yuv420p',
                
                # 웹 스트리밍 최적화
                '-movflags', '+faststart',
                
                # 오디오 없음 (비디오만)
                '-an',
                
                video_path
            ]
            
            # FFmpeg 실행
            result = subprocess.run(
                ffmpeg_cmd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            if result.returncode != 0:
                print(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
            
            print(f"Saved web-compatible H.264 MP4 to {video_path}")
            
        finally:
            # 임시 파일 정리
            if temp_video and os.path.exists(temp_video):
                try:
                    os.unlink(temp_video)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_video}: {e}")

        return frame  # Return the last frame for reference
    
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
                            target_image: torch.Tensor,
                            target_binary_mask=None) -> Tuple[torch.Tensor, ...]:
        """
        Initialize parameters using the provided initializer.
        
        Args:
            initializer: Initializer object (e.g., StructureAwareInitializer)
            target_image: Target image to match
            
        Returns:
            Tuple of initialized parameters (x, y, r, v, theta, c)
        """
        # We have to modify this method for better performance with none background images,
        # since we don't have to splat primitives on the background(which is transparent) at initialization.        
        if target_image.shape[2] == 4:
            target_image = target_image[:, :, :3]  # Ignore alpha channel

        # Initialize from target image
        x, y, r, v, theta, c = initializer.initialize(target_image, target_binary_mask = target_binary_mask)
        
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
                      opt_conf: Dict[str, Any],
                      target_binary_mask: Optional[torch.Tensor] = None,
                      target_dist_mask: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, ...]:
        """Optimize rendering parameters to match target image."""
        N = x.shape[0]
        self.sample_size = max(1, int(0.2 * N))  # 20% of primitives
        # Randomly sample indices
        self.sample_indices = torch.randperm(N, device=self.device)[:self.sample_size]

        # Use whole optimization (batch optimization removed as it's never used)
        return self._optimize_parameters_whole(
            x, y, r, v, theta, c, target_image, opt_conf, target_binary_mask, target_dist_mask
        )

    def _optimize_parameters_whole(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                  v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                  target_image: torch.Tensor, opt_conf: dict,
                                  target_binary_mask: torch.Tensor = None,
                                  target_dist_mask: torch.Tensor = None):
        """
        Override optimization to use tile-based rendering instead of cached_masks.
        This is the core difference from VectorRenderer - we render directly from parameters.
        """
        from torch.amp import GradScaler, autocast
        from tqdm import tqdm
        import datetime
        import os
        
        is_no_bg_mode = target_image.shape[2] == 4 and target_binary_mask is not None
        
        # Get optimization parameters from config
        num_iterations = opt_conf.get("num_iterations", 100)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Mixed-precision scaler (only used if use_fp16 is True)
        scaler = GradScaler('cuda') if self.use_fp16 else None
        
        # Pre-calculate configurations
        blur_sigma = opt_conf.get("blur_sigma", 1.0)
        
        # Create output directory for saving images if it doesn't exist
        save_image_intervals = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
            {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
            {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},
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
        
        print(f"Starting tile-based optimization, {num_iterations} iterations...")
        
        # Optimization loop
        for iteration in tqdm(range(num_iterations), desc="Optimizing"):
            optimizer.zero_grad()
            
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
                        rendered = self.render_from_params(
                            x, y, r, theta, v, c, sigma=current_sigma, no_background=True
                        )
                    else:
                        white_bg = torch.ones((self.H, self.W, 3), device=self.device)
                        rendered = self.render_from_params(
                            x, y, r, theta, v, c, sigma=current_sigma, I_bg=white_bg
                        )
                    
                    # Compute loss
                    loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Clamp r to prevent NaN in grid_sample (r must be positive)
                with torch.no_grad():
                    r.clamp_(min=2.0, max=64.0)  # min=2, max=64
               
            else:
                if is_no_bg_mode:
                    rendered = self.render_from_params(
                        x, y, r, theta, v, c, sigma=current_sigma, no_background=True
                    )
                else:
                    white_bg = torch.ones((self.H, self.W, 3), device=self.device)
                    rendered = self.render_from_params(
                        x, y, r, theta, v, c, sigma=current_sigma, I_bg=white_bg
                    )
                
                # Compute loss
                loss = self.compute_loss(rendered, target_image, x, y, r, v, theta, c)
                
                # Backward pass
                loss.backward()
                optimizer.step()

                # Clamp r to prevent NaN in grid_sample (r must be positive)
                with torch.no_grad():
                    r.clamp_(min=2.0, max=64.0)  # min=2, max=64
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Log progress
            if iteration % 10 == 0 or iteration in save_image_intervals:
                print(f"Iteration {iteration}: Loss = {loss.item():.6f}")
                
                # Save intermediate images
                if iteration in save_image_intervals:
                    img_path = os.path.join(self.output_path, f"tile_iter_{iteration:04d}_{timestamp}.png")
                    self.save_image_tensor(rendered, img_path)
        
        print(f"Tile-based optimization completed. Final loss: {loss.item():.6f}")
        return x, y, r, v, theta, c

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
        white_bg = torch.ones((self.H, self.W, 3), device=cached_masks.device)
        final_render = self.render(cached_masks, v, c, I_bg=white_bg)
        final_render_np = final_render.detach().cpu().numpy()
        final_render_np = (final_render_np * 255).astype(np.uint8)
        
        # Save the image using PIL
        Image.fromarray(final_render_np).save(output_path)
       
    def render_export_mp4_hires(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              r: torch.Tensor,
                              theta: torch.Tensor,
                              v: torch.Tensor,
                              c: torch.Tensor,
                              video_path: str,
                              scale_factor: float = 4.0,
                              fps: int = 60) -> None:
        """
        Export high-resolution MP4 video using streaming approach to avoid VRAM overflow.
        Shows progressive primitive addition at high resolution.
        
        Args:
            x, y, r, theta, v, c: Primitive parameters
            video_path: Path to save the MP4 video
            scale_factor: Resolution multiplier (e.g., 4.0 = 4x resolution)
            chunk_size: Number of primitives to process at once (smaller = less VRAM)
            fps: Frames per second for the video
        """
        import cv2
        
        # Store original resolution
        orig_H, orig_W = self.H, self.W
        
        # Temporarily scale up resolution
        self.H = int(orig_H * scale_factor)
        self.W = int(orig_W * scale_factor)
        
        # Scale up coordinates and sizes
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        r_scaled = r * scale_factor
        
        # Recreate coordinate grid for new resolution
        old_X, old_Y = self.X, self.Y
        self.X, self.Y = self._create_coordinate_grid()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (self.W, self.H))
        
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        
        try:
            N = len(x_scaled)
            print(f"Generating high-resolution MP4 with {N} primitives at {self.W}x{self.H}...")
            
            # Initialize with white background
            current_frame = torch.ones((self.H, self.W, 3), device=x.device, dtype=torch.float32)
            
            # Write first frame (white background)
            frame_np = current_frame.detach().cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            
            # Process primitives incrementally (efficient approach)
            for frame_idx in range(N):
                if frame_idx % max(1, N // 20) == 0:  # Progress updates
                    print(f"Processing frame {frame_idx}/{N}...")
                
                # Get current primitive parameters (single primitive)
                curr_x = x_scaled[frame_idx:frame_idx + 1]
                curr_y = y_scaled[frame_idx:frame_idx + 1]
                curr_r = r_scaled[frame_idx:frame_idx + 1]
                curr_theta = theta[frame_idx:frame_idx + 1]
                curr_v = v[frame_idx:frame_idx + 1]
                curr_c = c[frame_idx:frame_idx + 1]
                
                # Render only the current primitive
                with torch.no_grad():
                    # Generate mask for current primitive
                    mask = self._batched_soft_rasterize(
                        curr_x, curr_y, curr_r, curr_theta, sigma=0.0
                    ).squeeze(0)  # Remove batch dimension -> [H, W]
                    
                    # Calculate alpha and color for current primitive
                    alpha = (self.alpha_upper_bound * torch.sigmoid(curr_v)).item()
                    color = torch.sigmoid(curr_c).squeeze(0)  # [3]
                    
                    # Apply alpha to mask
                    alpha_mask = alpha * mask  # [H, W]
                    
                    # Create colored primitive: alpha_mask * color
                    colored_primitive = alpha_mask.unsqueeze(-1) * color.unsqueeze(0).unsqueeze(0)  # [H, W, 3]
                    
                    # Alpha composite: new_frame = colored_primitive + (1 - alpha_mask) * current_frame
                    inv_alpha = 1.0 - alpha_mask.unsqueeze(-1)  # [H, W, 1]
                    current_frame = colored_primitive + inv_alpha * current_frame
                
                # Convert to numpy and write frame
                frame_np = current_frame.detach().cpu().numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                
                # Clear GPU memory periodically
                if frame_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            print(f"Saved high-resolution MP4 ({self.W}x{self.H}) to {video_path}")
            
        finally:
            # Clean up
            writer.release()
            
            # Restore original resolution and coordinate grid
            self.H, self.W = orig_H, orig_W
            self.X, self.Y = old_X, old_Y
            
            # Clear GPU memory
            torch.cuda.empty_cache()

    def create_video_from_collected_frames(self, output_video_path: str, fps: int = 30) -> str:
        """
        Create an MP4 video from frames collected during optimization.
        
        Args:
            output_video_path: Path for the output MP4 file
            fps: Frames per second for the video
            
        Returns:
            Path to the created video file
        """
        if not self.saved_frames:
            print("No frames collected during optimization.")
            return None
        
        print(f"Creating video from {len(self.saved_frames)} collected frames...")
        
        # Read the first frame to get dimensions
        first_frame = cv2.imread(self.saved_frames[0])
        if first_frame is None:
            print(f"Could not read first frame: {self.saved_frames[0]}")
            return None
        
        height, width, layers = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Could not create video writer for: {output_video_path}")
            return None
        
        # Add frames to video
        for i, frame_path in enumerate(tqdm(self.saved_frames, desc="Creating video")):
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_path}")
        
        # Release video writer
        video_writer.release()
        
        print(f"Video created successfully: {output_video_path}")
        print(f"Video contains {len(self.saved_frames)} frames at {fps} FPS")
        
        return output_video_path 
        
    def render_image(self, x, y, r, v, theta, c):
        """
        Render an image from primitive parameters.
        Args:
            x, y, r, v, theta, c: Primitive parameters (torch.Tensor)
        Returns:
            Rendered image (H, W, 3)
        """
        cached_masks = self._batched_soft_rasterize(x, y, r, theta)
        return self.render(cached_masks, v, c, no_background=True)
    
    def over(self, I_hat, I_bg):
        """
        Alpha blending (Porter-Duff 'over') between I_hat and I_bg.
        I_hat: (H, W, 3) foreground
        I_bg:  (H, W, 3) background
        Returns:
            Blended image (H, W, 3)
        """
        # If alpha channel exists, use it, otherwise assume 1
        if I_hat.shape[-1] == 4:
            alpha = I_hat[..., 3:4]
            rgb = I_hat[..., :3]
        else:
            alpha = torch.ones_like(I_hat[..., :1])
            rgb = I_hat
        return rgb * alpha + I_bg * (1 - alpha)
    
    def memory_report(self, message="Memory usage"):
        """
        Detailed memory usage report.
        """
        torch.cuda.synchronize()
        
        current = torch.cuda.memory_allocated() / 1024 / 1024
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        print(f"{message}: Current={current:.2f}MB, Peak={peak:.2f}MB, Reserved={reserved:.2f}MB")
        
        # Print top tensors by memory usage
        tensors_report = defaultdict(int)
        sizes_report = {}
        precision_report = {}
        
        # Check tensors in self
        for name, obj in vars(self).items():
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                size_bytes = obj.nelement() * obj.element_size()
                tensors_report[name] = size_bytes
                sizes_report[name] = obj.shape
                precision_report[name] = obj.dtype
        
        # Print sorted tensors by memory usage (largest first)
        if tensors_report:
            print("  Top tensors by memory usage:")
            sorted_tensors = sorted(tensors_report.items(), key=lambda x: x[1], reverse=True)
            for name, size in sorted_tensors[:5]:  # Print top 5 only
                print(f"    {name}: {size/1024/1024:.2f}MB, shape={sizes_report[name]}, dtype={precision_report[name]}")
        
        # Compare memory usage between FP16 and FP32
        fp16_count = sum(1 for dtype in precision_report.values() if dtype == torch.float16)
        fp32_count = sum(1 for dtype in precision_report.values() if dtype == torch.float32)
        fp16_bytes = sum(size for name, size in tensors_report.items() if precision_report.get(name) == torch.float16)
        fp32_bytes = sum(size for name, size in tensors_report.items() if precision_report.get(name) == torch.float32)
        
        if fp16_count + fp32_count > 0:
            print(f"  FP16: {fp16_count} tensors, {fp16_bytes/1024/1024:.2f}MB")
            print(f"  FP32: {fp32_count} tensors, {fp32_bytes/1024/1024:.2f}MB")
        
        return current, peak, reserved

    def log_gradient_statistics(self, x, y, r, v, theta, c, epoch):
        """
        Log gradient statistics for 20% of primitives and 0th primitive to wandb.
        
        Args:
            x, y, r, v, theta, c: Parameter tensors
            epoch: Current epoch number
        """
        
        # Collect gradients for sampled primitives
        gradients = {}
        params = {'x': x, 'y': y, 'r': r, 'v': v, 'theta': theta, 'c': c}
        
        # for param_name, param in params.items():
        #     if param.grad is not None:
        #         # Get gradients for sampled indices
        #         if param_name == 'c':
        #             # c has shape (N, 3), so we need to handle it differently
        #             grad_sample = param.grad[self.sample_indices]  # (sample_size, 3)
        #             gradients[f'{param_name}_grad_mean'] = grad_sample.mean().item()
        #             gradients[f'{param_name}_grad_std'] = grad_sample.std().item()
        #             gradients[f'{param_name}_grad_norm'] = grad_sample.norm().item()
                    
        #             # Log individual RGB channel gradients
        #             for i, color in enumerate(['r', 'g', 'b']):
        #                 gradients[f'{param_name}_{color}_grad_mean'] = grad_sample[:, i].mean().item()
        #                 gradients[f'{param_name}_{color}_grad_std'] = grad_sample[:, i].std().item()
        #         else:
        #             # Other parameters have shape (N,)
        #             grad_sample = param.grad[self.sample_indices]
        #             gradients[f'{param_name}_grad_mean'] = grad_sample.mean().item()
        #             gradients[f'{param_name}_grad_std'] = grad_sample.std().item()
        #             gradients[f'{param_name}_grad_norm'] = grad_sample.norm().item()
        #     else:
        #         # No gradients available
        #         gradients[f'{param_name}_grad_mean'] = 0.0
        #         gradients[f'{param_name}_grad_std'] = 0.0
        #         gradients[f'{param_name}_grad_norm'] = 0.0
        #         if param_name == 'c':
        #             for color in ['r', 'g', 'b']:
        #                 gradients[f'{param_name}_{color}_grad_mean'] = 0.0
        #                 gradients[f'{param_name}_{color}_grad_std'] = 0.0
        
        # Log 0th primitive gradients and values
        for param_name, param in params.items():
            if param.grad is not None:
                if param_name == 'c':
                    # c has shape (N, 3)
                    grad_0th = param.grad[0]  # (3,)
                    param_0th = param[0]      # (3,)
                    
                    # Log 0th primitive gradient values
                    gradients[f'{param_name}_0th_r'] = grad_0th[0].item()
                    gradients[f'{param_name}_0th_g'] = grad_0th[1].item()
                    gradients[f'{param_name}_0th_b'] = grad_0th[2].item()
                else:
                    # Other parameters have shape (N,)
                    grad_0th = param.grad[0]
                    param_0th = param[0]
                    
                    # Log 0th primitive gradient and value
                    gradients[f'{param_name}_0th_grad'] = grad_0th.item()
            else:
                # No gradients available for 0th primitive
                if param_name == 'c':
                    for color in ['r', 'g', 'b']:
                        gradients[f'{param_name}_0th_grad_{color}'] = 0.0
                else:
                    gradients[f'{param_name}_0th_grad'] = 0.0
        
        # Log to wandb if enabled
        if hasattr(self, 'wandb_enabled') and self.wandb_enabled:
            wandb.log(gradients)

    def clear_cuda_cache(self):
        """
        Completely clear CUDA cache and perform garbage collection.
        """
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        # Reset peak memory stats for memory usage minimization
        torch.cuda.reset_peak_memory_stats()
        
        current = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"After cache clearing: Current={current:.2f}MB, Reserved={reserved:.2f}MB")

    def create_stream(self):
        """
        Create a new CUDA stream to handle memory operations asynchronously.
        This allows multiple operations to be performed simultaneously, reducing memory usage.
        """
        return torch.cuda.Stream()

    def with_stream(self, stream):
        """
        Return a context manager to execute code block in given stream.
        """
        class StreamContext:
            def __init__(self, stream):
                self.stream = stream
                self.prev_stream = None
                
            def __enter__(self):
                self.prev_stream = torch.cuda.current_stream()
                torch.cuda.set_stream(self.stream)
                return self.stream
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                torch.cuda.set_stream(self.prev_stream)
                self.stream.synchronize()
                
        return StreamContext(stream)

from collections import defaultdict

def pretty_mem(x):       # byte → MB unit string
    return f"{x/1024/1024:8.2f} MB"

def tensor_vram_report(namespace: dict):
    """
    namespace(dict): globals()   or locals()
    Prints a table of all torch.Tensor objects (including .grad) on GPU.
    """
    seen_data_ptr = set()
    rows, total = [], 0
    for name, obj in namespace.items():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            # same storage can be shared among multiple variables
            ptr = obj.data_ptr()
            if ptr in seen_data_ptr:
                continue
            seen_data_ptr.add(ptr)

            size_bytes = obj.numel() * obj.element_size()
            rows.append((name, str(tuple(obj.shape)), obj.dtype, pretty_mem(size_bytes)))
            total += size_bytes

            # grad buffer also exists if it exists
            if obj.grad is not None:
                gbytes = obj.grad.numel() * obj.grad.element_size()
                rows.append((name + ".grad", str(tuple(obj.grad.shape)),
                             obj.grad.dtype, pretty_mem(gbytes)))
                total += gbytes

    # Sort: largest first
    rows.sort(key=lambda r: float(r[3].split()[0]), reverse=True)

    print(f"{'tensor':25} {'shape':20} {'dtype':9} {'mem':>10}")
    print("-"*70)
    for n,s,d,m in rows:
        print(f"{n:25} {s:20} {str(d):9} {m:>10}")
    print("-"*70)
    print(f"{'TOTAL':25} {'':20} {'':9} {pretty_mem(total):>10}")


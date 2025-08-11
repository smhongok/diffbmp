import torch
import torch.nn.functional as F
from typing import Tuple, List
from .vector_renderer import VectorRenderer

# Try to import CUDA extension, fallback to PyTorch if not available
try:
    import sys
    import os
    # Add CUDA extension path to sys.path
    cuda_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cuda_tile_rasterizer')
    if cuda_path not in sys.path:
        sys.path.insert(0, cuda_path)
    
    from cuda_tile_rasterizer import rasterize_tiles as cuda_rasterize_tiles
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
        super().__init__(canvas_size, S, **kwargs)
        self.tile_size = tile_size
        
        # Calculate tile grid dimensions
        self.tiles_h = (self.H + tile_size - 1) // tile_size
        self.tiles_w = (self.W + tile_size - 1) // tile_size
        
        # Compute bounding boxes for each primitive in self.S
        self.primitive_bboxes = self._compute_primitive_bboxes()
    
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
        """Sequential tile processing (original method)."""
        for tile_y in range(self.tiles_h):
            for tile_x in range(self.tiles_w):
                # Calculate tile boundaries
                y_start = tile_y * self.tile_size
                y_end = min(y_start + self.tile_size, self.H)
                x_start = tile_x * self.tile_size  
                x_end = min(x_start + self.tile_size, self.W)
                
                # Find primitives that affect this tile
                tile_primitive_indices = self._get_tile_primitives(
                    x, y, r, theta, x_start, x_end, y_start, y_end
                )
                
                # Skip if no primitives affect this tile
                if len(tile_primitive_indices) == 0:
                    continue
                
                # Render this tile with selected primitives only
                tile_result = self._render_tile(
                    x, y, r, theta, v, c, tile_primitive_indices,
                    x_start, x_end, y_start, y_end, sigma, I_bg, no_background,
                    global_bmp_sel=global_bmp_sel
                )
                
                # Place result in output canvas
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
        
        # Vectorized primitive-to-tile assignment
        primitive_tile_masks = self._vectorized_primitive_assignment(
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
    
    def _vectorized_primitive_assignment(self, x: torch.Tensor, y: torch.Tensor, 
                                       r: torch.Tensor, theta: torch.Tensor,
                                       x_starts: torch.Tensor, x_ends: torch.Tensor,
                                       y_starts: torch.Tensor, y_ends: torch.Tensor) -> torch.Tensor:
        """Vectorized computation of which primitives affect which tiles."""
        N = len(x)
        total_tiles = len(x_starts)
        
        # Expand dimensions for broadcasting: (N, 1) and (1, total_tiles)
        x_exp = x.unsqueeze(1)  # (N, 1)
        y_exp = y.unsqueeze(1)  # (N, 1)
        r_exp = r.unsqueeze(1)  # (N, 1)
        
        x_starts_exp = x_starts.unsqueeze(0)  # (1, total_tiles)
        x_ends_exp = x_ends.unsqueeze(0)      # (1, total_tiles)
        y_starts_exp = y_starts.unsqueeze(0)  # (1, total_tiles)
        y_ends_exp = y_ends.unsqueeze(0)      # (1, total_tiles)
        
        # Compute primitive bounding boxes (considering rotation and radius)
        # For simplicity, use conservative bounding box: center ± (r * sqrt(2))
        bbox_margin = r_exp * 1.5  # Conservative margin for rotation
        
        prim_x_min = x_exp - bbox_margin
        prim_x_max = x_exp + bbox_margin
        prim_y_min = y_exp - bbox_margin
        prim_y_max = y_exp + bbox_margin
        
        # Check intersection: primitive bbox overlaps with tile bbox
        # Intersection condition: not (prim_max < tile_min or prim_min > tile_max)
        x_intersect = ~((prim_x_max < x_starts_exp) | (prim_x_min > x_ends_exp))
        y_intersect = ~((prim_y_max < y_starts_exp) | (prim_y_min > y_ends_exp))
        
        # Both x and y must intersect
        intersections = x_intersect & y_intersect  # (N, total_tiles)
        
        # Transpose to get (total_tiles, N) - each row is primitives affecting that tile
        return intersections.t()
    
    def render(self, cached_masks: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
               return_alpha: bool = False, I_bg: torch.Tensor = None,
               no_background: bool = False) -> torch.Tensor:
        """
        VectorRenderer-compatible render method that uses pre-computed cached_masks.
        This method provides compatibility with the existing optimization pipeline.
        
        Args:
            cached_masks: (N, H, W) pre-computed soft masks
            v: (N,) visibility logits
            c: (N, 3) RGB logits
            return_alpha: Whether to return alpha channel
            I_bg: Background image
            no_background: Whether to use no background
            
        Returns:
            Rendered image tensor
        """
        # Use parent class's render method for compatibility
        return super().render(cached_masks, v, c, return_alpha, I_bg, no_background)
    
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
        save_image_intervals = [1, 5, 10, 20, 50, 100]
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
    
    def compute_loss(self, rendered: torch.Tensor, target: torch.Tensor, 
                    x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, 
                    v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between rendered and target images."""
        # Simple MSE loss for now
        return F.mse_loss(rendered, target)
    
    def _get_tile_primitives(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, 
                            theta: torch.Tensor, x_start: int, x_end: int, 
                            y_start: int, y_end: int) -> List[int]:
        """
        Find primitives whose transformed bounding boxes intersect with the given tile.
        
        Args:
            x, y: (N,) primitive positions
            r: (N,) primitive scales
            theta: (N,) primitive rotations
            x_start, x_end, y_start, y_end: Tile boundaries
            
        Returns:
            List of primitive indices that affect this tile
        """
        N = x.shape[0]
        p = len(self.primitive_bboxes)
        intersecting_indices = []
        
        for i in range(N):
            # Get primitive index (cycling through available primitives)
            prim_idx = i % p
            min_u, max_u, min_v, max_v = self.primitive_bboxes[prim_idx]
            
            # Transform bounding box corners to world coordinates
            # Apply scale and rotation
            cos_t = torch.cos(theta[i])
            sin_t = torch.sin(theta[i])
            scale = r[i]
            
            # Bounding box corners in primitive space
            corners_u = torch.tensor([min_u, max_u, min_u, max_u], device=self.device)
            corners_v = torch.tensor([min_v, min_v, max_v, max_v], device=self.device)
            
            # Transform to world coordinates
            world_x = x[i] + scale * (corners_u * cos_t - corners_v * sin_t)
            world_y = y[i] + scale * (corners_u * sin_t + corners_v * cos_t)
            
            # Check if transformed bounding box intersects tile
            bbox_x_min = world_x.min().item()
            bbox_x_max = world_x.max().item()
            bbox_y_min = world_y.min().item()
            bbox_y_max = world_y.max().item()
            
            # Intersection test
            if (bbox_x_min < x_end and bbox_x_max > x_start and
                bbox_y_min < y_end and bbox_y_max > y_start):
                intersecting_indices.append(i)
        
        return intersecting_indices
    
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
        
        # Try CUDA acceleration if available
        if CUDA_AVAILABLE and len(primitive_indices) > 0:
            try:
                print(f"[DEBUG] Using CUDA for tile ({x_start}-{x_end}, {y_start}-{y_end}) with {len(primitive_indices)} primitives")
                
                # Prepare data for CUDA - ensure all tensors are float32
                means2D = torch.stack([tile_x, tile_y], dim=1).float()  # (N, 2)
                radii = tile_r.float()  # (N,)
                rotations = tile_theta.float()  # (N,)
                opacities = tile_v.float()  # (N,) logits
                colors = tile_c.float()  # (N, 3) logits
                
                # Get primitive templates for selected primitives
                if global_bmp_sel is not None:
                    selected_templates = global_bmp_sel[primitive_indices].float()  # (N, H, W)
                else:
                    # Use default templates
                    selected_templates = self.S[primitive_indices].float()  # (N, H, W)
                
                # Call CUDA rasterizer
                cuda_color, cuda_alpha = cuda_rasterize_tiles(
                    means2D, radii, rotations, opacities, colors,
                    selected_templates, tile_h, tile_w, 
                    min(tile_h, tile_w), sigma  # Use smaller dimension as tile_size
                )
                
                # CUDA returns (H, W, 3) and (H, W)
                comp_m = cuda_color
                comp_a = cuda_alpha
                
                print(f"[DEBUG] CUDA tile rendering successful: {comp_m.shape}, {comp_a.shape}")
                
            except Exception as e:
                print(f"[DEBUG] CUDA failed, falling back to PyTorch: {e}")
                # Fallback to PyTorch
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
        else:
            print(f"[DEBUG] Using PyTorch fallback for tile ({x_start}-{x_end}, {y_start}-{y_end})")
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
        result = renderer.render_from_params(x, y, r, theta, v, c, sigma=1.0)
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
                
                tile_primitives = renderer._get_tile_primitives(
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
            
            tile_primitives = renderer._get_tile_primitives(
                x, y, r, theta, x_start, x_end, y_start, y_end
            )
            print(f"  Tile ({tile_y},{tile_x}) [{x_start}-{x_end}, {y_start}-{y_end}]: {len(tile_primitives)} primitives {tile_primitives}")
        
        print(f"\n📁 All visualizations saved to: {output_dir}/")
        print("\n✅ SimpleTileRenderer test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        import traceback
        traceback.print_exc()

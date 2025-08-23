import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .mse_renderer import MseRenderer


class SequentialFrameRenderer(MseRenderer):
    """
    Specialized renderer for subsequent frames in sequential optimization.
    Inherits from MseRenderer but uses warmup scheduling for loss.
    """
    
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None, tile_size=32):
        super().__init__(canvas_size, S, alpha_upper_bound, device, use_fp16, gamma, output_path, tile_size)
    
    def compute_loss_with_warmup(self, 
                               rendered: torch.Tensor, 
                               target: torch.Tensor, 
                               x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor,
                               r: torch.Tensor, v: torch.Tensor, c: torch.Tensor,
                               loss_config: dict = None,
                               current_iter: int = 0,
                               total_iters: int = 100) -> torch.Tensor:
        """
        Compute loss with warmup ramping.
        
        Args:
            rendered: Rendered image tensor
            target: Target image tensor
            x, y, theta, r, v, c: Current frame parameters
            loss_config: Loss configuration dict
            current_iter: Current iteration number
            total_iters: Total number of iterations
            
        Returns:
            Loss tensor with warmup applied
        """
        if loss_config is None:
            loss_config = {'reconstruction_weight': 1.0}
        
        # Get warmup steps for loss ramping
        warmup_steps = loss_config.get('warmup_steps', 0)
        
        # Compute warmup multiplier for all losses
        if warmup_steps > 0 and current_iter < warmup_steps:
            warmup_multiplier = current_iter / warmup_steps
        else:
            warmup_multiplier = 1.0
        
        # Check if combined loss is enabled
        combined_loss_config = loss_config.get('combined_loss', {})
        use_combined_loss = combined_loss_config.get('enabled', False)
        
        if use_combined_loss:
            # Use combined loss with grayscale, color, gradient, and canny components
            reconstruction_loss = self.compute_combined_loss(
                rendered, target, x, y, r, v, theta, c,
                grayscale_weight=combined_loss_config.get('grayscale_weight', 0.7),
                color_weight=combined_loss_config.get('color_weight', 0.3),
                use_gradient_loss=combined_loss_config.get('use_gradient_loss', False),
                gradient_weight=combined_loss_config.get('gradient_weight', 0.1),
                use_cosine_similarity=combined_loss_config.get('use_cosine_similarity', False),
                use_canny_loss=combined_loss_config.get('use_canny_loss', False),
                canny_weight=combined_loss_config.get('canny_weight', 0.1)
            )
            
        else:
            # Use standard loss
            reconstruction_loss = self.compute_loss(rendered, target, x, y, r, v, theta, c)
            
        
        reconstruction_weight = loss_config.get('reconstruction_weight', 1.0)
        total_loss = reconstruction_weight * reconstruction_loss * warmup_multiplier
        
        return total_loss
    
    def _extract_learning_rates(self, opt_conf: dict) -> dict:
        """Extract learning rate configuration from optimization config."""
        lr_config = opt_conf.get("learning_rate", {})
        
        if isinstance(lr_config, (int, float)):
            # Simple learning rate format
            return {
                'default_lr': lr_config,
                'gain_x': 1.0, 'gain_y': 1.0, 'gain_r': 1.0,
                'gain_v': 1.0, 'gain_theta': 1.0, 'gain_c': 1.0,
                'decay_rate': opt_conf.get('decay_rate', 0.95)
            }
        else:
            # Complex learning rate format with gains
            return {
                'default_lr': lr_config.get('default', 0.005),
                'gain_x': lr_config.get('gain_x', 1.0),
                'gain_y': lr_config.get('gain_y', 1.0),
                'gain_r': lr_config.get('gain_r', 1.0),
                'gain_v': lr_config.get('gain_v', 1.0),
                'gain_theta': lr_config.get('gain_theta', 1.0),
                'gain_c': lr_config.get('gain_c', 1.0),
                'decay_rate': opt_conf.get('decay_rate', 0.95)
            }
    
    def optimize_parameters_full_temporal(self,
                                        x: torch.Tensor,
                                        y: torch.Tensor,
                                        r: torch.Tensor,
                                        v: torch.Tensor,
                                        theta: torch.Tensor,
                                        c: torch.Tensor,
                                        target: torch.Tensor,
                                        prev_params: dict,
                                        opt_conf: dict = None) -> tuple:
        """
        Optimize all parameters for subsequent frames with warmup scheduling.
        
        Args:
            x, y, r, v, theta, c: Current parameter tensors
            target: Target image tensor
            prev_params: Previous frame parameters (not used anymore)
            opt_conf: Optimization configuration
            
        Returns:
            Optimized parameter tensors (x, y, r, v, theta, c)
        """
        if opt_conf is None:
            opt_conf = {"num_iterations": 50, "learning_rate": {"default": 0.005}, "decay_rate": 0.95}
        
        # Ensure all parameters are leaf tensors with gradients enabled
        x.requires_grad_(True)
        y.requires_grad_(True)
        r.requires_grad_(True)
        v.requires_grad_(True)
        theta.requires_grad_(True)
        c.requires_grad_(True)
        
        # Extract learning rate configuration
        lr_config = self._extract_learning_rates(opt_conf)
        
        # Setup optimizer with parameter-specific learning rates
        optimizer = torch.optim.Adam([
            {'params': [x], 'lr': lr_config['default_lr'] * lr_config['gain_x']},
            {'params': [y], 'lr': lr_config['default_lr'] * lr_config['gain_y']},
            {'params': [r], 'lr': lr_config['default_lr'] * lr_config['gain_r']},
            {'params': [v], 'lr': lr_config['default_lr'] * lr_config['gain_v']},
            {'params': [theta], 'lr': lr_config['default_lr'] * lr_config['gain_theta']},
            {'params': [c], 'lr': lr_config['default_lr'] * lr_config['gain_c']}
        ])
        
        # Use exponential decay scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_config['decay_rate'])
        
        num_iter = opt_conf.get("num_iterations", opt_conf.get("num_iter", 50))
        adaptive_config = opt_conf.get('adaptive_control', {})
        apply_every_n = adaptive_config.get('apply_every_n_iterations', 10)
        
        pbar = tqdm(range(num_iter), desc="Optimizing with warmup scheduling")
        
        print(f"Starting tile-based optimization for sequential frames, {num_iter} iterations...")

        # Main optimization loop
        for i in pbar:
            # Apply adaptive control periodically if enabled
            if (adaptive_config.get('enabled', False) and 
                i > 0 and i % apply_every_n == 0):
                
                # Apply adaptive control and get new leaf tensors
                x, y, r, v, theta, c = self.apply_adaptive_control(
                    x, y, r, v, theta, c, target, adaptive_config)
                
                # Update optimizer parameters with new tensors
                optimizer.param_groups[0]['params'] = [x]
                optimizer.param_groups[1]['params'] = [y]
                optimizer.param_groups[2]['params'] = [r]
                optimizer.param_groups[3]['params'] = [v]
                optimizer.param_groups[4]['params'] = [theta]
                optimizer.param_groups[5]['params'] = [c]
            
            optimizer.zero_grad()
            
            # Generate rendered image using tile-based rendering
            rendered = self.render_from_params(x, y, r, theta, v, c, sigma=0.0)
            
            # Compute loss with warmup scheduling
            loss_config = {
                'reconstruction_weight': 1.0,
                'warmup_steps': opt_conf.get('warmup_steps', 0),
                'combined_loss': opt_conf.get('combined_loss', {})
            }
            loss = self.compute_loss_with_warmup(
                rendered, target, x, y, theta, r, v, c, 
                loss_config, i, num_iter
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            postfix = {'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'}
            if (adaptive_config.get('enabled', False) and 
                i > 0 and i % apply_every_n == 0):
                postfix['adaptive'] = 'applied'
            pbar.set_postfix(postfix)
        
        return x, y, r, v, theta, c
    
    def apply_adaptive_control(self, 
                             x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, 
                             v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                             target_image: torch.Tensor,
                             adaptive_config: dict = None) -> tuple:
        """
        Apply adaptive control to reduce artifacts by lowering opacity of problematic primitives.
        
        Args:
            x, y, r, v, theta, c: Current primitive parameters
            target_image: Target image tensor [H, W, 3] for pixel-based color_nerf
            adaptive_config: Configuration dict for adaptive control
            
        Returns:
            Tuple of adapted parameters (x, y, r, v, theta, c)
        """
        if adaptive_config is None or not adaptive_config.get('enabled', False):
            return x, y, r, v, theta, c
        
        # Extract configuration parameters
        tile_rows = adaptive_config.get('tile_rows', 4)
        tile_cols = adaptive_config.get('tile_cols', 4)
        scale_threshold = adaptive_config.get('scale_threshold', 8.0)
        opacity_threshold = adaptive_config.get('opacity_threshold', 0.7)
        opacity_reduction_factor = adaptive_config.get('opacity_reduction_factor', 0.5)
        max_primitives_per_tile = adaptive_config.get('max_primitives_per_tile', 3)
        
        # Extract gradient-based criterion configuration
        gradient_config = adaptive_config.get('gradient_criterion', {})
        use_gradient_criterion = gradient_config.get('enabled', False)
        gradient_threshold_percentile = gradient_config.get('threshold_percentile', 0.7)
        
        # Extract color_nerf configuration
        color_nerf_config = adaptive_config.get('color_nerf', {})
        color_nerf_enabled = color_nerf_config.get('enabled', False)
        color_nerf_mode = color_nerf_config.get('mode', 'mean')
        
        # Create new leaf tensors - single detach operation
        x_adapted = x.detach().requires_grad_(True)
        y_adapted = y.detach().requires_grad_(True)
        r_adapted = r.detach().requires_grad_(True)
        v_adapted = v.detach().requires_grad_(True)
        theta_adapted = theta.detach().requires_grad_(True)
        c_adapted = c.detach().requires_grad_(True)
        
        # Get canvas dimensions
        canvas_height, canvas_width = self.H, self.W
        
        # Calculate tile dimensions
        tile_height = canvas_height // tile_rows
        tile_width = canvas_width // tile_cols
        
        # Compute gradient magnitudes once for ranking (always computed now)
        gradient_magnitudes = self._compute_per_primitive_gradient_magnitude(
            x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted, target_image, adaptive_config
        )
        
        # Process each tile
        for row in range(tile_rows):
            for col in range(tile_cols):
                # Define tile boundaries
                y_min = row * tile_height
                y_max = (row + 1) * tile_height
                x_min = col * tile_width
                x_max = (col + 1) * tile_width
                
                # Find primitives in this tile
                in_tile_mask = ((x_adapted >= x_min) & (x_adapted < x_max) & 
                               (y_adapted >= y_min) & (y_adapted < y_max))
                
                if not in_tile_mask.any():
                    continue
                
                # Get indices of primitives in this tile
                tile_indices = torch.where(in_tile_mask)[0]
                
                # Find problematic primitives (already limited by max_primitives_per_tile)
                problematic_indices = self._find_problematic_primitives(
                    tile_indices, x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted,
                    scale_threshold, opacity_threshold, gradient_magnitudes, max_primitives_per_tile
                )
                
                # Apply opacity reduction using in-place operation to maintain leaf tensor status
                if len(problematic_indices) > 0:
                    with torch.no_grad():
                        v_adapted[problematic_indices] *= opacity_reduction_factor
                
                # Apply color normalization (color_nerf) if enabled
                if color_nerf_enabled:
                    self._apply_color_nerf(tile_indices, problematic_indices, c_adapted, 
                                          x_adapted, y_adapted, target_image, color_nerf_mode)
        
        return x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted
    
    def _find_problematic_primitives(self, 
                               tile_indices: torch.Tensor,
                               x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                               v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                               scale_threshold: float, opacity_threshold: float,
                               gradient_magnitudes: torch.Tensor = None,
                               max_primitives_per_tile: int = 16) -> torch.Tensor:
        """
        Find problematic primitives within a tile based on three criteria,
        then select top-k by gradient magnitude.
        
        Args:
            tile_indices: Indices of primitives in this tile
            x, y, r, v, theta, c: Primitive parameters
            scale_threshold: Threshold for large scale primitives
            opacity_threshold: Threshold for high opacity primitives
            gradient_magnitudes: Per-primitive gradient magnitudes for ranking
            max_primitives_per_tile: Maximum number of primitives to select
            
        Returns:
            Indices of problematic primitives (top-k by gradient magnitude)
        """
        if len(tile_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=tile_indices.device)
        
        # Extract parameters for primitives in this tile
        tile_r = r[tile_indices]
        tile_v = v[tile_indices]
        tile_opacity = self.alpha_upper_bound*torch.sigmoid(tile_v)
        
        # Criterion 1: Large scale primitives
        large_scale_mask = tile_r >= scale_threshold
        
        # Criterion 2: High opacity primitives
        high_opacity_mask = tile_opacity >= opacity_threshold
        
        # Criterion 3: Front primitives (those with higher z-order)
        # In splatting, higher array indices are rendered later, thus appearing in front
        # We consider primitives in the top 30% of indices within this tile as "front"
        if len(tile_indices) > 0:
            tile_indices_sorted = torch.sort(tile_indices)[0]  # Sort tile indices
            front_threshold_idx = int(len(tile_indices_sorted) * 0.7)  # Top 30%
            front_indices_threshold = tile_indices_sorted[front_threshold_idx] if front_threshold_idx < len(tile_indices_sorted) else tile_indices_sorted[-1]
            
            # Create mask for primitives that are in the front (high z-order) - vectorized
            front_mask = tile_indices >= front_indices_threshold
        else:
            front_mask = torch.tensor([], dtype=torch.bool, device=tile_indices.device)
        
        # Debug: Print criteria counts
        print(f"[Adaptive Control Debug] Tile criteria counts:")
        print(f"  - Large scale (>={scale_threshold}): {large_scale_mask.sum().item()}/{len(tile_indices)}")
        print(f"  - High opacity (>={opacity_threshold:.2f}): {high_opacity_mask.sum().item()}/{len(tile_indices)}")
        print(f"  - Front primitives (top 30%): {front_mask.sum().item()}/{len(tile_indices)}")
        
        # Combine criteria: primitives meeting at least 2 out of 3 criteria
        criteria_count = (large_scale_mask.float() + high_opacity_mask.float() + front_mask.float())
        problematic_mask = criteria_count >= 2.0
        
        print(f"  - Problematic by criteria (>=2/3): {problematic_mask.sum().item()}/{len(tile_indices)}")
        
        # Get indices of problematic primitives within the tile
        problematic_tile_indices = torch.where(problematic_mask)[0]
        problematic_global_indices = tile_indices[problematic_tile_indices]
        
        # If we have gradient magnitudes, select top-k by gradient magnitude
        if gradient_magnitudes is not None and len(problematic_global_indices) > 0:
            # Get gradient magnitudes for problematic primitives
            problematic_gradients = gradient_magnitudes[problematic_global_indices]
            
            # Select top-k by gradient magnitude
            num_to_select = min(max_primitives_per_tile, len(problematic_global_indices))
            if num_to_select < len(problematic_global_indices):
                _, top_gradient_indices = torch.topk(problematic_gradients, num_to_select)
                selected_indices = problematic_global_indices[top_gradient_indices]
                print(f"  - Selected top-{num_to_select} by gradient: {num_to_select}/{len(problematic_global_indices)}")
            else:
                selected_indices = problematic_global_indices
                print(f"  - Selected all problematic: {len(problematic_global_indices)}/{len(problematic_global_indices)}")
        else:
            # Fallback: limit by max_primitives_per_tile using scale*opacity score
            num_to_select = min(max_primitives_per_tile, len(problematic_global_indices))
            if num_to_select < len(problematic_global_indices):
                scores = r[problematic_global_indices] * torch.sigmoid(v[problematic_global_indices])
                _, top_indices = torch.topk(scores, num_to_select)
                selected_indices = problematic_global_indices[top_indices]
                print(f"  - Selected top-{num_to_select} by scale*opacity: {num_to_select}/{len(problematic_global_indices)}")
            else:
                selected_indices = problematic_global_indices
                print(f"  - Selected all problematic: {len(problematic_global_indices)}/{len(problematic_global_indices)}")
        
        return selected_indices
    
    def _apply_color_nerf(self, 
                         tile_indices: torch.Tensor,
                         selected_indices: torch.Tensor,
                         c: torch.Tensor,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         target_image: torch.Tensor,
                         mode: str = "mean") -> None:
        """
        Apply color normalization (color_nerf) to selected primitives within a tile.
        
        Args:
            tile_indices: Indices of all primitives in the current tile
            selected_indices: Indices of selected problematic primitives to modify
            c: Color tensor to modify in-place
            x: X coordinates of primitives
            y: Y coordinates of primitives
            target_image: Target image tensor [H, W, 3]
            mode: "mean" for tile mean color, "pixel" for target image pixel color
        """
        if len(tile_indices) == 0 or len(selected_indices) == 0:
            return
        
        with torch.no_grad():
            if mode == "mean":
                # Compute mean color of all primitives in the tile
                tile_colors = c[tile_indices]  # Shape: [num_tile_primitives, 3]
                mean_color = torch.mean(tile_colors, dim=0, keepdim=True)  # Shape: [1, 3]
                c[selected_indices] = mean_color.expand(len(selected_indices), -1)
                
            elif mode == "pixel":
                # Get target image pixel colors at primitive positions
                selected_x = x[selected_indices].long().clamp(0, target_image.shape[1] - 1)
                selected_y = y[selected_indices].long().clamp(0, target_image.shape[0] - 1)
                
                # Sample colors from target image at primitive positions
                pixel_colors = target_image[selected_y, selected_x]  # Shape: [num_selected, 3]
                c[selected_indices] = pixel_colors
    
    def _compute_per_primitive_gradient_magnitude(self, 
                                                 x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                                 v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                                 target: torch.Tensor,
                                                 adaptive_config: dict = None) -> torch.Tensor:
        """
        Compute per-primitive gradient magnitude based on absolute x,y gradients.
        Inspired by AbsGS research - measures how much each primitive's position affects the total loss.
        Uses fixed-count pixel sampling per tile for performance optimization.
        
        Args:
            x, y, r, v, theta, c: Primitive parameters with gradients
            target: Target image tensor [H, W, 3]
            adaptive_config: Configuration dict containing tile and sampling parameters
            
        Returns:
            Tensor of gradient magnitudes per primitive [N]
        """
        # Ensure x, y parameters require gradients
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        
        # Render current state
        rendered = self.render_from_params(x, y, r, v, theta, c)
        
        # Pixel-wise loss (reduction 없음)
        pixel_losses = (rendered - target).pow(2).mean(dim=2)  # [H, W]
        H, W = pixel_losses.shape
        
        # Get tile configuration (same as apply_adaptive_control)
        if adaptive_config is None:
            adaptive_config = {}
        tile_rows = adaptive_config.get('tile_rows', 4)
        tile_cols = adaptive_config.get('tile_cols', 4)
        
        # Get gradient sampling configuration
        gradient_config = adaptive_config.get('gradient_ranking', {})
        pixels_per_tile = gradient_config.get('pixels_per_tile', 16)
        
        # Calculate tile dimensions (same as apply_adaptive_control)
        tile_height = H // tile_rows
        tile_width = W // tile_cols
        
        # Initialize gradient magnitude accumulator
        gradient_magnitudes = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        
        total_pixels_processed = 0
        total_pixels_available = H * W
        
        print(f"[Gradient Computation] Processing {tile_rows}x{tile_cols} tiles with {pixels_per_tile} pixels per tile")
        
        # Process each tile
        for row in range(tile_rows):
            for col in range(tile_cols):
                # Define tile boundaries (same as apply_adaptive_control)
                y_min = row * tile_height
                y_max = (row + 1) * tile_height
                x_min = col * tile_width
                x_max = (col + 1) * tile_width
                
                # Get all pixel coordinates in this tile
                tile_pixels = []
                for i in range(y_min, min(y_max, H)):
                    for j in range(x_min, min(x_max, W)):
                        tile_pixels.append((i, j))
                
                # Sample fixed number of pixels within this tile
                num_tile_pixels = len(tile_pixels)
                num_sample_pixels = min(pixels_per_tile, num_tile_pixels)  # Don't exceed available pixels
                
                if num_tile_pixels > 0 and num_sample_pixels > 0:
                    # Random sampling within this tile
                    if num_sample_pixels < num_tile_pixels:
                        sample_indices = torch.randperm(num_tile_pixels)[:num_sample_pixels]
                        sampled_pixels = [tile_pixels[idx] for idx in sample_indices]
                    else:
                        # Use all pixels if requested count >= available pixels
                        sampled_pixels = tile_pixels
                    
                    # Process sampled pixels in this tile
                    for i, j in sampled_pixels:
                        pixel_loss = pixel_losses[i, j]
                        
                        # Compute gradients w.r.t. x, y for this pixel
                        if pixel_loss.requires_grad:
                            grads = torch.autograd.grad(
                                outputs=pixel_loss,
                                inputs=[x, y],
                                retain_graph=True,
                                create_graph=False,
                                allow_unused=True
                            )
                            
                            # Sum absolute gradients for x and y per primitive (AbsGS style)
                            if grads[0] is not None:  # x gradients [N]
                                gradient_magnitudes += torch.abs(grads[0])
                            if grads[1] is not None:  # y gradients [N]
                                gradient_magnitudes += torch.abs(grads[1])
                    
                    total_pixels_processed += len(sampled_pixels)
        
        # Scale by actual sampling ratio to approximate full image gradient
        actual_sample_ratio = total_pixels_processed / total_pixels_available if total_pixels_available > 0 else 1.0
        gradient_magnitudes = gradient_magnitudes / actual_sample_ratio
        
        print(f"[Gradient Computation] Processed {total_pixels_processed}/{total_pixels_available} pixels ({actual_sample_ratio*100:.1f}%)")
        
        return gradient_magnitudes
    


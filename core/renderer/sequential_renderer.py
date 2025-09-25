import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .simple_tile_renderer import SimpleTileRenderer

# =============================================================================
# DEBUG CONFIGURATION FOR GRADIENT VISUALIZATION
# =============================================================================
# Set to True to enable gradient visualization during adaptive control
ENABLE_GRADIENT_DEBUG_VISUALIZATION = False

# Set to True to enable non-problematic primitive gradient visualization for comparison
ENABLE_NON_PROBLEMATIC_PRIMITIVE_GRADIENT_VISUALIZATION = False

# Directory to save gradient visualization images
GRADIENT_DEBUG_SAVE_DIR = "./outputs/debug_gradients_sequential"

# =============================================================================
# CUDA CONFIGURATION FOR GRADIENT COMPUTATION
# =============================================================================
# Set to True to use CUDA implementation for gradient computation, False to use Python implementation
USE_CUDA_GRADIENT_COMPUTATION = False

# =============================================================================


class SequentialFrameRenderer(SimpleTileRenderer):
    """
    Specialized renderer for subsequent frames in sequential optimization.
    Inherits from MseRenderer but uses warmup scheduling for loss.
    """
    
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None, tile_size=32):
        # Pass parameters to SimpleTileRenderer using keyword arguments
        super().__init__(canvas_size, S, tile_size=tile_size, 
                        alpha_upper_bound=alpha_upper_bound, device=device, 
                        use_fp16=use_fp16, gamma=gamma, output_path=output_path)
    
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
                
                # Choose between debug visualization or normal adaptive control
                if ENABLE_GRADIENT_DEBUG_VISUALIZATION:
                    # Use debug version with gradient visualization
                    import os
                    from datetime import datetime
                    
                    # Create timestamped subdirectory for this iteration
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    iteration_save_dir = os.path.join(GRADIENT_DEBUG_SAVE_DIR, f"iter_{i:04d}_{timestamp}")
                    
                    print(f"\n[Debug Mode] Applying adaptive control with gradient visualization at iteration {i}")
                    print(f"[Debug Mode] Saving visualizations to: {iteration_save_dir}")
                    
                    # Apply debug adaptive control with visualization
                    (x, y, r, v, theta, c), tile_info = self.debug_adaptive_control_with_visualization(
                        x, y, r, v, theta, c, target, adaptive_config, iteration_save_dir
                    )
                    
                    # Log debug information
                    total_selected = sum(len(tile['selected_indices']) for tile in tile_info)
                    print(f"[Debug Mode] Iteration {i}: {total_selected} primitives selected across {len(tile_info)} tiles")
                    
                else:
                    # Use normal adaptive control without visualization
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
            I_bg = torch.ones((self.H, self.W, 3), device=self.device)
            rendered = self.render_from_params(x, y, r, theta, v, c, I_bg=I_bg, sigma=0.0)
            
            # Compute loss with warmup scheduling
            loss_config = {
                'reconstruction_weight': 1.0,
                'warmup_steps': opt_conf.get('warmup_steps', 0)
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

        print("DEBUG: apply_adaptive_control")


        if adaptive_config is None or not adaptive_config.get('enabled', False):
            return x, y, r, v, theta, c
        
        # Extract configuration parameters
        tile_rows = adaptive_config.get('tile_rows', 4)
        tile_cols = adaptive_config.get('tile_cols', 4)
        scale_threshold = adaptive_config.get('scale_threshold', 8.0)
        opacity_threshold = adaptive_config.get('opacity_threshold', 0.7)
        opacity_reduction_factor = adaptive_config.get('opacity_reduction_factor', 0.5)
        max_primitives_per_tile = adaptive_config.get('max_primitives_per_tile', 3)
        
        

        
        # Extract gradient_ranking configuration
        gradient_ranking_config = adaptive_config.get('gradient_ranking', {})
        gradient_ranking_enabled = gradient_ranking_config.get('enabled', True)
        print(f"[Adaptive Control] Gradient ranking enabled: {gradient_ranking_enabled}")
        
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
        gradient_magnitudes = self._compute_per_pixel_gradient_magnitude(
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
                    scale_threshold, opacity_threshold, gradient_magnitudes, max_primitives_per_tile,
                    gradient_ranking_enabled
                )
                
                # Apply opacity reduction using in-place operation to maintain leaf tensor status
                if len(problematic_indices) > 0:
                    with torch.no_grad():
                        v_adapted[problematic_indices] *= opacity_reduction_factor
      
        return x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted
    
    def _find_problematic_primitives(self, 
                           tile_indices: torch.Tensor,
                           x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                           v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                           scale_threshold: float, opacity_threshold: float,
                           gradient_magnitudes: torch.Tensor = None,
                           max_primitives_per_tile: int = 16,
                           gradient_ranking_enabled: bool = True) -> torch.Tensor:
        """
        Find problematic primitives within a tile based on three criteria,
        then select top-k by gradient magnitude or scale*opacity based on configuration.
        
        Args:
            tile_indices: Indices of primitives in this tile
            x, y, r, v, theta, c: Primitive parameters
            scale_threshold: Threshold for large scale primitives
            opacity_threshold: Threshold for high opacity primitives
            gradient_magnitudes: Per-primitive gradient magnitudes for ranking
            max_primitives_per_tile: Maximum number of primitives to select
            gradient_ranking_enabled: Whether to use gradient-based ranking (True) or scale*opacity fallback (False)
            
        Returns:
            Indices of problematic primitives (top-k by gradient magnitude or scale*opacity)
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
        
        # Combine criteria: primitives meeting all 3 criteria
        criteria_count = (large_scale_mask.float() + high_opacity_mask.float() + front_mask.float())
        problematic_mask = criteria_count >= 2.0
        
        print(f"  - Problematic by criteria (>=2/3): {problematic_mask.sum().item()}/{len(tile_indices)}")
        
        # Get indices of problematic primitives within the tile
        problematic_tile_indices = torch.where(problematic_mask)[0]
        problematic_global_indices = tile_indices[problematic_tile_indices]
        
        # Select top-k primitives based on configuration
        if gradient_ranking_enabled and gradient_magnitudes is not None and len(problematic_global_indices) > 0:
            # Use gradient-based ranking
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
            # Use scale*opacity ranking (either disabled by config or gradient magnitudes unavailable)
            num_to_select = min(max_primitives_per_tile, len(problematic_global_indices))
            if num_to_select < len(problematic_global_indices):
                scores = r[problematic_global_indices] * torch.sigmoid(v[problematic_global_indices])
                _, top_indices = torch.topk(scores, num_to_select)
                selected_indices = problematic_global_indices[top_indices]
                ranking_method = "gradient (unavailable)" if gradient_ranking_enabled else "scale*opacity (config)"
                print(f"  - Selected top-{num_to_select} by {ranking_method}: {num_to_select}/{len(problematic_global_indices)}")
            else:
                selected_indices = problematic_global_indices
                print(f"  - Selected all problematic: {len(problematic_global_indices)}/{len(problematic_global_indices)}")
        
        return selected_indices
    
    def _find_non_problematic_primitives(self, 
                               tile_indices: torch.Tensor,
                               x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                               v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                               scale_threshold: float, opacity_threshold: float,
                               max_primitives_per_tile: int = 16) -> torch.Tensor:
        """
        Find non-problematic primitives within a tile based on three opposite criteria,
        then select top-k by scale * opacity score.
        
        Args:
            tile_indices: Indices of primitives in this tile
            x, y, r, v, theta, c: Primitive parameters
            scale_threshold: Threshold for small scale primitives (opposite of large)
            opacity_threshold: Threshold for low opacity primitives (opposite of high)
            max_primitives_per_tile: Maximum number of primitives to select
            
        Returns:
            Indices of non-problematic primitives (top-k by scale * opacity)
        """
        if len(tile_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=tile_indices.device)
        
        # Extract parameters for primitives in this tile
        tile_r = r[tile_indices]
        tile_v = v[tile_indices]
        tile_opacity = self.alpha_upper_bound * torch.sigmoid(tile_v)
        
        # Criterion 1: Small scale primitives (opposite of large scale)
        small_scale_mask = tile_r < scale_threshold
        
        # Criterion 2: Low opacity primitives (opposite of high opacity)
        low_opacity_mask = tile_opacity < opacity_threshold
        
        # Criterion 3: Back primitives (those with lower z-order, opposite of front)
        # In splatting, lower array indices are rendered earlier, thus appearing in back
        # We consider primitives in the bottom 30% of indices within this tile as "back"
        if len(tile_indices) > 0:
            tile_indices_sorted = torch.sort(tile_indices)[0]  # Sort tile indices
            back_threshold_idx = int(len(tile_indices_sorted) * 0.3)  # Bottom 30%
            back_indices_threshold = tile_indices_sorted[back_threshold_idx] if back_threshold_idx < len(tile_indices_sorted) else tile_indices_sorted[0]
            
            # Create mask for primitives that are in the back (low z-order) - vectorized
            back_mask = tile_indices <= back_indices_threshold
        else:
            back_mask = torch.tensor([], dtype=torch.bool, device=tile_indices.device)
        
        # Debug: Print criteria counts
        print(f"[Non-Problematic Debug] Tile criteria counts:")
        print(f"  - Small scale (<{scale_threshold}): {small_scale_mask.sum().item()}/{len(tile_indices)}")
        print(f"  - Low opacity (<{opacity_threshold:.2f}): {low_opacity_mask.sum().item()}/{len(tile_indices)}")
        print(f"  - Back primitives (bottom 30%): {back_mask.sum().item()}/{len(tile_indices)}")
        
        # Combine criteria: primitives meeting all 3 criteria (opposite approach)
        criteria_count = (small_scale_mask.float() + low_opacity_mask.float() + back_mask.float())
        non_problematic_mask = criteria_count >= 3.0
        
        print(f"  - Non-problematic by criteria (>=3/3): {non_problematic_mask.sum().item()}/{len(tile_indices)}")
        
        # Get indices of non-problematic primitives within the tile
        non_problematic_tile_indices = torch.where(non_problematic_mask)[0]
        non_problematic_global_indices = tile_indices[non_problematic_tile_indices]
        
        # Select randomly from non-problematic primitives (for unbiased comparison)
        num_to_select = min(max_primitives_per_tile, len(non_problematic_global_indices))
        if num_to_select < len(non_problematic_global_indices):
            random_indices = torch.randperm(len(non_problematic_global_indices))[:num_to_select]
            selected_indices = non_problematic_global_indices[random_indices]
            print(f"  - Selected random {num_to_select} primitives: {num_to_select}/{len(non_problematic_global_indices)}")
        else:
            selected_indices = non_problematic_global_indices
            print(f"  - Selected all non-problematic: {len(non_problematic_global_indices)}/{len(non_problematic_global_indices)}")
        return selected_indices


    
    def _compute_per_pixel_gradient_magnitude(self, 
                                                 x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                                 v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                                 target: torch.Tensor,
                                                 adaptive_config: dict = None) -> torch.Tensor:
        """
        Compute per-pixel gradient magnitude based on absolute x,y gradients.
        Inspired by AbsGS research - measures how much each primitive's position affects the total loss.
        Uses CUDA kernel for parallel processing of all pixels.
        
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
    
        # Get configuration parameters
        if adaptive_config is None:
            adaptive_config = {}
        tile_rows = adaptive_config.get('tile_rows', 4)
        tile_cols = adaptive_config.get('tile_cols', 4)
    
        # Get gradient sampling configuration
        gradient_config = adaptive_config.get('gradient_ranking', {})
        pixels_per_tile = gradient_config.get('pixels_per_tile', 16)
    
        H, W = target.shape[:2]
    
        # Check if CUDA gradient computation is enabled
        if not USE_CUDA_GRADIENT_COMPUTATION:
            print("[Gradient Computation] CUDA disabled, using Python implementation")
            return self._compute_per_pixel_gradient_magnitude_python(x, y, r, v, theta, c, target, adaptive_config)
    
        print(f"[CUDA Gradient Computation] Processing {tile_rows}x{tile_cols} tiles with {pixels_per_tile} pixels per tile")
    
        try:
            # Import CUDA tile rasterizer
            import cuda_tile_rasterizer
        
            # Prepare parameters for CUDA kernel
            means2D = torch.stack([x, y], dim=1).contiguous()  # [N, 2]
        
            # Get primitive templates and selection indices
            primitive_templates = self.primitive_templates if hasattr(self, 'primitive_templates') else torch.zeros((1, 32, 32), device=x.device)
            global_bmp_sel = self.global_bmp_sel if hasattr(self, 'global_bmp_sel') else torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)
        
            # Ensure target is in correct format [H, W, 3]
            if target.dim() == 3 and target.shape[2] == 3:
                target_cuda = target.contiguous()
            else:
                target_cuda = target.unsqueeze(-1).repeat(1, 1, 3).contiguous()
        
            # Call CUDA kernel for parallel gradient computation
            gradient_magnitudes = cuda_tile_rasterizer.compute_per_pixel_gradients(
                means2D,                    # [N, 2] - primitive positions
                r,                          # [N] - primitive radii  
                theta,                      # [N] - primitive rotations
                v,                          # [N] - primitive opacities (logits)
                c,                          # [N, 3] - primitive colors
                primitive_templates,        # [T, Ht, Wt] - template masks
                global_bmp_sel,            # [N] - template selection indices
                target_cuda,               # [H, W, 3] - target image
                pixels_per_tile            # Number of pixels to sample per tile
            )
        
            print(f"[CUDA Gradient Computation] Successfully computed gradients for {gradient_magnitudes.shape[0]} primitives")
            print(f"[CUDA Gradient Computation] Gradient statistics: min={gradient_magnitudes.min().item():.6f}, max={gradient_magnitudes.max().item():.6f}, mean={gradient_magnitudes.mean().item():.6f}")
        
            return gradient_magnitudes
        
        except ImportError:
            print("[Gradient Computation] CUDA tile rasterizer not available, falling back to Python implementation")
            # Fallback to original Python implementation
            return self._compute_per_pixel_gradient_magnitude_python(x, y, r, v, theta, c, target, adaptive_config)
    
        except Exception as e:
            print(f"[Gradient Computation] CUDA implementation failed: {e}, falling back to Python implementation")
            # Fallback to original Python implementation  
            return self._compute_per_pixel_gradient_magnitude_python(x, y, r, v, theta, c, target, adaptive_config)

    def _compute_per_pixel_gradient_magnitude_python(self,
                                                x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                                v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                                target: torch.Tensor,
                                                adaptive_config: dict = None) -> torch.Tensor:
        """
        Original Python implementation as fallback.
        Compute per-pixel gradient magnitude based on absolute x,y gradients.
        Uses fixed-count pixel sampling per tile for performance optimization.
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
    
        print(f"[Python Gradient Computation] Processing {tile_rows}x{tile_cols} tiles with {pixels_per_tile} pixels per tile")
    
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
    
        print(f"[Python Gradient Computation] Processed {total_pixels_processed}/{total_pixels_available} pixels ({actual_sample_ratio:.2%})")
    
        return gradient_magnitudes

    def visualize_gradient_directions_white_background(self,
                                                   x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                                   v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                                   target: torch.Tensor,
                                                   selected_indices: torch.Tensor,
                                                   save_path: str = None,
                                                   non_problematic_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Visualize per-pixel gradient directions comparing problematic vs non-problematic primitives.
        
        This method creates a side-by-side comparison visualization showing gradient directions
        for both problematic and non-problematic primitives on white backgrounds.
        
        Args:
            x, y, r, v, theta, c: Primitive parameters
            target: Target image tensor [H, W, 3]
            selected_indices: Indices of problematic primitives to visualize [K]
            save_path: Optional path to save the visualization
            non_problematic_indices: Optional indices of non-problematic primitives [M]
            
        Returns:
            Visualization image tensor [H, W*2, 3] with gradient directions comparison
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import hsv_to_rgb
        
        # Determine what we're visualizing
        num_problematic = len(selected_indices)
        num_non_problematic = len(non_problematic_indices) if non_problematic_indices is not None else 0
        
        print(f"[Comparison Gradient Visualization] Visualizing {num_problematic} problematic vs {num_non_problematic} non-problematic primitives")
        
        # Ensure x, y parameters require gradients
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        
        # Render current state
        rendered = self.render_from_params(x, y, r, v, theta, c)
        H, W = rendered.shape[:2]
        
        # Compute pixel-wise loss (no reduction)
        pixel_losses = (rendered - target).pow(2).mean(dim=2)  # [H, W]
        
        # Create side-by-side canvases: [H, W*2, 3]
        # Left side: problematic primitives, Right side: non-problematic primitives
        selected_vis_image = np.ones((H, W, 3), dtype=np.float32)  # White background for selected (problematic)
        non_problematic_vis_image = np.ones((H, W, 3), dtype=np.float32)  # White background for non-problematic
        
        # Create gradient direction overlays
        selected_gradient_mask = np.zeros((H, W), dtype=bool)
        non_problematic_gradient_mask = np.zeros((H, W), dtype=bool)
        
        # Process each selected primitive
        for idx, prim_idx in enumerate(selected_indices):
            prim_idx = int(prim_idx.item()) if torch.is_tensor(prim_idx) else int(prim_idx)
            
            # Get primitive parameters
            prim_x = float(x[prim_idx].item())
            prim_y = float(y[prim_idx].item())
            prim_r = float(r[prim_idx].item())
            
            # Define circular region (radius = 1.5 * r)
            radius = 1.5 * prim_r
            
            print(f"[White Background Gradient Visualization] Processing primitive {prim_idx}: center=({prim_x:.1f}, {prim_y:.1f}), radius={radius:.1f}")
            
            # Find pixels within the circular region
            y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            distances = np.sqrt((x_coords - prim_x)**2 + (y_coords - prim_y)**2)
            circle_mask = distances <= radius
            
            # Get pixel coordinates within the circle
            circle_pixels = np.where(circle_mask)
            num_pixels = len(circle_pixels[0])
            
            if num_pixels == 0:
                print(f"[White Background Gradient Visualization] No pixels found for primitive {prim_idx}")
                continue
                
            print(f"[White Background Gradient Visualization] Processing {num_pixels} pixels for primitive {prim_idx}")
            
            # Compute gradients for pixels in this circle
            gradient_directions = np.zeros((num_pixels, 2))  # [num_pixels, 2] for x,y gradients
            
            for pixel_idx, (i, j) in enumerate(zip(circle_pixels[0], circle_pixels[1])):
                pixel_loss = pixel_losses[i, j]
                
                if pixel_loss.requires_grad:
                            # Compute gradients w.r.t. x, y for this pixel
                            grads = torch.autograd.grad(
                                outputs=pixel_loss,
                                inputs=[x, y],
                                retain_graph=True,
                                create_graph=False,
                                allow_unused=True
                            )
                            
                            # Extract gradients for the current primitive
                            if grads[0] is not None and grads[1] is not None:
                                grad_x = float(grads[0][prim_idx].item())
                                grad_y = float(grads[1][prim_idx].item())
                                gradient_directions[pixel_idx] = [grad_x, grad_y]
                    
            # Compute magnitude statistics for this primitive to enable adaptive normalization
            magnitudes = np.sqrt(gradient_directions[:, 0]**2 + gradient_directions[:, 1]**2)
            non_zero_magnitudes = magnitudes[magnitudes > 1e-12]  # Very small threshold for numerical stability
            
            if len(non_zero_magnitudes) > 0:
                # Adaptive normalization based on this primitive's gradient range
                mag_min = np.min(non_zero_magnitudes)
                mag_max = np.max(non_zero_magnitudes)
                mag_median = np.median(non_zero_magnitudes)
                
                print(f"[White Background Gradient Visualization] Primitive {prim_idx} gradient stats: min={mag_min:.2e}, max={mag_max:.2e}, median={mag_median:.2e}")
                
                # Use logarithmic scaling to better visualize small gradients
                log_min = np.log10(mag_min + 1e-12)
                log_max = np.log10(mag_max + 1e-12)
                log_range = log_max - log_min if log_max > log_min else 1.0
                
                # Convert gradient directions to colors and apply to white background
                for pixel_idx, (i, j) in enumerate(zip(circle_pixels[0], circle_pixels[1])):
                    grad_x, grad_y = gradient_directions[pixel_idx]
                    
                    # Compute gradient magnitude and direction
                    magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    
                    if magnitude > 1e-12:  # Much lower threshold for numerical stability only
                        # Compute angle in radians, then convert to [0, 1] for hue
                        angle = np.arctan2(grad_y, grad_x)  # [-π, π]
                        hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]
                        
                        # Enhanced saturation calculation using logarithmic scaling
                        if log_range > 0:
                            # Logarithmic normalization to better show small gradients
                            log_magnitude = np.log10(magnitude + 1e-12)
                            normalized_log_mag = (log_magnitude - log_min) / log_range
                            saturation = np.clip(normalized_log_mag, 0.1, 1.0)  # Minimum 0.1 for visibility
                        else:
                            saturation = 0.5  # Default saturation when all magnitudes are similar
                        
                        # Adaptive value (brightness) based on magnitude relative to median
                        if magnitude >= mag_median:
                            value = 0.9  # High brightness for above-median gradients
                        else:
                            # Scale brightness for below-median gradients (0.4 to 0.8)
                            relative_mag = magnitude / mag_median
                            value = 0.4 + 0.4 * relative_mag
                        
                        # Convert HSV to RGB
                        rgb_color = hsv_to_rgb([hue, saturation, value])
                        
                        # Set color directly on selected primitives canvas
                        selected_vis_image[i, j] = rgb_color
                        selected_gradient_mask[i, j] = True
                    else:
                        # For truly zero gradients, use a very light gray to indicate the region
                        selected_vis_image[i, j] = [0.95, 0.95, 0.95]  # Very light gray
                        selected_gradient_mask[i, j] = True
        
        # Process non-problematic primitives for comparison (if provided and enabled)
        non_problematic_to_process = []
        if ENABLE_NON_PROBLEMATIC_PRIMITIVE_GRADIENT_VISUALIZATION and non_problematic_indices is not None and len(non_problematic_indices) > 0:
            # Convert to list if tensor
            if torch.is_tensor(non_problematic_indices):
                non_problematic_to_process = non_problematic_indices.cpu().numpy().tolist()
            else:
                non_problematic_to_process = list(non_problematic_indices)
                
            print(f"[White Background Gradient Visualization] Processing {len(non_problematic_to_process)} non-problematic primitives: {non_problematic_to_process[:5]}{'...' if len(non_problematic_to_process) > 5 else ''}")
                
            # Process each non-problematic primitive (same logic as selected primitives)
            for idx, prim_idx in enumerate(non_problematic_to_process):
                prim_idx = int(prim_idx)
                
                # Get primitive parameters
                prim_x = float(x[prim_idx].item())
                prim_y = float(y[prim_idx].item())
                prim_r = float(r[prim_idx].item())
                
                # Define circular region (radius = 1.5 * r)
                radius = 1.5 * prim_r
                
                print(f"[White Background Gradient Visualization] Processing non-problematic primitive {prim_idx}: center=({prim_x:.1f}, {prim_y:.1f}), radius={radius:.1f}")
                
                # Find pixels within the circular region
                y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                distances = np.sqrt((x_coords - prim_x)**2 + (y_coords - prim_y)**2)
                circle_mask = distances <= radius
                
                # Get pixel coordinates within the circle
                circle_pixels = np.where(circle_mask)
                num_pixels = len(circle_pixels[0])
                
                if num_pixels == 0:
                    print(f"[White Background Gradient Visualization] No pixels found for non-problematic primitive {prim_idx}")
                    continue
                    
                print(f"[White Background Gradient Visualization] Processing {num_pixels} pixels for non-problematic primitive {prim_idx}")
                
                # Compute gradients for pixels in this circle
                gradient_directions = np.zeros((num_pixels, 2))  # [num_pixels, 2] for x,y gradients
                
                for pixel_idx, (i, j) in enumerate(zip(circle_pixels[0], circle_pixels[1])):
                    pixel_loss = pixel_losses[i, j]
                    
                    if pixel_loss.requires_grad:
                            # Compute gradients w.r.t. x, y for this pixel
                            grads = torch.autograd.grad(
                                outputs=pixel_loss,
                                inputs=[x, y],
                                retain_graph=True,
                                create_graph=False,
                                allow_unused=True
                            )
                            
                            # Extract gradients for the current primitive
                            if grads[0] is not None and grads[1] is not None:
                                grad_x = float(grads[0][prim_idx].item())
                                grad_y = float(grads[1][prim_idx].item())
                                gradient_directions[pixel_idx] = [grad_x, grad_y]
                    
                # Compute magnitude statistics for this primitive to enable adaptive normalization
                magnitudes = np.sqrt(gradient_directions[:, 0]**2 + gradient_directions[:, 1]**2)
                non_zero_magnitudes = magnitudes[magnitudes > 1e-12]  # Very small threshold for numerical stability
                
                if len(non_zero_magnitudes) > 0:
                    # Adaptive normalization based on this primitive's gradient range
                    mag_min = np.min(non_zero_magnitudes)
                    mag_max = np.max(non_zero_magnitudes)
                    mag_median = np.median(non_zero_magnitudes)
                    
                    print(f"[White Background Gradient Visualization] Non-problematic primitive {prim_idx} gradient stats: min={mag_min:.2e}, max={mag_max:.2e}, median={mag_median:.2e}")
                        
                    # Use logarithmic scaling to better visualize small gradients
                    log_min = np.log10(mag_min + 1e-12)
                    log_max = np.log10(mag_max + 1e-12)
                    log_range = log_max - log_min if log_max > log_min else 1.0
                    
                    # Convert gradient directions to colors and apply to white background
                    # Use different color scheme for non-problematic primitives (cooler colors)
                    for pixel_idx, (i, j) in enumerate(zip(circle_pixels[0], circle_pixels[1])):
                        grad_x, grad_y = gradient_directions[pixel_idx]
                        
                        # Compute gradient magnitude and direction
                        magnitude = np.sqrt(grad_x**2 + grad_y**2)
                        
                        if magnitude > 1e-12:  # Much lower threshold for numerical stability only
                            # Compute angle in radians, then convert to [0, 1] for hue
                            angle = np.arctan2(grad_y, grad_x)  # [-π, π]
                            hue = (angle + np.pi) / (2 * np.pi)  # [0, 1] - Same as selected primitives
                            
                            # Enhanced saturation calculation using logarithmic scaling
                            if log_range > 0:
                                # Logarithmic normalization to better show small gradients
                                log_magnitude = np.log10(magnitude + 1e-12)
                                normalized_log_mag = (log_magnitude - log_min) / log_range
                                saturation = np.clip(normalized_log_mag, 0.1, 1.0)  # Minimum 0.1 for visibility
                            else:
                                saturation = 0.5  # Default saturation when all magnitudes are similar
                            
                            # Adaptive value (brightness) based on magnitude relative to median
                            if magnitude >= mag_median:
                                value = 0.9  # High brightness for above-median gradients
                            else:
                                # Scale brightness for below-median gradients (0.4 to 0.8)
                                relative_mag = magnitude / mag_median
                                value = 0.4 + 0.4 * relative_mag
                            
                            # Convert HSV to RGB
                            rgb_color = hsv_to_rgb([hue, saturation, value])
                            
                            # Set color directly on non-problematic primitives canvas
                            non_problematic_vis_image[i, j] = rgb_color
                            non_problematic_gradient_mask[i, j] = True
                        else:
                            # For truly zero gradients, use a very light blue-gray to indicate the region
                            non_problematic_vis_image[i, j] = [0.90, 0.95, 0.95]  # Very light blue-gray
                            non_problematic_gradient_mask[i, j] = True
                else:
                    print(f"[White Background Gradient Visualization] Non-problematic primitive {prim_idx} has no significant gradients (all magnitudes <= 1e-12)")
        
        # Add primitive centers as dots for better contrast on white background
        # Selected primitives: black dots on selected canvas
        for prim_idx in selected_indices:
            prim_idx = int(prim_idx.item()) if torch.is_tensor(prim_idx) else int(prim_idx)
            prim_x = int(x[prim_idx].item())
            prim_y = int(y[prim_idx].item())
            
            # Draw a small black circle at selected primitive center
            if 0 <= prim_x < W and 0 <= prim_y < H:
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if dx*dx + dy*dy <= 9:  # Circle with radius 3
                            nx, ny = prim_x + dx, prim_y + dy
                            if 0 <= nx < W and 0 <= ny < H:
                                selected_vis_image[ny, nx] = [0.0, 0.0, 0.0]  # Black for selected primitives
        
        # Non-problematic primitives: blue dots on non-problematic canvas (if enabled and processed)
        if ENABLE_NON_PROBLEMATIC_PRIMITIVE_GRADIENT_VISUALIZATION and len(non_problematic_to_process) > 0:
            # Use the same non_problematic_to_process that were processed above
            for prim_idx in non_problematic_to_process:
                    prim_x = int(x[prim_idx].item())
                    prim_y = int(y[prim_idx].item())
                    
                    # Draw a small blue circle at non-problematic primitive center
                    if 0 <= prim_x < W and 0 <= prim_y < H:
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                if dx*dx + dy*dy <= 4:  # Circle with radius 2 (smaller than selected)
                                    nx, ny = prim_x + dx, prim_y + dy
                                    if 0 <= nx < W and 0 <= ny < H:
                                        non_problematic_vis_image[ny, nx] = [0.0, 0.0, 1.0]  # Blue for non-problematic primitives
        
        # Convert back to tensor (use selected canvas as main return)
        final_vis_tensor = torch.from_numpy(selected_vis_image).to(rendered.device)
        
        # Save separate visualizations if path provided
        if save_path:
            import os
            
            # Extract directory and filename components
            save_dir = os.path.dirname(save_path)
            filename = os.path.basename(save_path)
            name, ext = os.path.splitext(filename)
            
            # Save selected primitives visualization
            selected_path = os.path.join(save_dir, f"{name}_SELECTED{ext}")
            plt.figure(figsize=(12, 8))
            plt.imshow(selected_vis_image)
            plt.title(f'Gradient Visualization - SELECTED Primitives ({len(selected_indices)} primitives)\n'
                     f'Black dots = primitive centers, Color = gradient direction & magnitude')
            plt.axis('off')
            
            # Add colorbar for gradient directions (full spectrum for selected)
            from matplotlib.patches import Circle
            ax = plt.gca()
            
            # Create a small color wheel in the corner
            wheel_center = (W * 0.9, H * 0.1)
            wheel_radius = min(W, H) * 0.05
            
            angles = np.linspace(0, 2*np.pi, 360)
            for i, angle in enumerate(angles):
                hue = angle / (2 * np.pi)
                color = hsv_to_rgb([hue, 1.0, 0.9])
                
                x_pos = wheel_center[0] + wheel_radius * np.cos(angle)
                y_pos = wheel_center[1] + wheel_radius * np.sin(angle)
                
                circle = Circle((x_pos, y_pos), wheel_radius/20, color=color)
                ax.add_patch(circle)
            
            plt.tight_layout()
            plt.savefig(selected_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[White Background Gradient Visualization] Saved SELECTED primitives to {selected_path}")
            
            # Save non-problematic primitives visualization (if any were processed)
            if ENABLE_NON_PROBLEMATIC_PRIMITIVE_GRADIENT_VISUALIZATION and len(non_problematic_to_process) > 0:
                non_problematic_path = os.path.join(save_dir, f"{name}_NON_PROBLEMATIC{ext}")
                plt.figure(figsize=(12, 8))
                plt.imshow(non_problematic_vis_image)
                plt.title(f'Gradient Visualization - NON-PROBLEMATIC Primitives ({len(non_problematic_to_process)} primitives)\n'
                         f'Blue dots = primitive centers, Color = gradient direction & magnitude')
                plt.axis('off')
                
                # Add colorbar for gradient directions (full spectrum for non-problematic)
                ax = plt.gca()
                
                # Create a small color wheel showing the full color spectrum (same as selected)
                wheel_center = (W * 0.9, H * 0.1)
                wheel_radius = min(W, H) * 0.05
                
                angles = np.linspace(0, 2*np.pi, 360)
                for i, angle in enumerate(angles):
                    hue = angle / (2 * np.pi)
                    color = hsv_to_rgb([hue, 1.0, 0.9])
                    
                    x_pos = wheel_center[0] + wheel_radius * np.cos(angle)
                    y_pos = wheel_center[1] + wheel_radius * np.sin(angle)
                    
                    circle = Circle((x_pos, y_pos), wheel_radius/20, color=color)
                    ax.add_patch(circle)
                
                plt.tight_layout()
                plt.savefig(non_problematic_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"[White Background Gradient Visualization] Saved NON-PROBLEMATIC primitives to {non_problematic_path}")
        
        print(f"[White Background Gradient Visualization] Completed visualization for {len(selected_indices)} primitives")
        return final_vis_tensor

    def debug_adaptive_control_with_visualization(self,
                                                   x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                                   v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                                   target_image: torch.Tensor,
                                                   adaptive_config: dict = None,
                                                   save_dir: str = "./debug_gradients") -> tuple:
        """
        Apply adaptive control with gradient visualization for debugging gradient collisions.
        
        This method:
        1. Applies normal adaptive control to identify problematic primitives
        2. Visualizes gradient directions for selected primitives
        3. Saves visualization images for analysis
        
        Args:
            x, y, r, v, theta, c: Current primitive parameters
            target_image: Target image tensor [H, W, 3]
            adaptive_config: Configuration dict for adaptive control
            save_dir: Directory to save visualization images
            
        Returns:
            Tuple of (adapted_parameters, selected_indices_per_tile)
        """
        import os
        from datetime import datetime
        
        if adaptive_config is None or not adaptive_config.get('enabled', False):
            return (x, y, r, v, theta, c), []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract configuration parameters
        tile_rows = adaptive_config.get('tile_rows', 4)
        tile_cols = adaptive_config.get('tile_cols', 4)
        scale_threshold = adaptive_config.get('scale_threshold', 8.0)
        opacity_threshold = adaptive_config.get('opacity_threshold', 0.7)
        max_primitives_per_tile = adaptive_config.get('max_primitives_per_tile', 3)

        
        # Extract gradient_ranking configuration
        gradient_ranking_config = adaptive_config.get('gradient_ranking', {})
        gradient_ranking_enabled = gradient_ranking_config.get('enabled', True)
        
        print(f"[Debug Adaptive Control] Starting debug with gradient visualization")
        print(f"[Debug Adaptive Control] Tile grid: {tile_rows}x{tile_cols}, Gradient ranking enabled: {gradient_ranking_enabled}")
        
        # Create new leaf tensors
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
        
        # Compute gradient magnitudes for ranking
        gradient_magnitudes = self._compute_per_pixel_gradient_magnitude(
            x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted, target_image, adaptive_config
        )
        
        # Track selected primitives for each tile
        all_selected_indices = []
        tile_info = []
        
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
                
                # Find problematic primitives
                problematic_indices = self._find_problematic_primitives(
                    tile_indices, x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted,
                    scale_threshold, opacity_threshold, gradient_magnitudes, max_primitives_per_tile,
                    gradient_ranking_enabled
                )
                
                # Find non-problematic primitives for comparison
                non_problematic_indices = self._find_non_problematic_primitives(
                    tile_indices, x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted,
                    scale_threshold, opacity_threshold, max_primitives_per_tile
                )
                
                if len(problematic_indices) > 0 or len(non_problematic_indices) > 0:
                    all_selected_indices.extend(problematic_indices.tolist())
                    tile_info.append({
                        'row': row, 'col': col,
                        'bounds': (x_min, y_min, x_max, y_max),
                        'selected_indices': problematic_indices.tolist(),
                        'non_problematic_indices': non_problematic_indices.tolist(),
                        'total_primitives': len(tile_indices)
                    })
                    
                    print(f"[Debug Adaptive Control] Tile ({row},{col}): {len(problematic_indices)}/{len(tile_indices)} problematic, {len(non_problematic_indices)}/{len(tile_indices)} non-problematic")
                    
                    # Visualize gradients for this tile's selected primitives
                    if len(problematic_indices) > 0 or len(non_problematic_indices) > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # White background visualization with non-problematic comparison
                        vis_path_white = os.path.join(save_dir, f"gradient_vis_white_tile_{row}_{col}_{timestamp}.png")
                        self.visualize_gradient_directions_white_background(
                            x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted,
                            target_image, problematic_indices, vis_path_white, non_problematic_indices
                        )
                
                # Apply opacity reduction (same as original adaptive control)
                if len(problematic_indices) > 0:
                    opacity_reduction_factor = adaptive_config.get('opacity_reduction_factor', 0.5)
                    with torch.no_grad():
                        v_adapted[problematic_indices] *= opacity_reduction_factor
        
        # Create overall visualization with all selected primitives
        if len(all_selected_indices) > 0:
            all_selected_tensor = torch.tensor(all_selected_indices, device=x.device)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # New white background visualization (cleaner view)
            overall_vis_path_white = os.path.join(save_dir, f"gradient_vis_white_all_selected_{timestamp}.png")
            self.visualize_gradient_directions_white_background(
                x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted,
                target_image, all_selected_tensor, overall_vis_path_white
            )
            
            print(f"[Debug Adaptive Control] Total selected primitives: {len(all_selected_indices)}")
            print(f"[Debug Adaptive Control] Saved visualizations to {save_dir}")
        else:
            print(f"[Debug Adaptive Control] No primitives selected for opacity reduction")
        
        return (x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted), tile_info

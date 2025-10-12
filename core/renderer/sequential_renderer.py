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
GRADIENT_DEBUG_SAVE_DIR = "./outputs/vis_class/debug_gradients_sequential"

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
    
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None, tile_size=32, sigma = 1.0):
        # Pass parameters to SimpleTileRenderer using keyword arguments
        super().__init__(canvas_size, S, tile_size=tile_size, 
                        alpha_upper_bound=alpha_upper_bound, device=device, 
                        use_fp16=use_fp16, gamma=gamma, output_path=output_path,
                        sigma = sigma)
    
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
            # Simple learning rate format (constants already merged in config)
            return {
                'default_lr': lr_config,
                'gain_x': 1.0, 'gain_y': 1.0, 'gain_r': 1.0,
                'gain_v': 1.0, 'gain_theta': 1.0, 'gain_c': 1.0,
                'decay_rate': opt_conf['decay_rate']
            }
        else:
            # Complex learning rate format with gains (constants already merged in config)
            return {
                'default_lr': lr_config.get('default', 0.005),
                'gain_x': lr_config['gain_x'],
                'gain_y': lr_config['gain_y'],
                'gain_r': lr_config['gain_r'],
                'gain_v': lr_config['gain_v'],
                'gain_theta': lr_config['gain_theta'],
                'gain_c': lr_config.get('gain_c', 1.0),
                'decay_rate': opt_conf['decay_rate']
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
        # Read explicit epochs at which to apply adaptive control
        # Expect a list of iteration indices (0-based) in config under 'apply_epochs'
        apply_epochs = set(adaptive_config.get('apply_epochs', []))
        
        pbar = tqdm(range(num_iter), desc="Optimizing with warmup scheduling")
        
        print(f"Starting tile-based optimization for sequential frames, {num_iter} iterations...")

        # Main optimization loop
        for i in pbar:
            # Apply adaptive control at specified epochs if enabled
            if (adaptive_config.get('enabled', False) and i in apply_epochs):
                
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
            if (adaptive_config.get('enabled', False) and i in apply_epochs):
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
        if gradient_ranking_enabled:
            gradient_magnitudes = self._compute_per_pixel_gradient_magnitude(
                x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted, target_image, adaptive_config
            )
        else:
            gradient_magnitudes = None
        
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
        When gradient_ranking.process_all_pixels is True, processes all pixels.
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
        # Support both new and legacy config keys
        process_all_pixels = gradient_config.get('process_all_pixels', False)
        pixels_per_tile = gradient_config.get('pixels_per_tile', 16)
    
        # Calculate tile dimensions (same as apply_adaptive_control)
        tile_height = H // tile_rows
        tile_width = W // tile_cols
    
        # Initialize gradient magnitude accumulator
        gradient_magnitudes = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    
        total_pixels_processed = 0
        total_pixels_available = H * W
    
        mode_msg = "all pixels" if process_all_pixels else f"{pixels_per_tile} pixels per tile"
        print(f"[Python Gradient Computation] Processing {tile_rows}x{tile_cols} tiles with {mode_msg}")
    
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
    
                num_tile_pixels = len(tile_pixels)
                if num_tile_pixels == 0:
                    continue
    
                # Decide which pixels to process
                if process_all_pixels:
                    sampled_pixels = tile_pixels
                else:
                    num_sample_pixels = min(pixels_per_tile, num_tile_pixels)  # Don't exceed available pixels
                    if num_sample_pixels > 0:
                        if num_sample_pixels < num_tile_pixels:
                            sample_indices = torch.randperm(num_tile_pixels)[:num_sample_pixels]
                            sampled_pixels = [tile_pixels[idx] for idx in sample_indices]
                        else:
                            # Use all pixels if requested count >= available pixels
                            sampled_pixels = tile_pixels
                    else:
                        sampled_pixels = []
    
                # Process selected pixels in this tile
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
        # Avoid divide-by-zero; if processing all pixels, ratio will be 1.0
        if actual_sample_ratio > 0:
            gradient_magnitudes = gradient_magnitudes / actual_sample_ratio
    
        print(f"[Python Gradient Computation] Processed {total_pixels_processed}/{total_pixels_available} pixels ({actual_sample_ratio:.2%})")
    
        return gradient_magnitudes


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
        2. Visualizes gradient directions for selected primitives using GradientVisualizer
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
        from util.gradient_visualizer import GradientVisualizer
        
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
        
        print(f"[Debug Adaptive Control] Starting debug with gradient visualization using GradientVisualizer")
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
        
        # Create GradientVisualizer instances once for reuse across tiles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualizer for problematic primitives (warm colors, red dots)
        vis_path_problematic_base = os.path.join(save_dir, f"gradient_problematic_{timestamp}")
        visualizer_problematic = GradientVisualizer(
            target_image=target_image,
            save_path=vis_path_problematic_base,
            color_spectrum="warm",  # Use warm colors for problematic primitives
            background_color=(1.0, 1.0, 1.0),  # White background
            primitive_radius_multiplier=1.5,
            enable_logging=True
        )
        
        # Create visualizer for non-problematic primitives (cool colors, blue dots)
        vis_path_non_problematic_base = os.path.join(save_dir, f"gradient_non_problematic_{timestamp}")
        visualizer_non_problematic = GradientVisualizer(
            target_image=target_image,
            save_path=vis_path_non_problematic_base,
            color_spectrum="cool",  # Use cool colors for non-problematic primitives
            background_color=(1.0, 1.0, 1.0),  # White background
            primitive_radius_multiplier=1.5,
            enable_logging=True
        )
        
        # Track selected primitives for each tile
        all_selected_indices = []
        all_non_problematic_indices = []
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
                    all_non_problematic_indices.extend(non_problematic_indices.tolist())
                    tile_info.append({
                        'row': row, 'col': col,
                        'bounds': (x_min, y_min, x_max, y_max),
                        'selected_indices': problematic_indices.tolist(),
                        'non_problematic_indices': non_problematic_indices.tolist(),
                        'total_primitives': len(tile_indices)
                    })
                    
                    print(f"[Debug Adaptive Control] Tile ({row},{col}): {len(problematic_indices)}/{len(tile_indices)} problematic, {len(non_problematic_indices)}/{len(tile_indices)} non-problematic")
                    
                    # Visualize gradients for this tile's selected primitives using pre-created visualizers
                    if len(problematic_indices) > 0:
                        # Update save path for this specific tile
                        visualizer_problematic.save_path = os.path.join(save_dir, f"gradient_problematic_tile_{row}_{col}_{timestamp}")
                        
                        # Visualize problematic primitives
                        _, saved_path_problematic = visualizer_problematic.visualize_gradients(
                            renderer=self,
                            x=x_adapted, y=y_adapted, r=r_adapted,
                            v=v_adapted, theta=theta_adapted, c=c_adapted,
                            primitive_indices=problematic_indices,
                            suffix="problematic",
                            title_prefix=f"Problematic Primitives - Tile ({row},{col})",
                            center_dot_color=(1.0, 0.0, 0.0)  # Red dots for problematic
                        )
                        
                    # Visualize non-problematic primitives for comparison
                    if len(non_problematic_indices) > 0:
                        # Update save path for this specific tile
                        visualizer_non_problematic.save_path = os.path.join(save_dir, f"gradient_non_problematic_tile_{row}_{col}_{timestamp}")
                        
                        # Visualize non-problematic primitives
                        _, saved_path_non_problematic = visualizer_non_problematic.visualize_gradients(
                            renderer=self,
                            x=x_adapted, y=y_adapted, r=r_adapted,
                            v=v_adapted, theta=theta_adapted, c=c_adapted,
                            primitive_indices=non_problematic_indices,
                            suffix="non_problematic",
                            title_prefix=f"Non-Problematic Primitives - Tile ({row},{col})",
                            center_dot_color=(0.0, 0.0, 1.0)  # Blue dots for non-problematic
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
            
            # Overall problematic primitives visualization
            overall_vis_path = os.path.join(save_dir, f"gradient_all_problematic_{timestamp}")
            overall_visualizer = GradientVisualizer(
                target_image=target_image,
                save_path=overall_vis_path,
                color_spectrum="full",  # Use full spectrum for overall view
                background_color=(1.0, 1.0, 1.0),  # White background
                primitive_radius_multiplier=1.5,
                enable_logging=True
            )
            
            # Visualize all problematic primitives
            _, saved_path_overall = overall_visualizer.visualize_gradients(
                renderer=self,
                x=x_adapted, y=y_adapted, r=r_adapted,
                v=v_adapted, theta=theta_adapted, c=c_adapted,
                primitive_indices=all_selected_tensor,
                suffix="all_problematic",
                title_prefix="All Problematic Primitives",
                center_dot_color=(0.0, 0.0, 0.0)  # Black dots for overall view
            )
            
            print(f"[Debug Adaptive Control] Total selected primitives: {len(all_selected_indices)}")
            print(f"[Debug Adaptive Control] Saved visualizations to {save_dir}")
        else:
            print(f"[Debug Adaptive Control] No primitives selected for opacity reduction")
        
        return (x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted), tile_info

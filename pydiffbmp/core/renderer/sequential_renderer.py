import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
try:
    from torch.amp import autocast, GradScaler
    AUTOCAST_NEW_API = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AUTOCAST_NEW_API = False
from .simple_tile_renderer import SimpleTileRenderer

# Import necessary functions for statistics
from .simple_tile_renderer import vram_used_by_pid
try:
    from cuda_tile_rasterizer import print_cuda_timing_stats, print_cuda_timing_stats_fp16
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False



class SequentialFrameRenderer(SimpleTileRenderer):
    """
    Specialized renderer for subsequent frames in sequential optimization.
    Inherits from SimpleTileRenderer and uses warmup scheduling for loss.
    """
    
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None, tile_size=32, sigma = 1.0, c_blend=0.0, primitive_colors: torch.Tensor = None, max_prims_per_pixel: int = 1, debug_config: dict = None):
        # Pass parameters to SimpleTileRenderer using keyword arguments
        super().__init__(canvas_size, S, tile_size=tile_size, 
                        alpha_upper_bound=alpha_upper_bound, device=device, 
                        use_fp16=use_fp16, gamma=gamma, output_path=output_path,
                        sigma = sigma, c_blend=c_blend, primitive_colors=primitive_colors, max_prims_per_pixel=max_prims_per_pixel)
        
        # Store debug configuration (defaults are applied by constants.py)
        if debug_config is None:
            raise ValueError("Debug config is required for SequentialFrameRenderer")
        self.debug_config = debug_config
    
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
                                        opt_conf: dict = None,
                                        previous_target: torch.Tensor = None
                                        ) -> tuple:
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
        
        # Initialize loss composer from config
        from pydiffbmp.util.loss_functions import LossComposer
        loss_config = opt_conf.get("loss_config", {"type": "mse"})
        self.loss_composer = LossComposer(loss_config, device=self.device)
        print(f"Using loss configuration: {loss_config}")
        
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
        
        # Mixed-precision scaler (only used if use_fp16 is True)
        # PyTorch 2.0+: GradScaler('cuda'), PyTorch 1.x: GradScaler()
        if self.use_fp16:
            scaler = GradScaler('cuda') if AUTOCAST_NEW_API else GradScaler()
        else:
            scaler = None
        
        num_iter = opt_conf.get("num_iterations", opt_conf.get("num_iter", 50))
        adaptive_config = opt_conf.get('adaptive_control', {})
        # Read explicit epochs at which to apply adaptive control
        # Expect a list of iteration indices (0-based) in config under 'apply_epochs'
        apply_epochs = set(adaptive_config.get('apply_epochs', []))
        
        # Extract selective parameter optimization config
        selective_parameter_optimization_config = opt_conf.get('selective_parameter_optimization', {})
        
        # Compute diff_mask once for selective parameter optimization
        diff_mask_for_freeze = None
        if selective_parameter_optimization_config.get('enabled', False) and previous_target is not None:
            with torch.no_grad():
                # Compute absolute difference between target and next_target
                diff = torch.abs(target - previous_target)
                diff_magnitude = torch.mean(diff, dim=2)  # [H, W]
                
                # Create binary mask of changed regions (threshold at 0.1 for robustness)
                diff_magnitude_threshold = selective_parameter_optimization_config.get('diff_magnitude_threshold', 0.1)
                #print(f"Diff magnitude threshold: {diff_magnitude_threshold}")
                diff_mask_for_freeze = diff_magnitude > diff_magnitude_threshold  # [H, W]
                
                # DIFF MASK DEBUG VISUALIZATION
                # Save diff mask for visualization
                if self.debug_config.get("diff_mask", {}).get("enabled", False):
                    import os
                    from torchvision.utils import save_image
                    
                    # Create timestamped subfolder if not already created
                    if not hasattr(self, '_diff_mask_timestamp'):
                        from datetime import datetime
                        self._diff_mask_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    diff_mask_base_path = self.debug_config.get("diff_mask", {}).get("export_path", "./outputs/vis_class/diff_mask_sequential")
                    diff_mask_export_path = os.path.join(diff_mask_base_path, f"diff_masks_{self._diff_mask_timestamp}")
                    os.makedirs(diff_mask_export_path, exist_ok=True)
                    
                    # Use frame index from self if available, otherwise use a counter
                    frame_idx = getattr(self, 'current_frame_idx', 0)
                    diff_mask_path = os.path.join(diff_mask_export_path, f"diff_mask_frame_{frame_idx:04d}.png")
                    # Convert boolean mask to float and add batch/channel dimensions for save_image
                    diff_mask_float = diff_mask_for_freeze.float().unsqueeze(0)  # [1, H, W]
                    save_image(diff_mask_float, diff_mask_path)
                    print(f"[Selective Freeze] Saved diff mask to: {diff_mask_path}")
        
        pbar = tqdm(range(num_iter), desc="Optimizing with warmup scheduling")
        
        print(f"Starting tile-based optimization for sequential frames, {num_iter} iterations...")

        # Main optimization loop
        for i in pbar:
            # Select primitives to freeze based on diff mask
            freeze_mask = None
            if selective_parameter_optimization_config.get("enabled", False):
                freeze_mask = self.select_primitives_2_freeze(x, y, r, theta, diff_mask_for_freeze, selective_parameter_optimization_config)

            # Apply adaptive control at specified epochs if enabled
            if (adaptive_config.get('enabled', False) and i in apply_epochs):
                
                # Choose between debug visualization or normal adaptive control
                if self.debug_config.get("gradient_visualization", {}).get("enabled", False):
                    # Use debug version with gradient visualization
                    import os
                    from datetime import datetime
                    
                    # Create timestamped subdirectory for this iteration
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    gradient_debug_save_dir = self.debug_config.get("gradient_visualization", {}).get("save_dir", "./outputs/vis_class/debug_gradients_sequential")
                    iteration_save_dir = os.path.join(gradient_debug_save_dir, f"iter_{i:04d}_{timestamp}")
                    
                    print(f"\n[Debug Mode] Applying adaptive control with gradient visualization at iteration {i}")
                    print(f"[Debug Mode] Saving visualizations to: {iteration_save_dir}")
                    
                    # Apply debug adaptive control with visualization
                    (x, y, r, v, theta, c), tile_info = self.debug_adaptive_control_with_visualization(
                        x, y, r, v, theta, c, target, adaptive_config, freeze_mask, iteration_save_dir
                    )
                    
                    # Log debug information
                    total_selected = sum(len(tile['selected_indices']) for tile in tile_info)
                    print(f"[Debug Mode] Iteration {i}: {total_selected} primitives selected across {len(tile_info)} tiles")
                    
                else:
                    # Use normal adaptive control without visualization
                    x, y, r, v, theta, c = self.apply_adaptive_control(
                        x, y, r, v, theta, c, target, adaptive_config, freeze_mask)
                
                # Update optimizer parameters with new tensors
                optimizer.param_groups[0]['params'] = [x]
                optimizer.param_groups[1]['params'] = [y]
                optimizer.param_groups[2]['params'] = [r]
                optimizer.param_groups[3]['params'] = [v]
                optimizer.param_groups[4]['params'] = [theta]
                optimizer.param_groups[5]['params'] = [c]
            
            optimizer.zero_grad()
            
            # Generate rendered image using tile-based rendering with FP16 support
            I_bg = torch.ones((self.H, self.W, 3), device=self.device)
            
            if self.use_fp16:
                # FP16 path with autocast and GradScaler
                autocast_ctx = autocast(device_type='cuda') if AUTOCAST_NEW_API else autocast()
                with autocast_ctx:
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
                
                # Scale the loss and call backward
                scaler.scale(loss).backward()
                
                # Zero out gradients for frozen primitives if needed
                if selective_parameter_optimization_config.get("enabled", False):
                    with torch.no_grad():
                        x.grad[freeze_mask] = 0
                        y.grad[freeze_mask] = 0
                        r.grad[freeze_mask] = 0
                        v.grad[freeze_mask] = 0
                        theta.grad[freeze_mask] = 0
                        c.grad[freeze_mask] = 0
                
                # Update optimizer with gradient scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # FP32 path (original code)
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
                
                # Zero out gradients for frozen primitives if needed
                if selective_parameter_optimization_config.get("enabled", False):
                    with torch.no_grad():
                        x.grad[freeze_mask] = 0
                        y.grad[freeze_mask] = 0
                        r.grad[freeze_mask] = 0
                        v.grad[freeze_mask] = 0
                        theta.grad[freeze_mask] = 0
                        c.grad[freeze_mask] = 0
                
                optimizer.step()
            
            scheduler.step()
            
            # Update progress bar
            postfix = {'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'}
            if (adaptive_config.get('enabled', False) and i in apply_epochs):
                postfix['adaptive'] = 'applied'
            pbar.set_postfix(postfix)
        
        print(f"Tile-based optimization completed. Final loss: {loss.item():.6f}")
        
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
    

    def select_primitives_2_freeze(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   r: torch.Tensor,
                                   theta: torch.Tensor,
                                   diff_mask: torch.Tensor,
                                   config: dict) -> torch.Tensor:
        """
        Select primitives to freeze based on pre-computed diff_mask.
        
        Two modes available:
        - tight_freezemask=True: Uses bounding box overlap detection (more accurate, accounts for scale/rotation)
        - tight_freezemask=False: Uses legacy center-based distance check (simpler, faster)
        
        Args:
            x, y: Primitive position parameters [N]
            r: Primitive scale parameters [N]
            theta: Primitive rotation parameters [N]
            diff_mask: Pre-computed boolean mask of changed regions [H, W]
            config: Configuration dict with 'freeze_distance_threshold' and 'tight_freezemask' parameters
            
        Returns:
            freeze_mask: Boolean tensor [N] where True indicates primitive should be frozen
        """
        # Extract configuration
        freeze_distance_threshold = config.get('freeze_distance_threshold', 12.0)
        tight_freezemask = config.get('tight_freezemask', True)
        gradual_freeze_config = config.get('gradual_freeze', {})
        use_gradual_freeze = gradual_freeze_config.get('enabled', False)
        
        with torch.no_grad():
            N = len(x)
            
            # Find coordinates of all changed pixels
            changed_coords = torch.nonzero(diff_mask, as_tuple=False)  # [M, 2] where M is number of changed pixels
            
            if changed_coords.shape[0] == 0:
                # No changes detected, freeze all primitives
                print("[Selective Freeze] No changes detected between frames, freezing all primitives")
                return torch.ones(N, dtype=torch.bool, device=x.device)
            
            if tight_freezemask:
                # NEW: Bounding box-based overlap detection (more accurate)
                #print("[Selective Freeze] Using tight bounding box overlap detection")
                
                # Use shared helper to compute world-space bounding boxes
                bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = self._compute_primitive_world_bboxes(x, y, r, theta)
                
                # Add distance threshold margin to bounding boxes
                margin = freeze_distance_threshold * self.W
                bbox_x_min = bbox_x_min - margin  # (N,)
                bbox_x_max = bbox_x_max + margin  # (N,)
                bbox_y_min = bbox_y_min - margin  # (N,)
                bbox_y_max = bbox_y_max + margin  # (N,)
                
                # Extract changed pixel coordinates
                # changed_coords is [M, 2] in (y, x) format
                changed_y = changed_coords[:, 0].float()  # (M,)
                changed_x = changed_coords[:, 1].float()  # (M,)
                
                # Check if any changed pixel is within each primitive's bounding box
                # Using broadcasting: (N, 1) and (1, M)
                bbox_x_min_exp = bbox_x_min.unsqueeze(1)  # (N, 1)
                bbox_x_max_exp = bbox_x_max.unsqueeze(1)  # (N, 1)
                bbox_y_min_exp = bbox_y_min.unsqueeze(1)  # (N, 1)
                bbox_y_max_exp = bbox_y_max.unsqueeze(1)  # (N, 1)
                
                changed_x_exp = changed_x.unsqueeze(0)  # (1, M)
                changed_y_exp = changed_y.unsqueeze(0)  # (1, M)
                
                # Check if each changed pixel is inside each primitive's bounding box
                x_inside = (changed_x_exp >= bbox_x_min_exp) & (changed_x_exp <= bbox_x_max_exp)  # (N, M)
                y_inside = (changed_y_exp >= bbox_y_min_exp) & (changed_y_exp <= bbox_y_max_exp)  # (N, M)
                
                # Both x and y must be inside for a pixel to be inside the bounding box
                inside_bbox = x_inside & y_inside  # (N, M)
                
                # A primitive overlaps with the diff_mask if ANY changed pixel is inside its bbox
                has_overlap = inside_bbox.any(dim=1)  # (N,)
                
                # Create freeze mask: freeze primitives that do NOT overlap with changed regions
                freeze_mask = ~has_overlap
                
                # Log statistics
                num_frozen = torch.sum(freeze_mask).item()
                num_optimized = N - num_frozen
                # print(f"[Selective Freeze] Freeze threshold: {freeze_distance_threshold:.4f} (margin: {margin:.1f}px)")
                # print(f"[Selective Freeze] Changed pixels: {changed_coords.shape[0]}")
                # print(f"[Selective Freeze] Frozen primitives: {num_frozen}/{N} ({100*num_frozen/N:.1f}%)")
                # print(f"[Selective Freeze] Optimized primitives: {num_optimized}/{N} ({100*num_optimized/N:.1f}%)")
                
            else:
                # LEGACY: Center-based distance check (original implementation)
                #print("[Selective Freeze] Using legacy center-based distance check")
                
                # Convert primitive positions to pixel coordinates (already in pixel space)
                prim_x_pixel = x  # Already in pixel coordinates
                prim_y_pixel = y  # Already in pixel coordinates
                
                # Stack primitive coordinates [N, 2]
                prim_coords = torch.stack([prim_y_pixel, prim_x_pixel], dim=1)  # [N, 2] (y, x order to match image)
                
                # Compute distance from each primitive to all changed pixels
                # Using broadcasting: [N, 1, 2] - [1, M, 2] = [N, M, 2]
                distances = torch.cdist(prim_coords.float(), changed_coords.float())  # [N, M]
                
                # Find minimum distance from each primitive to any changed pixel
                min_distances, _ = torch.min(distances, dim=1)  # [N]
                
                # Create freeze mask: freeze primitives that are FARTHER than threshold from changes
                # (i.e., keep optimizing primitives that are close to changed regions)
                freeze_mask = min_distances > freeze_distance_threshold * self.W
                
                # Log statistics
                num_frozen = torch.sum(freeze_mask).item()
                num_optimized = N - num_frozen
                # print(f"[Selective Freeze] Freeze threshold: {freeze_distance_threshold:.1f}px")
                # print(f"[Selective Freeze] Changed pixels: {changed_coords.shape[0]}")
                # print(f"[Selective Freeze] Frozen primitives: {num_frozen}/{N} ({100*num_frozen/N:.1f}%)")
                # print(f"[Selective Freeze] Optimized primitives: {num_optimized}/{N} ({100*num_optimized/N:.1f}%)")
            
            # Optional: gradual freeze based on distance
            if use_gradual_freeze:
                # Not implemented in this version - would require gradient scaling instead of masking
                print("[Selective Freeze] Note: gradual_freeze is enabled but not yet implemented")
        
        return freeze_mask


    def apply_adaptive_control(self, 
                             x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, 
                             v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                             target_image: torch.Tensor,
                             adaptive_config: dict = None,
                             freeze_mask: torch.Tensor = None) -> tuple:
        """
        Apply adaptive control to reduce artifacts by lowering opacity of problematic primitives.
        
        Args:
            x, y, r, v, theta, c: Current primitive parameters
            target_image: Target image tensor [H, W, 3] for pixel-based color_nerf
            adaptive_config: Configuration dict for adaptive control
            
        Returns:
            Tuple of adapted parameters (x, y, r, v, theta, c)
        """

        #print("DEBUG: apply_adaptive_control")


        if adaptive_config is None or not adaptive_config.get('enabled', False):
            return x, y, r, v, theta, c
        
        # Extract configuration parameters
        tile_rows = adaptive_config.get('tile_rows', 4)
        tile_cols = adaptive_config.get('tile_cols', 4)
        scale_threshold = adaptive_config.get('scale_threshold', 8.0)
        opacity_threshold = adaptive_config.get('opacity_threshold', 0.7)
        opacity_reduction_factor = adaptive_config.get('opacity_reduction_factor', 0.5)
        max_primitives_per_tile = adaptive_config.get('max_primitives_per_tile', 3)
        min_criteria_count = adaptive_config.get('min_criteria_count', 2)
        front_primitives_percentile = adaptive_config.get('front_primitives_percentile', 0.7)
        
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
                    scale_threshold, opacity_threshold, max_primitives_per_tile,
                    freeze_mask, min_criteria_count, front_primitives_percentile
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
                           max_primitives_per_tile: int = 16,
                           freeze_mask: torch.Tensor = None,
                           min_criteria_count: int = 2,
                           front_primitives_percentile: float = 0.7) -> torch.Tensor:
        """
        Find problematic primitives within a tile based on three criteria,
        then select top-k by scale*opacity.
        
        Args:
            tile_indices: Indices of primitives in this tile
            x, y, r, v, theta, c: Primitive parameters
            scale_threshold: Threshold for large scale primitives
            opacity_threshold: Threshold for high opacity primitives
            max_primitives_per_tile: Maximum number of primitives to select
            freeze_mask: Optional mask indicating which primitives should be excluded from adaptive control
            min_criteria_count: Minimum number of criteria a primitive must meet to be considered problematic
            front_primitives_percentile: Percentile threshold for front primitives (z-order)
        
        Returns:
            Indices of problematic primitives (top-k by scale*opacity)
        """
        if len(tile_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=tile_indices.device)
        
        # Extract parameters for primitives in this tile
        tile_r = r[tile_indices]
        tile_v = v[tile_indices]
        tile_opacity = self.alpha_upper_bound*torch.sigmoid(tile_v)
        
        # Criterion 1: Large scale primitives
        large_scale_mask = tile_r >= scale_threshold*self.W
        
        # Criterion 2: High opacity primitives
        high_opacity_mask = tile_opacity >= opacity_threshold
        
        # Criterion 3: Front primitives (those with higher z-order)
        # In splatting, higher array indices are rendered later, thus appearing in front
        # We consider primitives above the percentile threshold as "front"
        if len(tile_indices) > 0:
            tile_indices_sorted = torch.sort(tile_indices)[0]  # Sort tile indices
            front_threshold_idx = int(len(tile_indices_sorted) * front_primitives_percentile)
            front_indices_threshold = tile_indices_sorted[front_threshold_idx] if front_threshold_idx < len(tile_indices_sorted) else tile_indices_sorted[-1]
            
            # Create mask for primitives that are in the front (high z-order) - vectorized
            front_mask = tile_indices >= front_indices_threshold
        else:
            front_mask = torch.tensor([], dtype=torch.bool, device=tile_indices.device)
        
        # Calculate actual percentage for display
        front_percentage = int((1.0 - front_primitives_percentile) * 100)

        # # Debug: Print criteria counts
        # print(f"[Adaptive Control Debug] Tile criteria counts:")
        # print(f"  - Large scale (>={scale_threshold}): {large_scale_mask.sum().item()}/{len(tile_indices)}")
        # print(f"  - High opacity (>={opacity_threshold:.2f}): {high_opacity_mask.sum().item()}/{len(tile_indices)}")
        # print(f"  - Front primitives (top {front_percentage}%): {front_mask.sum().item()}/{len(tile_indices)}")
        
        # Combine criteria: primitives meeting minimum criteria count
        criteria_count = (large_scale_mask.float() + high_opacity_mask.float() + front_mask.float())
        problematic_mask = criteria_count >= float(min_criteria_count)
        
        # Exclude frozen primitives from adaptive control candidates
        if freeze_mask is not None:
            # Extract freeze status for primitives in this tile
            tile_freeze_mask = freeze_mask[tile_indices]
            # Keep only primitives that are problematic AND not frozen
            problematic_mask = problematic_mask & (~tile_freeze_mask)
            # print(f"  - Problematic by criteria (>={min_criteria_count}/3): {(criteria_count >= float(min_criteria_count)).sum().item()}/{len(tile_indices)}")
        
        # Get indices of problematic primitives within the tile
        problematic_tile_indices = torch.where(problematic_mask)[0]
        problematic_global_indices = tile_indices[problematic_tile_indices]
        
        # Select top-k primitives based on scale*opacity ranking
        num_to_select = min(max_primitives_per_tile, len(problematic_global_indices))
        if num_to_select < len(problematic_global_indices):
            # Rank by scale*opacity score
            scores = r[problematic_global_indices] * torch.sigmoid(v[problematic_global_indices])
            _, top_indices = torch.topk(scores, num_to_select)
            selected_indices = problematic_global_indices[top_indices]
        else:
            selected_indices = problematic_global_indices
        
        return selected_indices
    
    def _find_non_problematic_primitives(self, 
                               tile_indices: torch.Tensor,
                               x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                               v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                               scale_threshold: float, opacity_threshold: float,
                               max_primitives_per_tile: int = 16,
                               min_criteria_count: int = 3,
                               back_primitives_percentile: float = 0.3) -> torch.Tensor:
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
        small_scale_mask = tile_r < scale_threshold*self.W
        
        # Criterion 2: Low opacity primitives (opposite of high opacity)
        low_opacity_mask = tile_opacity < opacity_threshold
        
        # Criterion 3: Back primitives (those with lower z-order, opposite of front)
        # In splatting, lower array indices are rendered earlier, thus appearing in back
        # We consider primitives below the percentile threshold as "back"
        if len(tile_indices) > 0:
            tile_indices_sorted = torch.sort(tile_indices)[0]  # Sort tile indices
            back_threshold_idx = int(len(tile_indices_sorted) * back_primitives_percentile)
            back_indices_threshold = tile_indices_sorted[back_threshold_idx] if back_threshold_idx < len(tile_indices_sorted) else tile_indices_sorted[0]
            
            # Create mask for primitives that are in the back (low z-order) - vectorized
            back_mask = tile_indices <= back_indices_threshold
        else:
            back_mask = torch.tensor([], dtype=torch.bool, device=tile_indices.device)
        
        # Calculate actual percentage for display
        back_percentage = int(back_primitives_percentile * 100)
        
        # Debug: Print criteria counts
        print(f"[Non-Problematic Debug] Tile criteria counts:")
        print(f"  - Small scale (<{scale_threshold}): {small_scale_mask.sum().item()}/{len(tile_indices)}")
        print(f"  - Low opacity (<{opacity_threshold:.2f}): {low_opacity_mask.sum().item()}/{len(tile_indices)}")
        print(f"  - Back primitives (bottom {back_percentage}%): {back_mask.sum().item()}/{len(tile_indices)}")
        
        # Combine criteria: primitives meeting minimum criteria count (opposite approach)
        criteria_count = (small_scale_mask.float() + low_opacity_mask.float() + back_mask.float())
        non_problematic_mask = criteria_count >= float(min_criteria_count)
        
        print(f"  - Non-problematic by criteria (>={min_criteria_count}/3): {non_problematic_mask.sum().item()}/{len(tile_indices)}")
        
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


    
    def debug_adaptive_control_with_visualization(self,
                                                   x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                                   v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                                   target_image: torch.Tensor,
                                                   adaptive_config: dict = None,
                                                   freeze_mask: torch.Tensor = None,
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
            freeze_mask: Optional tensor indicating which primitives should be excluded from adaptive control
            save_dir: Directory to save visualization images
            
        Returns:
            Tuple of (adapted_parameters, selected_indices_per_tile)
        """
        import os
        from datetime import datetime
        from pydiffbmp.util.gradient_visualizer import GradientVisualizer
        
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
        min_criteria_count = adaptive_config.get('min_criteria_count', 2)
        front_primitives_percentile = adaptive_config.get('front_primitives_percentile', 0.7)
        back_primitives_percentile = adaptive_config.get('back_primitives_percentile', 0.3)
        
        print(f"[Debug Adaptive Control] Starting debug with gradient visualization using GradientVisualizer")
        print(f"[Debug Adaptive Control] Tile grid: {tile_rows}x{tile_cols}")
        
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
        
        # Create GradientVisualizer instances once for reuse across tiles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualizer for problematic primitives (warm colors, red dots)
        vis_path_problematic_base = os.path.join(save_dir, f"gradient_problematic_{timestamp}")
        gradient_threshold = self.debug_config.get("gradient_visualization", {}).get("gradient_threshold", 1e-15)
        visualizer_problematic = GradientVisualizer(
            target_image=target_image,
            save_path=vis_path_problematic_base,
            color_spectrum="warm",  # Use warm colors for problematic primitives
            background_color=(1.0, 1.0, 1.0),  # White background
            primitive_radius_multiplier=1.5,
            enable_logging=True,
            gradient_threshold=gradient_threshold,
        )
        
        # Create visualizer for non-problematic primitives (cool colors, blue dots)
        vis_path_non_problematic_base = os.path.join(save_dir, f"gradient_non_problematic_{timestamp}")
        visualizer_non_problematic = GradientVisualizer(
            target_image=target_image,
            save_path=vis_path_non_problematic_base,
            color_spectrum="cool",  # Use cool colors for non-problematic primitives
            background_color=(1.0, 1.0, 1.0),  # White background
            primitive_radius_multiplier=1.5,
            enable_logging=True,
            gradient_threshold=gradient_threshold,
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
                    scale_threshold, opacity_threshold, max_primitives_per_tile,
                    freeze_mask, min_criteria_count, front_primitives_percentile
                )
                
                # Find non-problematic primitives for comparison
                non_problematic_indices = self._find_non_problematic_primitives(
                    tile_indices, x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted,
                    scale_threshold, opacity_threshold, max_primitives_per_tile, min_criteria_count, back_primitives_percentile
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
                enable_logging=True,
                gradient_threshold=gradient_threshold,
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

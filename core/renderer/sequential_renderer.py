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
    
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None):
        super().__init__(canvas_size, S, alpha_upper_bound, device, use_fp16, gamma, output_path)
    
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
        
        # Base reconstruction loss with warmup
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
        
        # Enable gradients for all parameters
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
        
        pbar = tqdm(range(num_iter), desc="Optimizing with warmup scheduling")
        
        # Get adaptive control configuration
        adaptive_config = opt_conf.get('adaptive_control', {})
        apply_every_n = adaptive_config.get('apply_every_n_iterations', 10)
        
        for i in pbar:
            optimizer.zero_grad()
            
            # Apply adaptive control at specified intervals
            if (adaptive_config.get('enabled', False) and 
                i > 0 and i % apply_every_n == 0):
                x, y, r, v, theta, c = self.apply_adaptive_control(
                    x, y, r, v, theta, c, adaptive_config
                )
            
            # Generate masks and render
            cached_masks = self._batched_soft_rasterize(x, y, r, theta, sigma=0)
            rendered = self.render(cached_masks, v, c)
            
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
            
            # Update progress bar with adaptive control info
            postfix = {'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'}
            if adaptive_config.get('enabled', False) and i % apply_every_n == 0:
                postfix['adaptive'] = 'applied'
            pbar.set_postfix(postfix)
        
        return x, y, r, v, theta, c
    
    def apply_adaptive_control(self, 
                             x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, 
                             v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                             adaptive_config: dict = None) -> tuple:
        """
        Apply adaptive control to reduce artifacts by lowering opacity of problematic primitives.
        
        Args:
            x, y, r, v, theta, c: Current primitive parameters
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
        
        # Clone parameters to avoid modifying originals
        v_adapted = v.clone()
        
        # Get canvas dimensions
        canvas_height, canvas_width = self.canvas_size
        
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
                in_tile_mask = ((x >= x_min) & (x < x_max) & 
                               (y >= y_min) & (y < y_max))
                
                if not in_tile_mask.any():
                    continue
                
                # Get indices of primitives in this tile
                tile_indices = torch.where(in_tile_mask)[0]
                
                # Apply criteria to find problematic primitives
                problematic_indices = self._find_problematic_primitives(
                    tile_indices, x, y, r, v, theta, c,
                    scale_threshold, opacity_threshold
                )
                
                # Limit to max primitives per tile
                if len(problematic_indices) > max_primitives_per_tile:
                    # Sort by combined score (scale * opacity) and take top ones
                    scores = r[problematic_indices] * torch.sigmoid(v[problematic_indices])
                    _, top_indices = torch.topk(scores, max_primitives_per_tile)
                    problematic_indices = problematic_indices[top_indices]
                
                # Apply opacity reduction
                if len(problematic_indices) > 0:
                    v_adapted[problematic_indices] *= opacity_reduction_factor
        
        return x, y, r, v_adapted, theta, c
    
    def _find_problematic_primitives(self, 
                                   tile_indices: torch.Tensor,
                                   x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                   v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                   scale_threshold: float, opacity_threshold: float) -> torch.Tensor:
        """
        Find primitives that meet the problematic criteria within a tile.
        
        Args:
            tile_indices: Indices of primitives in the current tile
            x, y, r, v, theta, c: Primitive parameters
            scale_threshold: Minimum scale to be considered large
            opacity_threshold: Minimum opacity to be considered opaque
            
        Returns:
            Tensor of indices of problematic primitives
        """
        if len(tile_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=x.device)
        
        # Extract parameters for primitives in this tile
        tile_x = x[tile_indices]
        tile_y = y[tile_indices]
        tile_r = r[tile_indices]
        tile_v = v[tile_indices]
        
        # Convert visibility to opacity using sigmoid
        tile_opacity = torch.sigmoid(tile_v)
        
        # Criterion 1: Large scale primitives
        large_scale_mask = tile_r >= scale_threshold
        
        # Criterion 2: High opacity primitives
        high_opacity_mask = tile_opacity >= opacity_threshold
        
        # Criterion 3: Front primitives (those with higher z-order)
        # We'll use a combination of scale and opacity as a proxy for "frontness"
        # since larger, more opaque primitives are more likely to occlude others
        frontness_score = tile_r * tile_opacity
        frontness_threshold = torch.quantile(frontness_score, 0.7)  # Top 30%
        front_mask = frontness_score >= frontness_threshold
        
        # Combine all criteria (primitives that meet at least 2 out of 3 criteria)
        criteria_count = large_scale_mask.float() + high_opacity_mask.float() + front_mask.float()
        problematic_mask = criteria_count >= 2.0
        
        # Get the original indices of problematic primitives
        problematic_tile_indices = torch.where(problematic_mask)[0]
        
        if len(problematic_tile_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=x.device)
        
        return tile_indices[problematic_tile_indices]

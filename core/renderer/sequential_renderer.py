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
        
        for i in pbar:
            optimizer.zero_grad()
            
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
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        return x, y, r, v, theta, c

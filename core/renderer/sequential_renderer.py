import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .mse_renderer import MseRenderer


class SequentialFrameRenderer(MseRenderer):
    """
    Specialized renderer for subsequent frames in sequential optimization.
    Inherits from MseRenderer but uses different loss functions and optimization strategies.
    """
    
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None):
        super().__init__(canvas_size, S, alpha_upper_bound, device, use_fp16, gamma, output_path)
        
    def _find_spatial_neighbors(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, k: int = 8) -> torch.Tensor:
        """
        Find k nearest spatial and color neighbors for each splat.
        
        Args:
            x, y: Position tensors
            c: Color tensor (N, 3)
            k: Number of neighbors to find
            
        Returns:
            neighbor_indices: (N, k) tensor of neighbor indices
        """
        N = x.shape[0]
        
        # Compute spatial distances
        pos = torch.stack([x, y], dim=1)  # (N, 2)
        spatial_dist = torch.cdist(pos, pos)  # (N, N)
        
        # Compute color distances
        color_dist = torch.cdist(c, c)  # (N, N)
        
        # Combined distance (spatial + color similarity)
        # Normalize both distances to [0, 1] range
        spatial_dist_norm = spatial_dist / (spatial_dist.max() + 1e-8)
        color_dist_norm = color_dist / (color_dist.max() + 1e-8)
        
        # Combine with equal weight
        combined_dist = 0.7 * spatial_dist_norm + 0.3 * color_dist_norm
        
        # Find k nearest neighbors (excluding self)
        combined_dist.fill_diagonal_(float('inf'))
        _, neighbor_indices = torch.topk(combined_dist, k, dim=1, largest=False)
        
        return neighbor_indices
    
    def _compute_rigidity_loss(self, 
                              x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor,
                              prev_x: torch.Tensor, prev_y: torch.Tensor, prev_theta: torch.Tensor,
                              neighbor_indices: torch.Tensor,
                              spatial_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute rigidity loss to maintain spatial relationships between neighboring splats.
        
        Args:
            x, y, theta: Current frame parameters
            prev_x, prev_y, prev_theta: Previous frame parameters
            neighbor_indices: (N, k) indices of neighbors for each splat
            spatial_weights: (N, k) weights based on spatial proximity
            
        Returns:
            Rigidity loss tensor
        """
        N, k = neighbor_indices.shape
        
        # Compute displacement for each splat
        dx = x - prev_x  # (N,)
        dy = y - prev_y  # (N,)
        
        # Compute theta change (handling circularity)
        dtheta = theta - prev_theta
        dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))  # Wrap to [-π, π]
        
        # Vectorized computation to avoid graph retention issues
        # Expand displacements to match neighbor structure
        dx_expanded = dx.unsqueeze(1).expand(-1, k)  # (N, k)
        dy_expanded = dy.unsqueeze(1).expand(-1, k)  # (N, k)
        dtheta_expanded = dtheta.unsqueeze(1).expand(-1, k)  # (N, k)
        
        # Gather neighbor displacements
        neighbor_dx = dx[neighbor_indices]  # (N, k)
        neighbor_dy = dy[neighbor_indices]  # (N, k)
        neighbor_dtheta = dtheta[neighbor_indices]  # (N, k)
        
        # Compute displacement differences (vectorized)
        dx_diff = (dx_expanded - neighbor_dx) ** 2
        dy_diff = (dy_expanded - neighbor_dy) ** 2
        dtheta_diff = (dtheta_expanded - neighbor_dtheta) ** 2
        
        # Apply spatial weights and sum
        weighted_loss = spatial_weights * (dx_diff + dy_diff + dtheta_diff)
        rigidity_loss = weighted_loss.sum() / N  # Normalize by number of splats
        
        return rigidity_loss
    
    def compute_loss_with_rigidity(self, 
                                  rendered: torch.Tensor, 
                                  target: torch.Tensor, 
                                  x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                  prev_params: dict = None,
                                  loss_config: dict = None) -> torch.Tensor:
        """
        Compute loss with modular loss components including rigidity loss.
        
        Args:
            rendered: Rendered image tensor
            target: Target image tensor
            x, y, theta, c: Current frame parameters
            prev_params: Previous frame parameters dict
            loss_config: Loss configuration with weights/gains
            
        Returns:
            Combined loss tensor
        """
        if loss_config is None:
            loss_config = {'reconstruction_weight': 1.0, 'rigidity_weight': 0.1}
        
        # Base reconstruction loss
        reconstruction_loss = F.mse_loss(rendered, target)
        total_loss = loss_config.get('reconstruction_weight', 1.0) * reconstruction_loss
        
        # Add rigidity loss if previous parameters are available
        if prev_params is not None and 'neighbor_indices' in prev_params:
            rigidity_weight = loss_config.get('rigidity_weight', 0.1)
            
            if rigidity_weight > 0:
                # Detach previous parameters from computation graph to avoid autograd issues
                prev_x = prev_params['x'].detach()
                prev_y = prev_params['y'].detach()
                prev_theta = prev_params['theta'].detach()
                neighbor_indices = prev_params['neighbor_indices'].detach()
                spatial_weights = prev_params['spatial_weights'].detach()
                
                rigidity_loss = self._compute_rigidity_loss(
                    x, y, theta, prev_x, prev_y, prev_theta,
                    neighbor_indices, spatial_weights
                )
                
                total_loss += rigidity_weight * rigidity_loss
        
        return total_loss
    
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
        Default loss computation - falls back to MSE loss.
        Use compute_loss_with_rigidity for subsequent frames with rigidity loss.
        """
        return super().compute_loss(rendered, target, x, y, r, v, theta, c)
    
    def _extract_learning_rates(self, opt_conf: dict) -> dict:
        """
        Extract learning rate configuration, handling both old and new formats.
        
        Args:
            opt_conf: Optimization configuration
            
        Returns:
            Dictionary with learning rate parameters
        """
        # Check if using new parameter-specific learning rate format
        if "learning_rate" in opt_conf and isinstance(opt_conf["learning_rate"], dict):
            lr_conf = opt_conf["learning_rate"]
            return {
                "default_lr": lr_conf.get("default", 0.1),
                "gain_x": lr_conf.get("gain_x", 10.0),
                "gain_y": lr_conf.get("gain_y", 10.0),
                "gain_r": lr_conf.get("gain_r", 10.0),
                "gain_v": lr_conf.get("gain_v", 1.5),
                "gain_theta": lr_conf.get("gain_theta", 1.0),
                "gain_c": lr_conf.get("gain_c", 1.0),
                "decay_rate": opt_conf.get("decay_rate", 0.99)
            }
        else:
            # Fallback to old simple format
            return {
                "default_lr": opt_conf.get("lr", 0.005),
                "gain_x": 1.0,
                "gain_y": 1.0,
                "gain_r": 1.0,
                "gain_v": 1.0,
                "gain_theta": 1.0,
                "gain_c": 1.0,
                "decay_rate": opt_conf.get("lr_decay", 0.95)
            }
    
    def optimize_parameters_position_only(self,
                                        x: torch.Tensor,
                                        y: torch.Tensor,
                                        r: torch.Tensor,
                                        v: torch.Tensor,
                                        theta: torch.Tensor,
                                        c: torch.Tensor,
                                        target: torch.Tensor,
                                        prev_params: dict = None,
                                        opt_conf: dict = None) -> tuple:
        """
        Optimize only position and rotation parameters (x, y, theta) for subsequent frames.
        Keep r, v, c fixed from previous frame.
        
        Args:
            x, y, r, v, theta, c: Current parameter tensors
            target: Target image tensor
            prev_params: Previous frame parameters for temporal consistency
            opt_conf: Optimization configuration
            
        Returns:
            Optimized parameter tensors (x, y, r, v, theta, c)
        """
        if opt_conf is None:
            opt_conf = {"num_iterations": 50, "learning_rate": {"default": 0.005}, "decay_rate": 0.95}
        
        # Only optimize x, y, theta - freeze r, v, c
        x.requires_grad_(True)
        y.requires_grad_(True)
        theta.requires_grad_(True)
        r.requires_grad_(False)
        v.requires_grad_(False)
        c.requires_grad_(False)
        
        # Extract learning rate configuration
        lr_config = self._extract_learning_rates(opt_conf)
        
        # Setup optimizer with parameter-specific learning rates for position-only
        optimizer = torch.optim.Adam([
            {'params': [x], 'lr': lr_config['default_lr'] * lr_config['gain_x']},
            {'params': [y], 'lr': lr_config['default_lr'] * lr_config['gain_y']},
            {'params': [theta], 'lr': lr_config['default_lr'] * lr_config['gain_theta']}
        ])
        
        # Use exponential decay scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_config['decay_rate'])
        
        num_iter = opt_conf.get("num_iterations", opt_conf.get("num_iter", 50))
        temporal_weight = opt_conf.get("temporal_weight", 0.1)
        
        pbar = tqdm(range(num_iter), desc="Optimizing position-only")
        
        for i in pbar:
            optimizer.zero_grad()
            
            # Generate masks and render
            cached_masks = self._batched_soft_rasterize(x, y, r, theta, sigma=0)
            rendered = self.render(cached_masks, v, c)
            
            # Compute loss with rigidity consistency if previous parameters provided
            if prev_params is not None:
                loss_config = {
                    'reconstruction_weight': 1.0,
                    'rigidity_weight': opt_conf.get('rigidity_weight', 0.1)
                }
                loss = self.compute_loss_with_rigidity(rendered, target, x, y, theta, c, prev_params, loss_config)
            else:
                loss = self.compute_loss(rendered, target, x, y, r, v, theta, c)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        return x, y, r, v, theta, c
    
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
        Optimize all parameters for subsequent frames with temporal consistency.
        
        Args:
            x, y, r, v, theta, c: Current parameter tensors
            target: Target image tensor
            prev_params: Previous frame parameters for temporal consistency
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
        temporal_weight = opt_conf.get("temporal_weight", 0.1)
        
        pbar = tqdm(range(num_iter), desc="Optimizing with temporal consistency")
        
        for i in pbar:
            optimizer.zero_grad()
            
            # Generate masks and render
            cached_masks = self._batched_soft_rasterize(x, y, r, theta, sigma=0)
            rendered = self.render(cached_masks, v, c)
            
            # Compute loss with rigidity consistency
            loss_config = {
                'reconstruction_weight': 1.0,
                'rigidity_weight': opt_conf.get('rigidity_weight', 0.1)
            }
            loss = self.compute_loss_with_rigidity(rendered, target, x, y, theta, c, prev_params, loss_config)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        return x, y, r, v, theta, c

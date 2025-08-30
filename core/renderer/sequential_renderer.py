import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .simple_tile_renderer import SimpleTileRenderer


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
    

    
    def compute_grayscale_loss(self, 
                          rendered_gray: torch.Tensor, 
                          target_gray: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between grayscale images.
        This focuses on structural similarity rather than color differences.
        
        Args:
            rendered_gray: Rendered grayscale image tensor (H, W, 1)
            target_gray: Target grayscale image tensor (H, W, 1)
            
        Returns:
            MSE loss value computed on grayscale images
        """
        # Ensure tensors are in consistent precision
        if self.use_fp16:
            # If target is in FP32, convert rendered to FP32
            if target_gray.dtype == torch.float32:
                rendered_gray = rendered_gray.float()
            # If rendered is in FP16, convert target to FP16
            elif rendered_gray.dtype == torch.float16 and target_gray.dtype != torch.float16:
                target_gray = target_gray.half()
        else:
            # In FP32 mode, ensure everything is float32
            rendered_gray = rendered_gray.float()
            target_gray = target_gray.float()
        
        # Compute grayscale MSE loss
        grayscale_mse_loss = F.mse_loss(rendered_gray, target_gray)
        
        return grayscale_mse_loss
    
    def compute_combined_loss(self, 
                     rendered: torch.Tensor, 
                     target: torch.Tensor, 
                     x: torch.Tensor,
                     y: torch.Tensor,
                     r: torch.Tensor,
                     v: torch.Tensor,
                     theta: torch.Tensor,
                     c: torch.Tensor,
                     grayscale_weight: float = 0.7,
                     color_weight: float = 0.3,
                     use_gradient_loss: bool = False,
                     gradient_weight: float = 0.1,
                     use_cosine_similarity: bool = False,
                     use_canny_loss: bool = False,
                     canny_weight: float = 0.1) -> torch.Tensor:
        """
        Compute combined loss using both grayscale and color MSE losses.
        This balances structural similarity (grayscale) with color matching.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            x, y, r, v, theta, c: Current parameter values
            grayscale_weight: Weight for grayscale loss component (default: 0.7)
            color_weight: Weight for color loss component (default: 0.3)
            use_gradient_loss: If True, adds gradient-based loss for edge similarity
            gradient_weight: Weight for gradient loss component (default: 0.1)
            use_cosine_similarity: If True, uses cosine similarity for gradient loss
            use_canny_loss: If True, adds Canny edge-based loss for edge similarity
            canny_weight: Weight for Canny edge loss component (default: 0.1)
            
        Returns:
            Combined weighted loss value
        """
        # Convert RGB to grayscale using standard luminance weights
        # Y = 0.299*R + 0.587*G + 0.114*B
        rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114], 
                                         device=rendered.device, 
                                         dtype=rendered.dtype)
        
        # Convert rendered image to grayscale
        rendered_gray = torch.sum(rendered * rgb_to_gray_weights, dim=-1, keepdim=True)
        
        # Convert target image to grayscale
        target_gray = torch.sum(target * rgb_to_gray_weights, dim=-1, keepdim=True)
        
        # Compute grayscale-based structural loss
        grayscale_loss = self.compute_grayscale_loss(rendered_gray, target_gray)
        
        # Compute color-based loss
        color_loss = self.compute_loss(rendered, target, x, y, r, v, theta, c)
        
        # Start with weighted grayscale and color losses
        combined_loss = grayscale_weight * grayscale_loss + color_weight * color_loss
        
        # Add gradient-based loss with its own weight if requested
        if use_gradient_loss:
            gradient_loss = self._compute_gradient_loss(rendered_gray, target_gray, use_cosine_similarity)
            combined_loss = combined_loss + gradient_weight * gradient_loss
        
        # Add Canny edge-based loss with its own weight if requested
        if use_canny_loss:
            canny_loss = self._compute_canny_loss(rendered_gray, target_gray)
            combined_loss = combined_loss + canny_weight * canny_loss
        
        return combined_loss

    def _compute_gradient_loss(self, rendered_gray: torch.Tensor, target_gray: torch.Tensor, use_cosine_similarity: bool = False) -> torch.Tensor:
        """
        Compute gradient-based loss between grayscale images to focus on edge similarity.
        Uses Sobel operators to compute gradients in x and y directions.
        
        Args:
            rendered_gray: Rendered grayscale image tensor (H, W, 1)
            target_gray: Target grayscale image tensor (H, W, 1)
            use_cosine_similarity: If True, uses cosine similarity loss on gradient vectors.
                                 If False, uses MSE loss on gradient magnitudes (absolute values).
            
        Returns:
            Gradient-based loss value
        """
        # Remove the channel dimension for gradient computation
        rendered_2d = rendered_gray.squeeze(-1)  # (H, W)
        target_2d = target_gray.squeeze(-1)      # (H, W)
        
        # Define Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=rendered_2d.dtype, device=rendered_2d.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=rendered_2d.dtype, device=rendered_2d.device).unsqueeze(0).unsqueeze(0)
        
        # Add batch and channel dimensions for conv2d
        rendered_batch = rendered_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        target_batch = target_2d.unsqueeze(0).unsqueeze(0)      # (1, 1, H, W)
        
        # Compute gradients using Sobel operators
        rendered_grad_x = F.conv2d(rendered_batch, sobel_x, padding=1)
        rendered_grad_y = F.conv2d(rendered_batch, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_batch, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_batch, sobel_y, padding=1)
        
        if use_cosine_similarity:
            # Use cosine similarity loss on gradient vectors (preserves sign information)
            # Flatten gradients to compute per-pixel cosine similarity
            rendered_grad_flat = torch.stack([rendered_grad_x.flatten(), rendered_grad_y.flatten()], dim=0)  # (2, H*W)
            target_grad_flat = torch.stack([target_grad_x.flatten(), target_grad_y.flatten()], dim=0)  # (2, H*W)
            
            # Compute cosine similarity for each pixel's gradient vector
            # cosine_sim = (a · b) / (||a|| * ||b||)
            dot_product = torch.sum(rendered_grad_flat * target_grad_flat, dim=0)  # (H*W,)
            rendered_norm = torch.norm(rendered_grad_flat, dim=0) + 1e-8  # (H*W,)
            target_norm = torch.norm(target_grad_flat, dim=0) + 1e-8  # (H*W,)
            
            cosine_sim = dot_product / (rendered_norm * target_norm)  # (H*W,)
            
            # Convert cosine similarity to loss (1 - cosine_sim), then take mean
            # Cosine similarity ranges from -1 to 1, so (1 - cosine_sim) ranges from 0 to 2
            gradient_loss = torch.mean(1.0 - cosine_sim)
        else:
            # Use MSE loss on gradient magnitudes (absolute values)
            rendered_grad_mag = torch.sqrt(rendered_grad_x**2 + rendered_grad_y**2 + 1e-8)
            target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
            gradient_loss = F.mse_loss(rendered_grad_mag, target_grad_mag)
    
        return gradient_loss

    def _compute_canny_loss(self, rendered_gray: torch.Tensor, target_gray: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified Canny-inspired edge loss between grayscale images.
        Uses differentiable operations to maintain gradient flow during optimization.
        
        Args:
            rendered_gray: Rendered grayscale image tensor (H, W, 1)
            target_gray: Target grayscale image tensor (H, W, 1)
            
        Returns:
            Canny-inspired edge loss value
        """
        # Remove the channel dimension for edge detection
        rendered_2d = rendered_gray.squeeze(-1)  # (H, W)
        target_2d = target_gray.squeeze(-1)      # (H, W)
        
        # Add batch and channel dimensions for processing
        rendered_batch = rendered_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        target_batch = target_2d.unsqueeze(0).unsqueeze(0)      # (1, 1, H, W)
        
        # Step 1: Apply Gaussian smoothing (3x3 kernel for efficiency)
        gaussian_kernel = self._get_gaussian_kernel(3, 0.8, rendered_2d.device, rendered_2d.dtype)
        rendered_smooth = F.conv2d(rendered_batch, gaussian_kernel, padding=1)
        target_smooth = F.conv2d(target_batch, gaussian_kernel, padding=1)
        
        # Step 2: Compute gradients using Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=rendered_2d.dtype, device=rendered_2d.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=rendered_2d.dtype, device=rendered_2d.device).unsqueeze(0).unsqueeze(0)
        
        # Compute gradients
        rendered_grad_x = F.conv2d(rendered_smooth, sobel_x, padding=1)
        rendered_grad_y = F.conv2d(rendered_smooth, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_smooth, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_smooth, sobel_y, padding=1)
        
        # Step 3: Compute gradient magnitude
        rendered_grad_mag = torch.sqrt(rendered_grad_x**2 + rendered_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # Step 4: Differentiable edge detection using soft thresholding
        # Normalize gradient magnitudes to [0, 1] range
        rendered_norm = rendered_grad_mag / (rendered_grad_mag.max() + 1e-8)
        target_norm = target_grad_mag / (target_grad_mag.max() + 1e-8)
        
        # Apply soft thresholding using sigmoid function for differentiability
        # This replaces the hard thresholding in traditional Canny
        # Using fixed thresholds: low=0.1, high=0.2 (commonly used values)
        high_threshold = 0.2
        steepness = 10.0  # Controls the steepness of the sigmoid
        rendered_edges = torch.sigmoid(steepness * (rendered_norm - high_threshold))
        target_edges = torch.sigmoid(steepness * (target_norm - high_threshold))
        
        # Compute MSE loss between edge maps
        canny_loss = F.mse_loss(rendered_edges, target_edges)
        
        return canny_loss

    
    def _get_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate a 2D Gaussian kernel for smoothing.
        """
        # Create coordinate grids
        coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        
        # Compute Gaussian values
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()  # Normalize
        
        return gaussian.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

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
                
                # Apply criteria to find problematic primitives
                problematic_indices = self._find_problematic_primitives(
                    tile_indices, x_adapted, y_adapted, r_adapted, v_adapted, theta_adapted, c_adapted,
                    scale_threshold, opacity_threshold
                )
                
                # Limit to max primitives per tile
                if len(problematic_indices) > max_primitives_per_tile:
                    # Sort by combined score (scale * opacity) and take top ones
                    scores = r_adapted[problematic_indices] * torch.sigmoid(v_adapted[problematic_indices])
                    _, top_indices = torch.topk(scores, max_primitives_per_tile)
                    problematic_indices = problematic_indices[top_indices]
                
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
        
        # Convert visibility to opacity using sigmoid (matching vector_renderer.py)
        tile_opacity = self.alpha_upper_bound * torch.sigmoid(tile_v)
        
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
        
        # Combine all criteria (primitives that meet at least 2 out of 3 criteria)
        criteria_count = large_scale_mask.float() + high_opacity_mask.float() + front_mask.float()
        problematic_mask = criteria_count >= 3.0
        
        # Get the original indices of problematic primitives
        problematic_tile_indices = torch.where(problematic_mask)[0]
        
        if len(problematic_tile_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=x.device)
        
        return tile_indices[problematic_tile_indices]

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

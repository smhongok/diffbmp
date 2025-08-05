import torch
import torch.nn.functional as F
from core.renderer.vector_renderer import VectorRenderer
from typing import Tuple, Dict, Any

class MseRenderer(VectorRenderer):
    """
    Renderer using MSE loss for optimization.
    This is the same as the base VectorRenderer implementation.
    """
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None):
        super().__init__(canvas_size, S, alpha_upper_bound, device, use_fp16, gamma, output_path)
        
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
        Compute MSE loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            MSE loss value
        """
        # Ensure tensors are in consistent precision
        if self.use_fp16:
            # If target is in FP32, convert rendered to FP32
            if target.dtype == torch.float32:
                rendered = rendered.float()
            # If rendered is in FP16, convert target to FP16
            elif rendered.dtype == torch.float16 and target.dtype != torch.float16:
                target = target.half()
        else:
            # In FP32 mode, ensure everything is float32
            rendered = rendered.float()
            target = target.float()
        
        return F.mse_loss(rendered, target)
    
    def optimize_xy_dynamics(self,
                           x1: torch.Tensor,
                           y1: torch.Tensor,
                           r: torch.Tensor,
                           v: torch.Tensor,
                           theta: torch.Tensor,
                           c: torch.Tensor,
                           target_image2: torch.Tensor,
                           xy_opt_conf: Dict[str, Any],
                           capture_optimization_states: bool = False):
        """
        Optimize all parameters (x, y, r, v, theta, c) to transition from SVGSplat1 to match target_image2.
        All parameters are optimized starting from the first image optimization results.
        
        Args:
            x1, y1: Initial x,y parameters from first image optimization
            r, v, theta, c: Initial parameters from first image optimization
            target_image2: Second target image to match
            xy_opt_conf: XY optimization configuration
            capture_optimization_states: Whether to capture states for optimization process MP4
            
        Returns:
            If capture_optimization_states is False:
                Tuple of optimized parameters (x2, y2, r2, v2, theta2, c2)
            If capture_optimization_states is True:
                Tuple of (optimized parameters, optimization_states_list)
        """
        print("Starting full parameter dynamics optimization...")
        
        # Clone all parameters for optimization
        x2 = x1.clone().detach().requires_grad_(True)
        y2 = y1.clone().detach().requires_grad_(True)
        r2 = r.clone().detach().requires_grad_(True)
        v2 = v.clone().detach().requires_grad_(True)
        theta2 = theta.clone().detach().requires_grad_(True)
        c2 = c.clone().detach().requires_grad_(True)
            
        # Setup optimizer for all parameters
        num_iterations = xy_opt_conf.get('num_iterations', 50)
        lr = xy_opt_conf['learning_rate']['default']
        lr_conf = xy_opt_conf['learning_rate']
        do_decay = xy_opt_conf.get('do_decay', True)
        decay_rate = xy_opt_conf.get('decay_rate', 0.98)
        
        param_groups = [
            {'params': x2, 'lr': lr * lr_conf.get("gain_x", 1.0)},
            {'params': y2, 'lr': lr * lr_conf.get("gain_y", 1.0)},
            {'params': r2, 'lr': lr * lr_conf.get("gain_r", 1.0)},
            {'params': v2, 'lr': lr * lr_conf.get("gain_v", 1.0)},
            {'params': theta2, 'lr': lr * lr_conf.get("gain_theta", 1.0)},
            {'params': c2, 'lr': lr * lr_conf.get("gain_c", 1.0)}
        ]
        optimizer = torch.optim.Adam(param_groups)
        
        # Learning rate scheduler
        if do_decay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        print(f"Starting full parameter optimization, {num_iterations} iterations...")
    
        # Initialize optimization states list if capturing is enabled
        optimization_states = [] if capture_optimization_states else None
        save_every_n_epochs = xy_opt_conf.get('optimization_process_mp4', {}).get('save_every_n_epochs', 1) if capture_optimization_states else 1
    
        from tqdm import tqdm
        for epoch in tqdm(range(num_iterations), desc="Full Parameter Dynamics Optimization"):
            optimizer.zero_grad()
            
            # Generate masks with new parameters
            cached_masks = self._batched_soft_rasterize(x2, y2, r2, theta2, sigma=0.0)
            
            # Render image with updated parameters
            rendered = self.render(cached_masks, v2, c2)
            
            # Compute loss against second target image
            use_combined_loss = xy_opt_conf.get('use_combined_loss', False)
            if use_combined_loss:
                grayscale_weight = xy_opt_conf.get('grayscale_weight', 0.7)
                color_weight = xy_opt_conf.get('color_weight', 0.3)
                use_gradient_loss = xy_opt_conf.get('use_gradient_loss', False)
                gradient_weight = xy_opt_conf.get('gradient_weight', 0.1)
                use_cosine_similarity = xy_opt_conf.get('use_cosine_similarity', False)
                use_canny_loss = xy_opt_conf.get('use_canny_loss', False)
                canny_weight = xy_opt_conf.get('canny_weight', 0.1)
                canny_low_threshold = xy_opt_conf.get('canny_low_threshold', 0.1)
                canny_high_threshold = xy_opt_conf.get('canny_high_threshold', 0.2)
                loss = self.compute_combined_loss(rendered, target_image2, x2, y2, r2, v2, theta2, c2, 
                                                grayscale_weight, color_weight, use_gradient_loss, gradient_weight, use_cosine_similarity,
                                                use_canny_loss, canny_weight, canny_low_threshold, canny_high_threshold)
                if epoch % 20 == 0:
                    gradient_info = " (cosine similarity)" if use_cosine_similarity else " (absolute value)"
                    loss_type = "gradient-based" if use_gradient_loss else "pixel-based"
                    canny_info = f", canny: {canny_weight}" if use_canny_loss else ""
                    print(f"Using combined loss {loss_type}{gradient_info if use_gradient_loss else ''} (grayscale: {grayscale_weight}, color: {color_weight}, gradient: {gradient_weight}{canny_info})")
            else:
                loss = self.compute_loss(rendered, target_image2, x2, y2, r, v, theta, c)
                if epoch % 20 == 0:
                    print("Using standard color loss")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            if do_decay:
                scheduler.step()
            
            # Capture optimization state if enabled
            if capture_optimization_states and epoch % save_every_n_epochs == 0:
                # Clone current parameters for state capture
                state = (
                    epoch,
                    x2.clone().detach(),
                    y2.clone().detach(), 
                    r2.clone().detach(),
                    v2.clone().detach(),
                    theta2.clone().detach(),
                    c2.clone().detach()
                )
                optimization_states.append(state)
        
            # Log progress
            if epoch % 20 == 0 or epoch == num_iterations - 1:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
        print("Full parameter dynamics optimization completed.")
        if capture_optimization_states:
            return (x2, y2, r2, v2, theta2, c2), optimization_states
        else:
            return x2, y2, r2, v2, theta2, c2
    
    def render_xy_transition_mp4(self,
                           x1: torch.Tensor, y1: torch.Tensor,
                           x2: torch.Tensor, y2: torch.Tensor,
                           r1: torch.Tensor,
                           v1: torch.Tensor,
                           theta1: torch.Tensor,
                           c1: torch.Tensor,
                           r2: torch.Tensor,
                           v2: torch.Tensor,
                           theta2: torch.Tensor,
                           c2: torch.Tensor,
                           video_path: str,
                           transition_frames: int = 60,
                           fps: int = 30):
        """
        Create MP4 video showing smooth transition between two SVGSplats.
        All parameters (x, y, r, v, theta, c) are interpolated between the two SVGSplats.
        
        Args:
            x1, y1: Position parameters for first SVGSplat
            x2, y2: Position parameters for second SVGSplat
            r1, v1, theta1, c1: Parameters for first SVGSplat
            r2, v2, theta2, c2: Parameters for second SVGSplat
            video_path: Path to save the MP4 video
            transition_frames: Number of frames for the transition
            fps: Frames per second for the video
        """
        import cv2
        import numpy as np
        import os
        from tqdm import tqdm
        
        print(f"Creating full parameter transition video with {transition_frames} frames...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (self.W, self.H))
        
        for frame_idx in tqdm(range(transition_frames), desc="Rendering transition frames"):
            # Interpolation factor (0 to 1)
            t = frame_idx / (transition_frames - 1) if transition_frames > 1 else 0
            
            # Interpolate all parameters
            x_interp = (1 - t) * x1 + t * x2
            y_interp = (1 - t) * y1 + t * y2
            r_interp = (1 - t) * r1 + t * r2
            v_interp = (1 - t) * v1 + t * v2
            theta_interp = (1 - t) * theta1 + t * theta2
            c_interp = (1 - t) * c1 + t * c2
            
            # Generate masks and render
            cached_masks = self._batched_soft_rasterize(x_interp, y_interp, r_interp, theta_interp, sigma=0.0)
            rendered = self.render(cached_masks, v_interp, c_interp)
            
            # Convert to numpy and prepare for video
            frame = rendered.detach().cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(frame_bgr)
        
        out.release()
        print(f"XY transition video saved to: {video_path}")
    
    def render_optimization_process_mp4(self,
                                      optimization_states: list,
                                      video_path: str,
                                      fps: int = 10):
        """
        Create MP4 video showing the optimization process evolution.
        Each frame shows the primitive state at a different optimization epoch.
        
        Args:
            optimization_states: List of optimization states (epoch, x, y, r, v, theta, c)
            video_path: Path to save the MP4 video
            fps: Frames per second for the video
        """
        import cv2
        import numpy as np
        
        if not optimization_states:
            print("No optimization states to render")
            return
            
        print(f"Creating optimization process video with {len(optimization_states)} frames...")
        
        # Get canvas dimensions from the first frame
        first_state = optimization_states[0]
        epoch, x, y, r, v, theta, c = first_state
        
        # Render first frame to get dimensions
        with torch.no_grad():
            cached_masks = self._batched_soft_rasterize(x, y, r, theta, sigma=0.0)
            rendered = self.render(cached_masks, v, c)
            frame = rendered.detach().cpu().numpy()
            
        height, width = frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Render each optimization state as a frame
        for i, state in enumerate(optimization_states):
            epoch, x, y, r, v, theta, c = state
            
            with torch.no_grad():
                # Generate masks and render
                cached_masks = self._batched_soft_rasterize(x, y, r, theta, sigma=0.0)
                rendered = self.render(cached_masks, v, c)
                
                # Convert to numpy and prepare for video
                frame = rendered.detach().cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                
                # Add epoch number overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (255, 255, 255)  # White text
                thickness = 2
                text = f"Epoch: {epoch}"
                
                # Get text size and position
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = 10
                text_y = text_height + 10
                
                # Add black background for text
                cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), 
                            (text_x + text_width + 5, text_y + baseline + 5), (0, 0, 0), -1)
                
                # Add text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(frame_bgr)
                
            if (i + 1) % 10 == 0 or i == len(optimization_states) - 1:
                print(f"Processed {i + 1}/{len(optimization_states)} frames")
        
        out.release()
        print(f"Optimization process video saved to: {video_path}")
    
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
                     canny_weight: float = 0.1,
                     canny_low_threshold: float = 0.1,
                     canny_high_threshold: float = 0.2) -> torch.Tensor:
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
            canny_low_threshold: Low threshold for Canny edge detection (default: 0.1)
            canny_high_threshold: High threshold for Canny edge detection (default: 0.2)
            
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
            canny_loss = self._compute_canny_loss(rendered_gray, target_gray, canny_low_threshold, canny_high_threshold)
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

    def _compute_canny_loss(self, rendered_gray: torch.Tensor, target_gray: torch.Tensor, 
                           low_threshold: float = 0.1, high_threshold: float = 0.2) -> torch.Tensor:
        """
        Compute simplified Canny-inspired edge loss between grayscale images.
        Uses differentiable operations to maintain gradient flow during optimization.
        
        Args:
            rendered_gray: Rendered grayscale image tensor (H, W, 1)
            target_gray: Target grayscale image tensor (H, W, 1)
            low_threshold: Low threshold for edge detection (default: 0.1)
            high_threshold: High threshold for edge detection (default: 0.2)
            
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

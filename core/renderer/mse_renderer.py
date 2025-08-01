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
                           xy_opt_conf: Dict[str, Any]):
        """
        Optimize only x,y parameters to transition from SVGSplat1 to match target_image2.
        All other parameters (r, v, theta, c) remain fixed from the first optimization.
        
        Args:
            x1, y1: Initial x,y parameters from first image optimization
            r, v, theta, c: Fixed parameters from first image optimization
            target_image2: Second target image to match
            xy_opt_conf: XY optimization configuration
            
        Returns:
            Tuple of optimized parameters (x2, y2, r, v, theta, c)
        """
        print("Starting XY dynamics optimization...")
        
        # Clone x,y parameters for optimization (others remain fixed)
        x2 = x1.clone().detach().requires_grad_(True)
        y2 = y1.clone().detach().requires_grad_(True)
        
        # Ensure other parameters don't require gradients
        r.requires_grad_(False)
        v.requires_grad_(False)
        theta.requires_grad_(False)
        c.requires_grad_(False)
        
        # Setup optimizer for only x,y parameters
        num_iterations = xy_opt_conf.get('num_iterations', 50)
        lr = xy_opt_conf['learning_rate']['default']
        lr_conf = xy_opt_conf['learning_rate']
        do_decay = xy_opt_conf.get('do_decay', True)
        decay_rate = xy_opt_conf.get('decay_rate', 0.98)
        
        param_groups = [
            {'params': x2, 'lr': lr * lr_conf.get("gain_x", 1.0)},
            {'params': y2, 'lr': lr * lr_conf.get("gain_y", 1.0)}
        ]
        optimizer = torch.optim.Adam(param_groups)
        
        # Learning rate scheduler
        if do_decay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        print(f"Starting XY optimization, {num_iterations} iterations...")
        
        from tqdm import tqdm
        for epoch in tqdm(range(num_iterations), desc="XY Dynamics Optimization"):
            optimizer.zero_grad()
            
            # Generate masks with new x,y positions
            cached_masks = self._batched_soft_rasterize(x2, y2, r, theta, sigma=0.0)
            
            # Render image with updated positions
            rendered = self.render(cached_masks, v, c)
            
            # Compute loss against second target image
            # Use combined loss if enabled in configuration
            use_combined_loss = xy_opt_conf.get('use_combined_loss', False)
            if use_combined_loss:
                grayscale_weight = xy_opt_conf.get('grayscale_weight', 0.7)
                color_weight = xy_opt_conf.get('color_weight', 0.3)
                loss = self.compute_combined_loss(rendered, target_image2, x2, y2, r, v, theta, c, 
                                                grayscale_weight, color_weight)
                if epoch % 10 == 0:
                    print(f"Using combined loss (grayscale: {grayscale_weight}, color: {color_weight})")
            else:
                loss = self.compute_loss(rendered, target_image2, x2, y2, r, v, theta, c)
                if epoch % 10 == 0:
                    print("Using standard color loss")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            if do_decay:
                scheduler.step()
            
            # Log progress
            if epoch % 10 == 0 or epoch == num_iterations - 1:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
        print("XY dynamics optimization completed.")
        return x2, y2, r, v, theta, c
    
    def render_xy_transition_mp4(self,
                               x1: torch.Tensor, y1: torch.Tensor,
                               x2: torch.Tensor, y2: torch.Tensor,
                               r: torch.Tensor,
                               v: torch.Tensor,
                               theta: torch.Tensor,
                               c: torch.Tensor,
                               video_path: str,
                               transition_frames: int = 60,
                               fps: int = 30):
        """
        Create MP4 video showing smooth transition between two SVGSplats.
        Only x,y parameters are interpolated; other parameters remain constant.
        
        Args:
            x1, y1: Position parameters for first SVGSplat
            x2, y2: Position parameters for second SVGSplat
            r, v, theta, c: Fixed parameters for both SVGSplats
            video_path: Path to save the MP4 video
            transition_frames: Number of frames for the transition
            fps: Frames per second for the video
        """
        import cv2
        import numpy as np
        import os
        from tqdm import tqdm
        
        print(f"Creating XY transition video with {transition_frames} frames...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (self.W, self.H))
        
        for frame_idx in tqdm(range(transition_frames), desc="Rendering transition frames"):
            # Interpolation factor (0 to 1)
            t = frame_idx / (transition_frames - 1) if transition_frames > 1 else 0
            
            # Interpolate x,y parameters
            x_interp = (1 - t) * x1 + t * x2
            y_interp = (1 - t) * y1 + t * y2
            
            # Generate masks and render
            cached_masks = self._batched_soft_rasterize(x_interp, y_interp, r, theta, sigma=0.0)
            rendered = self.render(cached_masks, v, c)
            
            # Convert to numpy and prepare for video
            frame = rendered.detach().cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(frame_bgr)
        
        out.release()
        print(f"XY transition video saved to: {video_path}")
    
    def compute_grayscale_loss(self, 
                              rendered: torch.Tensor, 
                              target: torch.Tensor, 
                              x: torch.Tensor,
                              y: torch.Tensor,
                              r: torch.Tensor,
                              v: torch.Tensor,
                              theta: torch.Tensor,
                              c: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between grayscale versions of rendered and target images.
        This focuses on structural similarity rather than color differences.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            MSE loss value computed on grayscale images
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
        
        return F.mse_loss(rendered_gray, target_gray)
    
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
                             color_weight: float = 0.3) -> torch.Tensor:
        """
        Compute combined loss using both grayscale and color MSE losses.
        This balances structural similarity (grayscale) with color matching.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            x, y, r, v, theta, c: Current parameter values
            grayscale_weight: Weight for grayscale loss component (default: 0.7)
            color_weight: Weight for color loss component (default: 0.3)
            
        Returns:
            Combined weighted loss value
        """
        # Compute grayscale-based structural loss
        grayscale_loss = self.compute_grayscale_loss(rendered, target, x, y, r, v, theta, c)
        
        # Compute color-based loss
        color_loss = self.compute_loss(rendered, target, x, y, r, v, theta, c)
        
        # Combine losses with specified weights
        combined_loss = grayscale_weight * grayscale_loss + color_weight * color_loss
        
        return combined_loss

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
            loss = self.compute_loss(rendered, target_image2, x2, y2, r2, v2, theta2, c2)
            
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
            if epoch % 10 == 0 or epoch == num_iterations - 1:
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
        Create MP4 video showing the optimization process during xy_dynamics.
        Each frame shows the state of primitives at different optimization epochs.
        
        Args:
            optimization_states: List of tuples (epoch, x, y, r, v, theta, c) captured during optimization
            video_path: Path to save the MP4 video
            fps: Frames per second for the video
        """
        import cv2
        import numpy as np
        import os
        from tqdm import tqdm
        
        if not optimization_states:
            print("No optimization states to render")
            return
        
        print(f"Creating optimization process video with {len(optimization_states)} frames...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (self.W, self.H))
        
        for i, (epoch, x, y, r, v, theta, c) in enumerate(tqdm(optimization_states, desc="Rendering optimization frames")):
            # Generate masks and render for this optimization state
            with torch.no_grad():
                cached_masks = self._batched_soft_rasterize(x, y, r, theta, sigma=0.0)
                rendered = self.render(cached_masks, v, c)
                
                # Convert to numpy and prepare for video
                frame = rendered.detach().cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add epoch text overlay
                cv2.putText(frame_bgr, f"Epoch: {epoch}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Write frame
                out.write(frame_bgr)
        
        out.release()
        print(f"Optimization process video saved to: {video_path}")

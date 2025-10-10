import torch
import torch.nn.functional as F
from core.renderer.vector_renderer import VectorRenderer
from util.loss_functions import LossComposer
from typing import Tuple, Dict, Any

class MseRenderer(VectorRenderer):
    """
    Renderer using flexible loss function system for optimization.
    """
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None, tile_size=32):
        super().__init__(canvas_size, S, alpha_upper_bound, device, use_fp16, gamma, output_path, tile_size)
        # Initialize loss composer (will be set later with config)
        self.loss_composer = None
        
    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor,
                    rendered_alpha: torch.Tensor = None,
                    loss_w_conf = None,
                    epoch = None,
                    return_components: bool = False):
        """
        Compute loss using flexible loss function system.
        
        Args:
            rendered: Rendered RGB image tensor (H, W, 3)
            target: Target image tensor (H, W, 3) or (H, W, 4) if it has an alpha channel
            x, y, r, v, theta, c: Current parameter values
            rendered_alpha: Optional alpha channel tensor (H, W) if available
            loss_w_conf: Deprecated, kept for compatibility
            epoch: Current epoch number
            return_components: If True, return (loss, components_dict)
            
        Returns:
            Loss value, or (loss, components_dict) if return_components=True
        """
        # Ensure loss composer is initialized
        if self.loss_composer is None:
            raise RuntimeError("Loss composer not initialized. Call optimization method first.")
        
        # Ensure consistent precision
        if self.use_fp16:
            if target.dtype == torch.float32:
                rendered = rendered.float()
                if rendered_alpha is not None:
                    rendered_alpha = rendered_alpha.float()
            elif rendered.dtype == torch.float16 and target.dtype != torch.float16:
                target = target.half()
        else:
            rendered = rendered.float()
            target = target.float()
            if rendered_alpha is not None:
                rendered_alpha = rendered_alpha.float()
        
        # Handle target with alpha channel
        if target.shape[2] == 4:
            # Extract RGB and alpha
            target_rgb = target[:, :, :3]     # Shape: (H, W, 3)
            target_alpha = target[:, :, 3]    # Shape: (H, W)
            
            # Handle rendered_alpha shape: could be (H, W) or (H, W, 1)
            if rendered_alpha is not None:
                if rendered_alpha.dim() == 3 and rendered_alpha.shape[2] == 1:
                    rendered_alpha = rendered_alpha.squeeze(-1)  # (H, W, 1) -> (H, W)
            
            # Create foreground mask
            alpha_mask = target_alpha > 0     # Shape: (H, W), boolean mask
            
            # Use loss composer
            result = self.loss_composer.compute_loss(
                rendered=rendered,
                target=target_rgb,
                rendered_alpha=rendered_alpha,
                target_alpha=target_alpha,
                mask=alpha_mask,
                return_components=return_components
            )
        else:
            # 3-channel target (with background)
            result = self.loss_composer.compute_loss(
                rendered=rendered,
                target=target,
                rendered_alpha=None,
                target_alpha=None,
                mask=None,
                return_components=return_components
            )
        
        return result    

    
    
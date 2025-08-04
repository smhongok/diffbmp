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
            target: Target image tensor (H, W, 3) or (H, W, 4) if it has an alpha channel
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            MSE loss value
        """
        # If target has an alpha channel, use it as a mask for loss calculation
        # In this case, we only compute loss for pixels where alpha > 0 (foreground pixels)
        if target.shape[2] == 4:
            # Extract alpha channel and RGB channels
            alpha_mask = target[:, :, 3] > 0  # Shape: (H, W), boolean mask
            target_rgb = target[:, :, :3]     # Shape: (H, W, 3)
            
            # Only compute loss for pixels where alpha > 0
            if alpha_mask.any():
                # Apply mask to both rendered and target
                rendered_masked = rendered[alpha_mask]    # Shape: (N_valid, 3)
                target_masked = target_rgb[alpha_mask]    # Shape: (N_valid, 3)
                
                # Ensure tensors are in consistent precision
                if self.use_fp16:
                    if target_masked.dtype == torch.float32:
                        rendered_masked = rendered_masked.float()
                    elif rendered_masked.dtype == torch.float16 and target_masked.dtype != torch.float16:
                        target_masked = target_masked.half()
                else:
                    rendered_masked = rendered_masked.float()
                    target_masked = target_masked.float()
                
                return F.mse_loss(rendered_masked, target_masked)
            else:
                # If no valid pixels, return zero loss
                return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        else:
            # Original behavior for 3-channel targets
            target_rgb = target
            
            # Ensure tensors are in consistent precision
            if self.use_fp16:
                if target_rgb.dtype == torch.float32:
                    rendered = rendered.float()
                elif rendered.dtype == torch.float16 and target_rgb.dtype != torch.float16:
                    target_rgb = target_rgb.half()
            else:
                rendered = rendered.float()
                target_rgb = target_rgb.float()

            return F.mse_loss(rendered, target_rgb)

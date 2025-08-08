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
                    c: torch.Tensor,
                    rendered_alpha: torch.Tensor = None) -> torch.Tensor:
        """
        Compute MSE loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3) or (H, W, 4) if it has an alpha channel
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            rendered_alpha: Optional alpha channel tensor (H, W) if available
            
        Returns:
            MSE loss value
        """
        # If target has an alpha channel, use it as a mask for loss calculation
        # In this case, we only compute loss for pixels where alpha > 0 (foreground pixels)
        # and include both RGB and alpha channel in the loss calculation
        if target.shape[2] == 4:
            assert rendered_alpha is not None, "Rendered alpha channel must be provided when target has an alpha channel."
            
            # Handle rendered_alpha shape: could be (H, W) or (H, W, 1)
            if rendered_alpha.dim() == 3 and rendered_alpha.shape[2] == 1:
                rendered_alpha = rendered_alpha.squeeze(-1)  # (H, W, 1) -> (H, W)

            # Extract alpha channel once and create mask
            target_alpha = target[:, :, 3]    # Shape: (H, W)
            alpha_mask = target_alpha > 0     # Shape: (H, W), boolean mask
            
            # Only compute loss for pixels where alpha > 0
            if alpha_mask.any():
                # Extract RGB channels
                target_rgb = target[:, :, :3]     # Shape: (H, W, 3)
                
                # Ensure consistent precision before masking to avoid unnecessary conversions
                if self.use_fp16:
                    if target_rgb.dtype == torch.float32:
                        rendered = rendered.float()
                        rendered_alpha = rendered_alpha.float()
                    elif rendered.dtype == torch.float16 and target_rgb.dtype != torch.float16:
                        target_rgb = target_rgb.half()
                        target_alpha = target_alpha.half()
                else:
                    rendered = rendered.float()
                    rendered_alpha = rendered_alpha.float()
                    target_rgb = target_rgb.float()
                    target_alpha = target_alpha.float()
                
                # Apply mask to all tensors
                rendered_masked = rendered[alpha_mask]              # Shape: (N_valid, 3)
                rendered_alpha_masked = rendered_alpha[alpha_mask]  # Shape: (N_valid,)
                target_rgb_masked = target_rgb[alpha_mask]          # Shape: (N_valid, 3)
                target_alpha_masked = target_alpha[alpha_mask]      # Shape: (N_valid,)
                
                # Combine RGB and alpha channels for single MSE computation
                # Concatenate along last dimension: RGB (3) + Alpha (1) = 4 channels
                rendered_combined = torch.cat([rendered_masked, rendered_alpha_masked.unsqueeze(-1)], dim=-1)  # (N_valid, 4)
                target_combined = torch.cat([target_rgb_masked, target_alpha_masked.unsqueeze(-1)], dim=-1)    # (N_valid, 4)
                
                return F.mse_loss(rendered_combined, target_combined)
            else:
                # If no valid pixels, return zero loss
                return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        else:
            # Original behavior for 3-channel targets
            # Ensure tensors are in consistent precision
            if self.use_fp16:
                if target.dtype == torch.float32:
                    rendered = rendered.float()
                elif rendered.dtype == torch.float16 and target.dtype != torch.float16:
                    target = target.half()
            else:
                rendered = rendered.float()
                target = target.float()

            return F.mse_loss(rendered, target)

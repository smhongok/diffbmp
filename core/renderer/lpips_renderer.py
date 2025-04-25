import torch
import torch.nn.functional as F
import piq
from core.renderer.vector_renderer import VectorRenderer
from typing import Tuple, Dict, Any

class LpipsRenderer(VectorRenderer):
    """
    Renderer using LPIPS (Learned Perceptual Image Patch Similarity) loss for optimization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpips = piq.LPIPS().to(self.device)
    
    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    cached_masks: torch.Tensor,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            LPIPS loss value
        """
        # LPIPS expects input in NCHW format and normalized to [-1, 1]
        rendered_lpips = rendered.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        target_lpips = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return self.lpips(rendered_lpips, target_lpips)

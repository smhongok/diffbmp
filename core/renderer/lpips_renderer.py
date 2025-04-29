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
        self.lpips = piq.LPIPS(reduction='mean').to(self.device)
    
    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor) -> torch.Tensor:
        # 1) Prepare input: NCHW, float32, on same device
        #    assume rendered/target already in [0,1] float
        rendered_lpips = rendered.permute(2,0,1).unsqueeze(0).to(self.device)
        target_lpips   = target  .permute(2,0,1).unsqueeze(0).to(self.device)

        # 2) Compute LPIPS
        return self.lpips(rendered_lpips, target_lpips)

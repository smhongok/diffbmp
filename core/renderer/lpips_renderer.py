import torch
import torch.nn.functional as F
import piq
from core.renderer.vector_renderer import VectorRenderer
from typing import Tuple, Dict, Any
import torch.utils.checkpoint as checkpoint

class LpipsRenderer(VectorRenderer):
    """
    Renderer using LPIPS (Learned Perceptual Image Patch Similarity) loss for optimization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create standard LPIPS module
        self.lpips = piq.LPIPS(reduction='mean').to(self.device)
        # By default, checkpointing is disabled
        self.use_lpips_checkpointing = False
    
    def enable_lpips_checkpointing(self):
        """Enable gradient checkpointing for the LPIPS model to save memory."""
        self.use_lpips_checkpointing = True
        print("LPIPS gradient checkpointing enabled")
    
    def disable_lpips_checkpointing(self):
        """Disable gradient checkpointing for the LPIPS model."""
        self.use_lpips_checkpointing = False
        print("LPIPS gradient checkpointing disabled")
    
    def enable_checkpointing(self):
        """Override to also enable LPIPS checkpointing when general checkpointing is enabled."""
        super().enable_checkpointing()
        self.enable_lpips_checkpointing()
    
    def disable_checkpointing(self):
        """Override to also disable LPIPS checkpointing when general checkpointing is disabled."""
        super().disable_checkpointing()
        self.disable_lpips_checkpointing()
    
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

        # 2) Compute LPIPS with optional checkpointing
        if self.use_lpips_checkpointing:
            # Define checkpoint function that can handle tensors
            def lpips_func(r, t):
                return self.lpips(r, t)
            
            # Apply checkpoint
            loss = checkpoint.checkpoint(
                lpips_func, 
                rendered_lpips, 
                target_lpips,
                use_reentrant=False
            )
            return loss
        else:
            # Standard computation without checkpointing
            return self.lpips(rendered_lpips, target_lpips)

import torch
import torch.nn.functional as F
from .svgsplat_initializater import StructureAwareInitializer
from core.renderer.vector_renderer import VectorRenderer
from tqdm import tqdm
from typing import Dict, Any
import piq
import numpy as np

class SingleLevelInitializer(StructureAwareInitializer):
    def __init__(self, init_opt:Dict[str, Any]):
        super().__init__(init_opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prune_tau = init_opt.get("prune_tau", 0.01)
        self.pruning_enabled = init_opt.get("pruning_enabled", True)
        self.lpips = piq.LPIPS(reduction='mean').to(self.device)

    def initialize(self, I_tar, I_bg=None, renderer:VectorRenderer=None, opt_conf:Dict[str, Any]=None):
        """
        Implements the SingleLevelSVGSplat algorithm from the provided pseudocode.
        Args:
            I_tar: Target image (torch.Tensor)
            renderer: VectorRenderer instance
            opt_conf: Dictionary of optimization configuration parameters
        Returns:
            tuple: (x, y, r, v, theta, c)
        """
        # [TODO] I_bg implementation 
        if I_bg is None:
            I_bg = torch.zeros_like(I_tar)
        
        # 1. Structure-aware initialization (use parent class)
        x, y, r, v, theta, c = renderer.initialize_parameters(super(), I_bg)
        
        # ---- c shape 보정 ----
        if c.shape[1] != 3:
            c = c[:, :3].contiguous().detach().clone().requires_grad_(True)
        
        num_iterations = opt_conf.get("num_iterations", 100)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
            {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
            {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},
            {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
            {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
        ])
        
        target_lpips   = I_tar.permute(2,0,1).unsqueeze(0).to(self.device)
        print(f"Starting optimization for {num_iterations} iterations...")
        for epoch in tqdm(range(num_iterations)):
            optimizer.zero_grad()
            
            I_hat = renderer.render_image(x, y, r, v, theta, c)
            I_mix = renderer.over(I_hat, I_bg)
           
            # MSE Loss
            loss = F.mse_loss(I_mix, I_tar)
            
            # LPIPS Loss
            rendered_lpips = I_mix.permute(2,0,1).unsqueeze(0).to(self.device)
            loss += self.lpips(rendered_lpips, target_lpips)

            loss.backward(retain_graph=True)
            
            # Update parameters
            optimizer.step()
            
            # Clamp parameters
            with torch.no_grad():
                x.clamp_(0, renderer.W)
                y.clamp_(0, renderer.H)
                r.clamp_(2, min(renderer.H, renderer.W) // 4)
                theta.clamp_(0, 2 * np.pi)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return x, y, r, v, theta, c
        
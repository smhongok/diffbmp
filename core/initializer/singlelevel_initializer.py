import torch
from .svgsplat_initializater import StructureAwareInitializer
from core.renderer.vector_renderer import VectorRenderer
from tqdm import tqdm
from typing import Dict, Any
class SingleLevelInitializer(StructureAwareInitializer):
    def __init__(self, num_init=100, alpha=0.3, min_distance=20, 
                 peak_threshold=0.5, radii_min=2, radii_max=None, 
                 v_init_bias=-5.0, v_init_slope=0.0, keypoint_extracting=False, debug_mode=False,
                 prune_tau=0.01, pruning_enabled=True):
        super().__init__(num_init, alpha, min_distance, peak_threshold, radii_min, 
                         radii_max, v_init_bias, v_init_slope, keypoint_extracting, debug_mode)
        self.prune_tau = prune_tau
        self.pruning_enabled = pruning_enabled
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self, I_tar, I_bg=None, S=None, N=None, renderer:VectorRenderer=None, opt_conf:Dict[str, Any]=None):
        """
        Implements the SingleLevelSVGSplat algorithm from the provided pseudocode.
        Args:
            I_tar: Target image (torch.Tensor)
            I_bg: Background image (torch.Tensor or None)
            S: Structure/context (optional, for compatibility)
            N: Number of splats (optional, defaults to self.num_init)
            renderer: VectorRenderer instance
            opt_conf: Dictionary of optimization configuration parameters
        Returns:
            tuple: (x, y, r, v, theta, c)
        """
        if N is None:
            N = self.num_init
        if I_bg is None:
            I_bg = torch.zeros_like(I_tar)

        # 1. Structure-aware initialization (use parent class)
        x, y, r, v, theta, c = super().initialize(I_tar)
        x, y, r, v, theta, c = renderer.initialize_parameters(self, I_tar)
        x, y, r, v, theta, c = renderer.optimize_parameters(x, y, r, v, theta, c, I_tar, I_bg, opt_conf)
        params = [x, y, r, v, theta, c]
        # Pruning step (drop splats with v < tau)
        if self.pruning_enabled:
            keep = params[3].detach().abs() > self.prune_tau  # v is the 4th param
            params = [p.detach()[keep] for p in params]

        return tuple(p.detach() for p in params)
 
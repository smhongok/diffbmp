import numpy as np
import torch
from .base_initializer import BaseInitializer
from pydiffbmp.core.renderer.vector_renderer import VectorRenderer
from typing import Dict, Any, List, Union


class DesignatedInitializer(BaseInitializer):
    """
    Initializer that uses user-specified parameter values.
    Allows direct control over initial positions, sizes, rotations, etc.
    """
    
    def __init__(self, init_opt: Dict[str, Any]):
        super().__init__(init_opt)
        
        # Extract designated parameters from config
        # These should be lists or single values
        self.x_values = init_opt.get("x", None)
        self.y_values = init_opt.get("y", None)
        self.r_values = init_opt.get("r", None)
        self.theta_values = init_opt.get("theta", None)
        self.v_values = init_opt.get("v", None)
        self.c_values = init_opt.get("c", None)
        
        # Validate that at least some parameters are provided
        if all(v is None for v in [self.x_values, self.y_values, self.r_values, 
                                     self.theta_values, self.v_values, self.c_values]):
            raise ValueError("DesignatedInitializer requires at least one parameter to be specified")
    
    def _normalize_param(self, param: Union[float, List[float], None], 
                        N: int, default_min: float, default_max: float) -> List[float]:
        """
        Normalize parameter to a list of N values.
        
        Args:
            param: Single value, list of values, or None
            N: Number of primitives
            default_min: Default minimum value if param is None
            default_max: Default maximum value if param is None
        
        Returns:
            List of N values
        """
        if param is None:
            # Use random values in default range
            return np.random.uniform(default_min, default_max, N).tolist()
        elif isinstance(param, (int, float)):
            # Single value - replicate for all primitives
            return [float(param)] * N
        elif isinstance(param, list):
            if len(param) == N:
                return [float(v) for v in param]
            elif len(param) == 1:
                return [float(param[0])] * N
            else:
                raise ValueError(f"Parameter list length {len(param)} does not match N={N}")
        else:
            raise ValueError(f"Invalid parameter type: {type(param)}")
    
    def _normalize_color_param(self, param: Union[List[float], List[List[float]], None], 
                               N: int) -> List[List[float]]:
        """
        Normalize color parameter to a list of N RGB values.
        
        Args:
            param: Single RGB, list of RGB values, or None
            N: Number of primitives
        
        Returns:
            List of N RGB values
        """
        if param is None:
            # Random colors
            return np.random.uniform(0, 1, (N, 3)).tolist()
        elif isinstance(param, list):
            if len(param) == 3 and all(isinstance(v, (int, float)) for v in param):
                # Single RGB value - replicate for all
                return [[float(v) for v in param]] * N
            elif len(param) == N and all(isinstance(v, list) and len(v) == 3 for v in param):
                # List of RGB values
                return [[float(c) for c in rgb] for rgb in param]
            elif len(param) == 1 and isinstance(param[0], list) and len(param[0]) == 3:
                # Single RGB in list - replicate
                return [[float(v) for v in param[0]]] * N
            else:
                raise ValueError(f"Invalid color parameter format")
        else:
            raise ValueError(f"Invalid color parameter type: {type(param)}")
    
    def initialize(self, I_target, target_binary_mask=None, I_bg=None, 
                   renderer: VectorRenderer = None, opt_conf: Dict[str, Any] = None):
        """
        Initialize parameters using designated values from config.
        
        Args:
            I_target: Target image tensor (H, W, 3) or (H, W)
            target_binary_mask: Optional binary mask
            I_bg: Optional background image
            renderer: Renderer instance
            opt_conf: Optimization configuration
        
        Returns:
            Tuple of (x, y, r, v, theta, c) tensors
        """
        device = I_target.device
        
        # Extract image dimensions
        if I_target.ndim == 3:
            H, W, _ = I_target.shape
        else:
            H, W = I_target.shape
        
        N = self.num_init
        
        print(f"Using designated initialization for {N} primitive(s)")
        
        # Normalize all parameters to lists of N values
        # x, y are in normalized coordinates [0, 1] or absolute pixel coordinates
        x_list = self._normalize_param(self.x_values, N, 0, W)
        y_list = self._normalize_param(self.y_values, N, 0, H)
        
        # Check if x, y are normalized (0-1) or absolute
        # If all values are <= 1, assume normalized coordinates
        if all(0 <= v <= 1 for v in x_list):
            print("  x values appear normalized [0,1], converting to pixel coordinates")
            x_list = [v * W for v in x_list]
        if all(0 <= v <= 1 for v in y_list):
            print("  y values appear normalized [0,1], converting to pixel coordinates")
            y_list = [v * H for v in y_list]
        
        # r (radius)
        r_min = self.radii_min
        r_max = self.radii_max if self.radii_max is not None else 0.5 * min(H, W)
        r_list = self._normalize_param(self.r_values, N, r_min, r_max)
        
        # v (visibility/opacity)
        v_list = self._normalize_param(self.v_values, N, 
                                       self.v_init_bias - 0.5, 
                                       self.v_init_bias + 0.5)
        
        # theta (rotation)
        if self.theta_init is not None:
            # Use base class theta_init if specified
            theta_list = [self.theta_init] * N
        else:
            theta_list = self._normalize_param(self.theta_values, N, 0, 2 * np.pi)
        
        # c (color)
        c_list = self._normalize_color_param(self.c_values, N)
        
        # Convert to tensors
        x = torch.tensor(x_list, dtype=torch.float32, device=device, requires_grad=True)
        y = torch.tensor(y_list, dtype=torch.float32, device=device, requires_grad=True)
        r = torch.tensor(r_list, dtype=torch.float32, device=device, requires_grad=True)
        v = torch.tensor(v_list, dtype=torch.float32, device=device, requires_grad=True)
        theta = torch.tensor(theta_list, dtype=torch.float32, device=device, requires_grad=True)
        c = torch.tensor(c_list, dtype=torch.float32, device=device, requires_grad=True)
        
        # Print initialization summary
        print(f"  Initialized {N} primitive(s):")
        for i in range(min(N, 5)):  # Print first 5
            print(f"    [{i}] x={x[i].item():.1f}, y={y[i].item():.1f}, "
                  f"r={r[i].item():.1f}, θ={theta[i].item():.3f}, "
                  f"v={v[i].item():.3f}, c={c[i].tolist()}")
        if N > 5:
            print(f"    ... and {N-5} more")
        
        return x, y, r, v, theta, c

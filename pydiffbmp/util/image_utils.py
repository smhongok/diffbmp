"""
Image processing utilities for differentiable rendering.
Generic image operations that can be used across different rendering pipelines.
"""

import torch
import torch.nn.functional as F


def rgb_to_grayscale(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to Grayscale using luminance weights.
    
    Args:
        img: (B, 3, H, W) or (B, C, H, W) tensor
        
    Returns:
        (B, 1, H, W) grayscale tensor
    """
    return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]


def apply_gaussian_blur(tensor: torch.Tensor, sigma: float = 3.0, kernel_size: int = 15) -> torch.Tensor:
    """
    Apply Gaussian blur to a tensor.
    
    Args:
        tensor: (C, H, W) tensor to blur
        sigma: Gaussian blur standard deviation (larger = more blur)
        kernel_size: Gaussian kernel size
    
    Returns:
        (C, H, W) blurred tensor
    """
    device = tensor.device
    dtype = tensor.dtype
    
    # Create Gaussian kernel
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    g_kernel = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    
    # Apply Gaussian blur per channel
    blurred = torch.zeros_like(tensor)
    for c in range(tensor.shape[0]):
        tensor_c = tensor[c:c+1].unsqueeze(0)  # (1, 1, H, W)
        blurred[c] = F.conv2d(tensor_c, g_kernel, padding=kernel_size // 2).squeeze()
    
    return blurred

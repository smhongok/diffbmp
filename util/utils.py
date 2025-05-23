import os
import random
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from typing import Tuple

def set_global_seed(seed: int = 42):
    """
    Fix RNG states for Python, NumPy, PyTorch (CPU & CUDA).
    Also forces deterministic behavior in cuDNN.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Gaussian Blur function (2D convolution method)
def gaussian_blur(input_tensor, sigma):
    """
    input_tensor: (N, H, W)
    sigma: Standard deviation of Gaussian kernel (scalar, float)
    """
    if sigma <= 0.0:
        return input_tensor
    # Kernel size is typically rounded 3*sigma with symmetry on both sides (odd size)
    kernel_size = int(2 * round(3 * sigma) + 1)
    # Generate coordinates: kernel center is 0
    ax = torch.arange(kernel_size, dtype=torch.float32, device=input_tensor.device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # 2D kernel: outer product
    kernel2d = kernel[:, None] * kernel[None, :]
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, kernel_size, kernel_size)
    
    padding = kernel_size // 2
    input_tensor = input_tensor.unsqueeze(1)  # (N, 1, H, W)
    blurred = F.conv2d(input_tensor, kernel2d, padding=padding)
    return blurred.squeeze(1)  # (N, H, W)

def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    return 10 * torch.log10(max_val**2 / mse)

def make_batch_indices(N: int, chunk: int, step: int) -> Tuple[slice, slice, slice]:
    """
    Divide depth-sorted [0…N-1] into
        front(F) | learning target(B_t) | back(B)
    three chunks (slices).
    """
    start = (step * chunk) % N
    end   = min(start + chunk, N)
    # For the last batch that would exceed N, don't wrap around circular, just use the smaller chunk
    batch = slice(start, end)                # Section to learn
    fg    = slice(0, start)                  # Front of screen
    bg    = slice(end, N)                    # Back of screen
    return fg, batch, bg

import os
import random
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import string
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

def gaussian_blur(input_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply a 2D Gaussian blur.

    Args:
        input_tensor: Tensor of shape
            - (N, H, W), or
            - (N, C, H, W)
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        - If input was (N, H, W): returns (N, H, W)
        - If input was (N, C, H, W): returns (N, C, H, W)
    """
    if sigma <= 0.0:
        return input_tensor

    # Build 1-D Gaussian kernel
    kernel_size = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(kernel_size, dtype=input_tensor.dtype, device=input_tensor.device) - kernel_size // 2
    kernel1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel1d = kernel1d / kernel1d.sum()

    # Make 2-D separable kernel
    kernel2d = kernel1d[:, None] * kernel1d[None, :]  # (K, K)

    padding = kernel_size // 2

    if input_tensor.dim() == 3:
        # (N, H, W) -> (N, 1, H, W)
        x = input_tensor.unsqueeze(1)
        weight = kernel2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        blurred = F.conv2d(x, weight, padding=padding)
        return blurred.squeeze(1)  # back to (N, H, W)

    elif input_tensor.dim() == 4:
        # (N, C, H, W): do depthwise conv with groups=C
        N, C, H, W = input_tensor.shape
        x = input_tensor
        # Expand weight to (C, 1, K, K) for depthwise
        weight = kernel2d.unsqueeze(0).unsqueeze(0).expand(C, 1, kernel_size, kernel_size)
        blurred = F.conv2d(x, weight, groups=C, padding=padding)
        return blurred  # (N, C, H, W)

    else:
        raise ValueError(f"gaussian_blur: unsupported input shape {input_tensor.shape}")


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


def extract_chars_from_file(
    file, ext,
    remove_whitespace=True,
    remove_punct=False,
    punct_to_remove=None
):
    chars = []
    char_counts = []
    word_lengths_per_line = []  # <--- Added: list of word lengths per line

    # Default characters to remove
    if remove_punct and punct_to_remove is None:
        punct_to_remove = set(string.punctuation)
    elif punct_to_remove is not None:
        punct_to_remove = set(punct_to_remove)

    with open(file, encoding="utf-8") as f:
        if ext == ".lrc":
            import pylrc
            lrc = pylrc.parse(f.read())
            texts = [line.text for line in lrc if line.text.strip()]
        else:
            texts = [line.strip() for line in f if line.strip()]

    for line in texts:
        line_chars = []
        words = line.split()
        word_lengths = []
        for word in words:
            # Count characters excluding whitespace and special characters as word_chars
            word_chars = [
                c for c in word
                if not (remove_whitespace and c.isspace())
                and not (remove_punct and c in punct_to_remove)
            ]
            word_lengths.append(len(word_chars))
            # Add only actually used characters to the full character list
            line_chars.extend(word_chars)
        chars.extend(line_chars)
        char_counts.append(len(line_chars))
        word_lengths_per_line.append(word_lengths)
    return chars, char_counts, word_lengths_per_line
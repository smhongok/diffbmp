import os
import random
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

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

# Gaussian Blur 함수 (2D convolution 방식)
def gaussian_blur(input_tensor, sigma):
    """
    input_tensor: (N, H, W)
    sigma: Gaussian kernel의 표준편차 (스칼라, float)
    """
    if sigma <= 0.0:
        return input_tensor
    # kernel 크기는 보통 3*sigma를 반올림한 값의 양쪽 대칭 (홀수 크기)
    kernel_size = int(2 * round(3 * sigma) + 1)
    # 좌표 생성: kernel 중심이 0이 되도록
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
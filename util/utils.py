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

def make_batch_indices(N: int, chunk: int, step: int) -> Tuple[slice, slice, slice]:
    """
    깊이 정렬된 [0…N-1]를
        앞쪽(F) | 학습 대상(B_t) | 뒤쪽(B)
    세 덩어리(slice)로 나눈다.
    """
    start = (step * chunk) % N
    end   = min(start + chunk, N)
    # 마지막 배치가 N을 넘어갈 때는 circular 하지 않고 그대로 작은 덩어리
    batch = slice(start, end)                # 학습할 구간
    fg    = slice(0, start)                  # 화면 앞
    bg    = slice(end, N)                    # 화면 뒤
    return fg, batch, bg

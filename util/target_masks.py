import cv2
import numpy as np
import torch
from PIL import Image

def DT_L2(binary_image, device: torch.device) -> torch.Tensor:
    """
    Computes the L2 distance transform of the alpha channel of the binary image.
    Foreground pixels are considered as 0, and background pixels as 0~255 corresponding to the L2 distance from the foreground.
    Args:
        binary_image (np.ndarray): Target binary image.
        device (torch.device): Device to which the output tensor should be moved.
    Returns:
        torch.Tensor: Distance transform of the binary image, moved to the specified device.
    """
    target_dist_mask = cv2.distanceTransform(
    binary_image, 
    distanceType=cv2.DIST_L2, 
    maskSize=cv2.DIST_MASK_PRECISE
    )
    target_dist_mask = torch.from_numpy(target_dist_mask).to(device)

    return target_dist_mask

def SADT_L2(binary_image, device: torch.device, alpha = 2.0) -> torch.Tensor:
    """
    Computes the SAD (Skeleton-Aware Distance) transform of the alpha channel of the binary image.
    Foreground pixels and boundary pixels are considered as 0, and the skeleton of background pixels as 1.
    If alpha > 1, the distance is convex with respect to the boundary distance. Else, it is concave.
    Args:
        binary_image (np.ndarray): Target binary image.
        device (torch.device): Device to which the output tensor should be moved.
        alpha (float): Exponent for the distance transform. Default is 2.0.
    Returns:
        torch.Tensor: SADT of the binary image, moved to the specified device.
    """

    boundary_image = 255-binary_image

    from skimage.morphology import skeletonize

    skeleton = skeletonize(binary_image // 255) # Convert to boolean for skeletonize
    skeleton_image = (skeleton * 255).astype(np.uint8)

    db = cv2.distanceTransform(255 - boundary_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # boundary distance
    ds = cv2.distanceTransform(255 - skeleton_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)   # skeleton distance

    dist = (db/(db+ds + 1e-6))** alpha 

    dist_normalized = dist / dist.max() if dist.max() > 0 else dist

    dist_img_gray = (dist_normalized * 255).astype(np.uint8)

    target_dist_mask = torch.from_numpy(dist_img_gray).to(device) 

    return target_dist_mask

def binary_mask(I_target: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Creates a binary mask from the alpha channel of the input tensor.
    Foreground pixels are set to 0, and background pixels are set to 1.
    Args:
        I_target (torch.Tensor): Input tensor with shape (R, G, B, A).
    Returns:
        torch.Tensor: Binary mask of the alpha channel.
    """
    if I_target.shape[2] < 4:
        raise ValueError("Input tensor must shape (H, W, 4). 4 for RGBA channels.")
    
    target_binary_mask = torch.from_numpy(1-(I_target[:, :, 3] > 0).cpu().numpy().astype(np.uint8)).to(device) 
    
    return target_binary_mask
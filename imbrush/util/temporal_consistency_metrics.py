"""Temporal Consistency Metrics for Video Rendering Evaluation

Implements three key metrics:
1. E_warp: Warping error using optical flow (from fast_blind_video_consistency)
2. tOF: Temporal optical flow difference (from TecoGAN)
3. tLP: Temporal LPIPS perceptual difference (from TecoGAN)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import piq
from typing import List, Tuple, Dict


def _nocclusion_mask_fb(flow_fw, flow_bw, alpha1=0.01, alpha0=0.5):
    """
    Motion-adaptive forward–backward consistency (Lai et al.):
    returns NON-occlusion mask M in {0,1}, shape [H,W]
    """
    H, W = flow_fw.shape[:2]
    # sample backward flow at forward-displaced coords (warp bwd by fwd)
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_w = np.clip(x + flow_fw[..., 0], 0, W - 1).astype(np.float32)
    y_w = np.clip(y + flow_fw[..., 1], 0, H - 1).astype(np.float32)
    bwx = cv2.remap(flow_bw[..., 0], x_w, y_w, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bwy = cv2.remap(flow_bw[..., 1], x_w, y_w, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    fb_x = flow_fw[..., 0] + bwx
    fb_y = flow_fw[..., 1] + bwy
    fb_err2 = fb_x**2 + fb_y**2

    mag_fw2  = (flow_fw[..., 0]**2 + flow_fw[..., 1]**2)
    mag_bww2 = (bwx**2 + bwy**2)

    thresh = alpha1 * (mag_fw2 + mag_bww2) + alpha0
    M = (fb_err2 <= thresh).astype(np.float32)  # 1 = non-occluded
    return M



def _warp_flow(img, flow_bwd, border=cv2.BORDER_REPLICATE):
    """
    Backward warp: for each pixel (x,y) in CURRENT frame,
    sample img at (x + u_bwd(x,y), y + v_bwd(x,y)).
    img: [H,W,3] or [C,H,W] (float32)
    flow_bwd: [H,W,2] (dx, dy), mapping CURRENT->PREVIOUS
    """
    if img.ndim == 3 and img.shape[0] in (1,3):  # CHW -> HWC
        img = img.transpose(1,2,0)
    H, W = img.shape[:2]
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    map_x = (x + flow_bwd[...,0]).astype(np.float32)
    map_y = (y + flow_bwd[...,1]).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border)
    return warped



def _compute_optical_flow(img1, img2):
    """Compute optical flow between two images using Farneback method.
    
    Args:
        img1: First image [H, W, 3] in range [0, 255]
        img2: Second image [H, W, 3] in range [0, 255]
        
    Returns:
        Optical flow [H, W, 2]
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Compute optical flow using Farneback method
    # Parameters from TecoGAN: (0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    return flow


def _crop_8x8(img):
    """Crop image to be divisible by 32 (from TecoGAN).
    
    Args:
        img: Input image [H, W, ...]
        
    Returns:
        Cropped image, y_offset, x_offset
    """
    ori_h, ori_w = img.shape[:2]
    
    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32
    
    while h > ori_h - 16:
        h = h - 32
    while w > ori_w - 16:
        w = w - 32
    
    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    
    crop_img = img[y:y+h, x:x+w]
    return crop_img, y, x


def compute_E_warp(
    rendered_frames: List[np.ndarray],
    target_frames: List[np.ndarray],
    occlusion_threshold: float = 1.0
) -> Dict[str, float]:
    """Compute E_warp (Warping Error) metric for temporal consistency."""
    n_frames = len(rendered_frames)
    
    if n_frames < 2:
        print("Warning: E_warp requires at least 2 frames. Returning 0.")
        return {'E_warp': 0.0}
    
    warp_errors = []
    
    for t in range(1, n_frames):
        # Get consecutive frames (float32, [0,255])
        tgt_prev = target_frames[t - 1].astype(np.float32)
        tgt_curr = target_frames[t].astype(np.float32)
        ren_prev = rendered_frames[t - 1].astype(np.float32)
        ren_curr = rendered_frames[t].astype(np.float32)

        # FLOW ON TARGETS ONLY (Lai protocol)
        flow_fw = _compute_optical_flow(tgt_prev, tgt_curr)   # F_{t-1->t}
        flow_bw = _compute_optical_flow(tgt_curr, tgt_prev)   # F_{t->t-1}

        # Occlusion mask (non-occluded = 1), motion-adaptive
        M_occ = _nocclusion_mask_fb(flow_fw, flow_bw, alpha1=0.01, alpha0=0.5)  # (H,W) in {0,1}

        # BACKWARD warp: need F_{t->t-1} defined on the CURRENT grid
        warped_prev = _warp_flow(ren_prev, flow_bw, border=cv2.BORDER_REPLICATE)  # W(O_{t-1}; F_{t->t-1})

        # Per Lai: L2^2 over color, averaged over non-occluded pixels
        diff = (warped_prev - ren_curr) ** 2  # (H,W,3)
        per_pixel = diff.sum(axis=2)          # sum over channels
        denom = np.maximum(M_occ.sum(), 1.0)
        error = (per_pixel * M_occ).sum() / denom / (255**2)
        warp_errors.append(error)
    
    # Compute average
    avg_warp_error = np.mean(warp_errors)
    
    return {
        'E_warp': avg_warp_error,
        'E_warp_per_frame': warp_errors
    }


def compute_tOF(
    rendered_frames: List[np.ndarray],
    target_frames: List[np.ndarray],
    crop_border: bool = True
) -> Dict[str, float]:
    """Compute tOF (temporal Optical Flow) metric.
    
    This metric compares optical flow between consecutive frames in
    target vs rendered sequences.
    
    Args:
        rendered_frames: List of rendered frames [H, W, 3], values in [0, 255]
        target_frames: List of target frames [H, W, 3], values in [0, 255]
        crop_border: Whether to crop to be divisible by 32
        
    Returns:
        Dictionary with 'tOF' and per-frame errors
    """
    n_frames = len(rendered_frames)
    
    if n_frames < 2:
        print("Warning: tOF requires at least 2 frames. Returning 0.")
        return {'tOF': 0.0}
    
    tof_errors = []
    
    for t in range(1, n_frames):
        # Get consecutive frames
        target_prev = target_frames[t - 1].astype(np.float32)
        target_curr = target_frames[t].astype(np.float32)
        rendered_prev = rendered_frames[t - 1].astype(np.float32)
        rendered_curr = rendered_frames[t].astype(np.float32)
        
        # Compute optical flow for target sequence
        target_flow = _compute_optical_flow(target_prev, target_curr)
        
        # Compute optical flow for rendered sequence
        rendered_flow = _compute_optical_flow(rendered_prev, rendered_curr)
        
        # Crop flows if requested
        if crop_border:
            target_flow, _, _ = _crop_8x8(target_flow)
            rendered_flow, _, _ = _crop_8x8(rendered_flow)
        
        # Compute difference
        flow_diff = np.absolute(target_flow - rendered_flow)
        flow_diff_magnitude = np.sqrt(np.sum(flow_diff * flow_diff, axis=-1))
        
        # Average error for this frame pair
        error = flow_diff_magnitude.mean()
        tof_errors.append(error)
    
    # Compute average
    avg_tof = np.mean(tof_errors)
    
    return {
        'tOF': avg_tof,
        'tOF_per_frame': tof_errors
    }


def compute_tLP(
    rendered_frames: List[np.ndarray],
    target_frames: List[np.ndarray],
    crop_border: bool = True
) -> Dict[str, float]:
    """Compute tLP (temporal LPIPS) metric.
    
    This metric compares perceptual temporal changes between
    target and rendered sequences using LPIPS.
    
    Args:
        rendered_frames: List of rendered frames [H, W, 3], values in [0, 255]
        target_frames: List of target frames [H, W, 3], values in [0, 255]
        crop_border: Whether to crop to be divisible by 32
        
    Returns:
        Dictionary with 'tLP' and per-frame errors
    """
    n_frames = len(rendered_frames)
    
    if n_frames < 2:
        print("Warning: tLP requires at least 2 frames. Returning 0.")
        return {'tLP': 0.0}
    
    # Initialize LPIPS model using piq
    lpips_model = piq.LPIPS()
    
    tlp_errors = []
    
    # Helper function to convert image to tensor
    def img_to_tensor(img):
        """Convert numpy image [H, W, 3] in [0, 255] to tensor [1, 3, H, W] in [0, 1]"""
        if crop_border:
            img, _, _ = _crop_8x8(img)
        
        # Normalize to [0, 1] (piq.LPIPS expects this range)
        img_norm = img / 255.0
        
        # Convert to tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        return img_tensor
    
    with torch.no_grad():
        for t in range(1, n_frames):
            # Get consecutive frames and convert to tensors
            target_prev_t = img_to_tensor(target_frames[t - 1].astype(np.float32))
            target_curr_t = img_to_tensor(target_frames[t].astype(np.float32))
            rendered_prev_t = img_to_tensor(rendered_frames[t - 1].astype(np.float32))
            rendered_curr_t = img_to_tensor(rendered_frames[t].astype(np.float32))
            
            # Compute LPIPS distance between consecutive target frames
            target_lpips = lpips_model(target_prev_t, target_curr_t).item()
            
            # Compute LPIPS distance between consecutive rendered frames
            rendered_lpips = lpips_model(rendered_prev_t, rendered_curr_t).item()
            
            # Temporal LPIPS: absolute difference multiplied by 100 (as in TecoGAN)
            tlp_error = np.abs(target_lpips - rendered_lpips) * 100.0
            tlp_errors.append(tlp_error)
    
    # Compute average
    avg_tlp = np.mean(tlp_errors)
    
    return {
        'tLP': avg_tlp,
        'tLP_per_frame': tlp_errors
    }


def compute_all_temporal_metrics(
    rendered_frames: List[np.ndarray],
    target_frames: List[np.ndarray],
    compute_warp: bool = True,
    compute_of: bool = True,
    compute_lp: bool = True
) -> Dict[str, float]:
    """Compute all temporal consistency metrics.
    
    Args:
        rendered_frames: List of rendered frames [H, W, 3], values in [0, 255]
        target_frames: List of target frames [H, W, 3], values in [0, 255]
        compute_warp: Whether to compute E_warp
        compute_of: Whether to compute tOF
        compute_lp: Whether to compute tLP
        
    Returns:
        Dictionary with all computed metrics
    """
    results = {}
    
    if compute_warp:
        print("Computing E_warp...")
        warp_results = compute_E_warp(rendered_frames, target_frames)
        results.update(warp_results)
    
    if compute_of:
        print("Computing tOF...")
        tof_results = compute_tOF(rendered_frames, target_frames)
        results.update(tof_results)
    
    if compute_lp:
        print("Computing tLP...")
        tlp_results = compute_tLP(rendered_frames, target_frames)
        results.update(tlp_results)
    
    return results

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


# def _nocclusion_mask_fb(flow_fw, flow_bw, alpha1=0.01, alpha0=0.5):
#     """
#     Motion-adaptive forward–backward consistency (Lai et al.):
#     returns NON-occlusion mask M in {0,1}, shape [H,W]
#     """
#     H, W = flow_fw.shape[:2]
#     # sample backward flow at forward-displaced coords (warp bwd by fwd)
#     y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
#     x_w = np.clip(x + flow_fw[..., 0], 0, W - 1).astype(np.float32)
#     y_w = np.clip(y + flow_fw[..., 1], 0, H - 1).astype(np.float32)
#     bwx = cv2.remap(flow_bw[..., 0], x_w, y_w, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#     bwy = cv2.remap(flow_bw[..., 1], x_w, y_w, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

#     fb_x = flow_fw[..., 0] + bwx
#     fb_y = flow_fw[..., 1] + bwy
#     fb_err2 = fb_x**2 + fb_y**2

#     mag_fw2  = (flow_fw[..., 0]**2 + flow_fw[..., 1]**2)
#     mag_bww2 = (bwx**2 + bwy**2)

#     thresh = alpha1 * (mag_fw2 + mag_bww2) + alpha0
#     M = (fb_err2 <= thresh).astype(np.float32)  # 1 = non-occluded
#     return M

def _nocclusion_mask_fb(flow_fw, flow_bw, alpha1=0.01, alpha0=0.5):
    H, W = flow_fw.shape[:2]
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_w = x + flow_fw[..., 0]
    y_w = y + flow_fw[..., 1]

    oob = (x_w < 0) | (x_w > W-1) | (y_w < 0) | (y_w > H-1)

    bwx = cv2.remap(flow_bw[..., 0], x_w.astype(np.float32), y_w.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bwy = cv2.remap(flow_bw[..., 1], x_w.astype(np.float32), y_w.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    fb_x = flow_fw[..., 0] + bwx
    fb_y = flow_fw[..., 1] + bwy
    fb_err2 = fb_x**2 + fb_y**2

    mag_fw2  = (flow_fw[..., 0]**2 + flow_fw[..., 1]**2)
    mag_bww2 = (bwx**2 + bwy**2)

    thresh = alpha1 * (mag_fw2 + mag_bww2) + alpha0

    M = (fb_err2 <= thresh).astype(np.float32)
    M[oob] = 0.0
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
        #print("DEBUG denom:", denom)
        error = (per_pixel * M_occ).sum() / denom / (255.0**2)
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


if __name__ == "__main__":

    # ---------- helpers ----------
    def _make_textured_image(H=240, W=320):
        """RGB image with plenty of gradients/textures for robust flow."""
        img = np.zeros((H, W, 3), np.uint8)
        # big gradient
        xv, yv = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        img[..., 0] = (255 * xv).astype(np.uint8)
        img[..., 1] = (255 * yv).astype(np.uint8)
        img[..., 2] = (255 * (0.5 * xv + 0.5 * yv)).astype(np.uint8)
        # add features
        for r in range(10, min(H, W), 30):
            cv2.circle(img, (W//2, H//2), r, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        for x in range(0, W, 20):
            cv2.line(img, (x, 0), (x, H-1), (40, 40, 40), 1)
        for y in range(0, H, 20):
            cv2.line(img, (0, y), (W-1, y), (40, 40, 40), 1)
        return img

    def _translate(img, dx, dy, border=cv2.BORDER_REFLECT):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=border)

    def _compute_E_warp_nomask(rendered_frames: List[np.ndarray],
                               target_frames: List[np.ndarray]) -> float:
        """Same as compute_E_warp but with M_occ = 1 everywhere (diagnostic)."""
        n_frames = len(rendered_frames)
        if n_frames < 2:
            return 0.0
        errs = []
        for t in range(1, n_frames):
            tgt_prev = target_frames[t-1].astype(np.float32)
            tgt_curr = target_frames[t].astype(np.float32)
            ren_prev = rendered_frames[t-1].astype(np.float32)
            ren_curr = rendered_frames[t].astype(np.float32)
            # flows on targets
            flow_fw = _compute_optical_flow(tgt_prev, tgt_curr)
            flow_bw = _compute_optical_flow(tgt_curr, tgt_prev)
            # backward warp prev rendered to current grid
            warped_prev = _warp_flow(ren_prev, flow_bw, border=cv2.BORDER_REPLICATE)
            diff = (warped_prev - ren_curr) ** 2
            per_pixel = diff.sum(axis=2)
            denom = per_pixel.size  # all pixels
            # keep the same 255^2 scaling convention as your compute_E_warp
            err = per_pixel.sum() / max(denom, 1) / (255.0 ** 2)
            errs.append(err)
        return float(np.mean(errs))

    # ---------- Test A: Pure translation ----------
    H, W = 240, 320
    base = _make_textured_image(H, W)
    dx, dy = 5, -3  # pixels
    t0 = base
    t1 = _translate(base, dx, dy)

    rendered_frames = [t0.copy(), t1.copy()]  # rendered == target for sanity
    target_frames   = [t0.copy(), t1.copy()]

    print("\n[Pure Translation Test] Shift = (%d, %d)" % (dx, dy))
    res = compute_E_warp(rendered_frames, target_frames)
    e_masked = res["E_warp"]
    e_nomask = _compute_E_warp_nomask(rendered_frames, target_frames)
    print("E_warp (masked):   %.8f" % e_masked)
    print("E_warp (no mask):  %.8f" % e_nomask)
    print("Note: masked should be ~0 (tiny >0 due to interpolation).")

    # ---------- Test B: Mask stress ----------
    # Move a black square to create true occlusions/disocclusions.
    scene0 = _make_textured_image(H, W)
    scene1 = scene0.copy()
    # Draw square in frame 0 at pos A, move to pos B in frame 1
    A = (60, 60); B = (90, 90); side = 60
    cv2.rectangle(scene0, A, (A[0]+side, A[1]+side), (0, 0, 0), -1)
    cv2.rectangle(scene1, B, (B[0]+side, B[1]+side), (0, 0, 0), -1)

    rendered_frames2 = [scene0.copy(), scene1.copy()]
    target_frames2   = [scene0.copy(), scene1.copy()]

    print("\n[Mask Stress Test] Moving occluder square")
    res2 = compute_E_warp(rendered_frames2, target_frames2)
    e_masked2 = res2["E_warp"]
    e_nomask2 = _compute_E_warp_nomask(rendered_frames2, target_frames2)
    print("E_warp (masked):   %.8f" % e_masked2)
    print("E_warp (no mask):  %.8f" % e_nomask2)
    print("Expectation: masked << no-mask (occluded regions excluded).")

    # ================================
    # EXTRA SANITY / FAILURE-DIAG TESTS
    # ================================

    def _flow_epe_stats(flow, dx, dy, name="fw"):
        """Endpoint error stats vs known constant translation (dx,dy)."""
        true = np.dstack([
            np.full_like(flow[...,0], dx, dtype=np.float32),
            np.full_like(flow[...,1], dy, dtype=np.float32)
        ])
        e = flow - true
        epe = np.sqrt(e[...,0]**2 + e[...,1]**2)
        return {
            "name": name,
            "mean_epe": float(np.mean(epe)),
            "med_epe": float(np.median(epe)),
            "p95_epe": float(np.percentile(epe, 95.0)),
        }

    def _compute_E_warp_raw255(rendered_frames, target_frames):
        """Re-implements your E_warp with 0–255 inputs (dividing by 255^2)."""
        n = len(rendered_frames)
        errs = []
        for t in range(1, n):
            tgt_prev = target_frames[t-1].astype(np.float32)
            tgt_curr = target_frames[t].astype(np.float32)
            ren_prev = rendered_frames[t-1].astype(np.float32)
            ren_curr = rendered_frames[t].astype(np.float32)
            flow_fw = _compute_optical_flow(tgt_prev, tgt_curr)   # F_{t-1->t}
            flow_bw = _compute_optical_flow(tgt_curr, tgt_prev)   # F_{t->t-1}
            M_occ   = _nocclusion_mask_fb(flow_fw, flow_bw, 0.01, 0.5)
            warped_prev = _warp_flow(ren_prev, flow_bw, border=cv2.BORDER_REPLICATE)
            diff = (warped_prev - ren_curr) ** 2
            per_pixel = diff.sum(axis=2)
            denom = max(M_occ.sum(), 1.0)
            err = (per_pixel * M_occ).sum() / denom / (255.0**2)
            errs.append(err)
        return float(np.mean(errs))

    def _compute_E_warp_unit01(rendered_frames, target_frames):
        """Same logic but normalizes frames to 0–1 and does NOT divide by 255^2.
           Should numerically match _compute_E_warp_raw255 within ~1e-7."""
        n = len(rendered_frames)
        errs = []
        for t in range(1, n):
            tgt_prev = (target_frames[t-1].astype(np.float32) / 255.0)
            tgt_curr = (target_frames[t].astype(np.float32) / 255.0)
            ren_prev = (rendered_frames[t-1].astype(np.float32) / 255.0)
            ren_curr = (rendered_frames[t].astype(np.float32) / 255.0)
            flow_fw = _compute_optical_flow(tgt_prev*255.0, tgt_curr*255.0)  # Farnebäck expects 0–255 uint8-like contrast; multiply back
            flow_bw = _compute_optical_flow(tgt_curr*255.0, tgt_prev*255.0)
            M_occ   = _nocclusion_mask_fb(flow_fw, flow_bw, 0.01, 0.5)
            warped_prev = _warp_flow(ren_prev, flow_bw, border=cv2.BORDER_REPLICATE)
            diff = (warped_prev - ren_curr) ** 2
            per_pixel = diff.sum(axis=2)
            denom = max(M_occ.sum(), 1.0)
            err = (per_pixel * M_occ).sum() / denom
            errs.append(err)
        return float(np.mean(errs))

    print("\n[Sanity C] Identity-frames test (should be ~0)")
    ident0 = _make_textured_image(H, W)
    rendered_I = [ident0.copy(), ident0.copy()]
    target_I   = [ident0.copy(), ident0.copy()]
    eI_masked = compute_E_warp(rendered_I, target_I)["E_warp"]
    eI_nomask = _compute_E_warp_nomask(rendered_I, target_I)
    print(f"E_warp (masked) = {eI_masked:.10f} | (no mask) = {eI_nomask:.10f}")
    if eI_masked > 1e-3 or eI_nomask > 1e-3:
        print("FAIL ❌  Identity test not ~0. Check warping/mask math or dtype scaling.")
    else:
        print("PASS ✅  Identity test ~0.")

    print("\n[Sanity E] Scale-consistency (0–255 with /255^2) == (0–1 without)")
    e_raw255 = _compute_E_warp_raw255(rendered_frames2, target_frames2)
    e_unit01 = _compute_E_warp_unit01(rendered_frames2, target_frames2)
    print(f"raw255={e_raw255:.10f}  unit01={e_unit01:.10f}  | abs diff={abs(e_raw255-e_unit01):.2e}")
    if abs(e_raw255 - e_unit01) > 1e-6:
        print("FAIL ❌  Scale mismatch. Ensure you either normalize to 0–1 OR divide by 255^2 (consistently).")
    else:
        print("PASS ✅  Scale handling is consistent.")

    print("\n[Sanity F] Border-OOB robustness (large shift; masked << no-mask)")
    dx_big, dy_big = 40, 25
    tb0 = _make_textured_image(H, W)
    tb1 = _translate(tb0, dx_big, dy_big)  # this creates large OOB areas
    rf_b = [tb0.copy(), tb1.copy()]
    tf_b = [tb0.copy(), tb1.copy()]
    e_masked_b = compute_E_warp(rf_b, tf_b)["E_warp"]
    e_nomask_b = _compute_E_warp_nomask(rf_b, tf_b)
    print(f"Large shift ({dx_big},{dy_big}) -> masked={e_masked_b:.6f}  no-mask={e_nomask_b:.6f}")
    if not (e_masked_b < e_nomask_b):
        print("FAIL ❌  Mask not protecting borders/OOB well. Tighten FB mask OOB handling.")
    else:
        print("PASS ✅  Mask suppresses border/OOB errors.")

    print("\n[Sanity G] Flow EPE on known translation (quality proxy)")
    # reuse pure translation setup (dx,dy) and targets
    fw = _compute_optical_flow(t0, t1)
    bw = _compute_optical_flow(t1, t0)
    stats_fw = _flow_epe_stats(fw, dx, dy, "fw")
    stats_bw = _flow_epe_stats(bw, -dx, -dy, "bw")  # backward flow should be (-dx,-dy)
    print(f"FW EPE  mean={stats_fw['mean_epe']:.3f}px  median={stats_fw['med_epe']:.3f}px  p95={stats_fw['p95_epe']:.3f}px")
    print(f"BW EPE  mean={stats_bw['mean_epe']:.3f}px  median={stats_bw['med_epe']:.3f}px  p95={stats_bw['p95_epe']:.3f}px")
    if stats_fw["mean_epe"] > 0.5 or stats_bw["mean_epe"] > 0.5:
        print("WARN ⚠️  Farnebäck flow is sloppy on this synthetic. Consider RGB/BGR fix or stronger params.")
    else:
        print("OK ✅  Flow accuracy is reasonable on the synthetic.")




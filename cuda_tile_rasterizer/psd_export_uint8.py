import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import os
import sys

# Try to import the compiled CUDA extension
try:
    import psd_export_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: PSD export CUDA extension not available. Falling back to PyTorch implementation.")

def export_psd_layers_cuda_uint8(
    renderer_S: torch.Tensor,        # [P, H_t, W_t] primitive templates
    x: torch.Tensor,                 # [N] x positions
    y: torch.Tensor,                 # [N] y positions
    r: torch.Tensor,                 # [N] radii
    theta: torch.Tensor,             # [N] rotations
    v: torch.Tensor,                 # [N] visibility (sigmoid applied)
    c: torch.Tensor,                 # [N, 3] colors (sigmoid applied)
    global_bmp_sel: torch.Tensor,    # [N] template selection indices
    H: int, W: int,                  # canvas dimensions
    scale_factor: float = 1.0,       # export scaling
    alpha_upper_bound: float = 1.0   # maximum alpha value
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
    """
    Export PSD layers using 2-stage CUDA kernel for memory-efficient cropped output.
    
    Returns:
        (PIL image list, bounding box list)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA PSD export extension not available")
    
    N = len(x)
    device = x.device
    
    # Convert inputs to float32 for computation precision
    x_f32 = x.float()
    y_f32 = y.float()
    r_f32 = r.float()
    theta_f32 = theta.float()
    v_f32 = v.float()  # Already sigmoid applied
    c_f32 = c.float()  # Already sigmoid applied
    renderer_S_f32 = renderer_S.float()
    
    # Stage 1: Compute bounding boxes
    bounding_boxes = torch.zeros(N, 4, dtype=torch.int32, device=device)
    cropped_sizes = torch.zeros(N, 2, dtype=torch.int32, device=device)
    
    # C++ binding expects: means2D, radii, rotations, primitive_templates, global_bmp_sel, bounding_boxes, cropped_sizes, H, W, scale_factor, alpha_upper_bound
    means2D = torch.stack([x_f32, y_f32], dim=1)  # [N, 2]
    psd_export_cuda.compute_bounding_boxes(
        means2D, r_f32, theta_f32, renderer_S_f32, global_bmp_sel.int(),
        bounding_boxes, cropped_sizes, H, W, scale_factor, alpha_upper_bound
    )
    
    # Calculate total memory requirement on CPU
    h_cropped_sizes = cropped_sizes.cpu().numpy()
    total_pixels = 0
    layer_offsets = []
    
    for i in range(N):
        layer_offsets.append(total_pixels)
        w, h = h_cropped_sizes[i]
        total_pixels += w * h * 4  # RGBA
    
    if total_pixels == 0:
        # All primitives are empty
        empty_images = [Image.fromarray(np.zeros((1, 1, 4), dtype=np.uint8), 'RGBA') for _ in range(N)]
        empty_bounds = [(0, 0, 1, 1) for _ in range(N)]
        return empty_images, empty_bounds
    
    # Allocate cropped output buffer (only needed pixels, not full canvas)
    cropped_buffer = torch.zeros(total_pixels, dtype=torch.uint8, device=device)
    layer_offsets_tensor = torch.tensor(layer_offsets, dtype=torch.int32, device=device)
    
    # Stage 2: Generate cropped layers
    # C++ binding expects: means2D, radii, rotations, colors, visibility, primitive_templates, global_bmp_sel, bounding_boxes, layer_offsets, cropped_output_buffer, H, W, scale_factor, alpha_upper_bound
    means2D = torch.stack([x_f32, y_f32], dim=1)  # [N, 2]
    psd_export_cuda.generate_cropped_layers(
        means2D, r_f32, theta_f32, c_f32, v_f32,
        renderer_S_f32, global_bmp_sel.int(), bounding_boxes,
        layer_offsets_tensor, cropped_buffer, H, W, scale_factor, alpha_upper_bound
    )
    
    # Convert to PIL images
    pil_images = []
    bounds_list = []
    h_bounding_boxes = bounding_boxes.cpu().numpy()
    h_cropped_buffer = cropped_buffer.cpu().numpy()
    
    for i in range(N):
        left, top, right, bottom = h_bounding_boxes[i]
        w, h = h_cropped_sizes[i]
        
        if w <= 1 or h <= 1:  # Empty layer
            pil_images.append(Image.fromarray(np.zeros((1, 1, 4), dtype=np.uint8), 'RGBA'))
            bounds_list.append((0, 0, 1, 1))
        else:
            # Extract cropped data
            offset = layer_offsets[i]
            layer_data = h_cropped_buffer[offset:offset + w * h * 4]
            rgba_array = layer_data.reshape(h, w, 4)
            
            pil_images.append(Image.fromarray(rgba_array, 'RGBA'))
            bounds_list.append((left, top, right, bottom))
    
    return pil_images, bounds_list


def export_psd_layers_pytorch_fallback(
    renderer_S: torch.Tensor,
    x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, theta: torch.Tensor,
    v: torch.Tensor, c: torch.Tensor, global_bmp_sel: torch.Tensor,
    H: int, W: int, scale_factor: float = 1.0, alpha_upper_bound: float = 1.0
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
    """
    Fallback PyTorch implementation matching the existing PSDExporter logic.
    This is used when CUDA extension is not available.
    """
    import torch.nn.functional as F
    
    N = len(x)
    device = x.device
    
    # Apply transformations using batched F.grid_sample (matching existing logic)
    # Create coordinate grids
    X, Y = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing='xy'
    )
    
    # Handle template selection
    if renderer_S.dim() == 2:
        templates = renderer_S.unsqueeze(0).expand(N, -1, -1)
    else:
        p = renderer_S.shape[0]
        template_indices = torch.arange(N, device=device) % p
        templates = renderer_S[template_indices]
    
    # Batch process transformations
    transformed_templates = []
    for i in range(N):
        # Normalize coordinates to [-1, 1]
        u_norm = (X - x[i] * scale_factor) / (r[i] * scale_factor)
        v_norm = (Y - y[i] * scale_factor) / (r[i] * scale_factor)
        
        # Apply inverse rotation
        cos_t = torch.cos(theta[i])
        sin_t = torch.sin(theta[i])
        u_rot = cos_t * u_norm + sin_t * v_norm
        v_rot = -sin_t * u_norm + cos_t * v_norm
        
        # Create grid for F.grid_sample
        grid = torch.stack([u_rot, v_rot], dim=-1).unsqueeze(0)
        template_4d = templates[i].unsqueeze(0).unsqueeze(0)
        
        # Grid sample
        sampled = F.grid_sample(template_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        transformed_templates.append(sampled.squeeze())
    
    # Convert to PIL images with bounds
    pil_images = []
    bounds_list = []
    
    for i in range(N):
        template = transformed_templates[i].detach().cpu().numpy()
        rgb = torch.sigmoid(c[i]).detach().cpu().numpy()
        vis = torch.sigmoid(v[i]).detach().cpu().numpy() * alpha_upper_bound
        
        # Create RGBA
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        template_mask = template > 0
        
        for ch in range(3):
            rgba[:, :, ch] = (template_mask * rgb[ch] * 255).astype(np.uint8)
        rgba[:, :, 3] = (template * vis * 255).astype(np.uint8)
        
        # Find bounding box
        alpha_mask = rgba[:, :, 3] > 0
        if not alpha_mask.any():
            pil_images.append(Image.fromarray(np.zeros((1, 1, 4), dtype=np.uint8), 'RGBA'))
            bounds_list.append((0, 0, 1, 1))
            continue
        
        rows = np.any(alpha_mask, axis=1)
        cols = np.any(alpha_mask, axis=0)
        top, bottom = np.where(rows)[0][[0, -1]]
        left, right = np.where(cols)[0][[0, -1]]
        
        bottom += 1
        right += 1
        
        # Crop and create PIL image
        cropped_rgba = rgba[top:bottom, left:right]
        pil_images.append(Image.fromarray(cropped_rgba, 'RGBA'))
        bounds_list.append((left, top, right, bottom))
    
    return pil_images, bounds_list

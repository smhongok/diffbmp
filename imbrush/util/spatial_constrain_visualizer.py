import numpy as np
from PIL import Image

def save_spatial_constraints(rendered, rendered_alpha, output_path):
    RGB_img = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
    A_img = (rendered_alpha.detach().cpu().numpy() * 255).astype(np.uint8)
    
    # Add alpha dimension if needed and concatenate along channel axis
    if len(A_img.shape) == 2:  # If alpha is 2D (H, W)
        A_img = A_img[..., np.newaxis]  # Make it (H, W, 1)
    elif len(A_img.shape) == 3 and A_img.shape[-1] == 1:  # Already (H, W, 1)
        pass
    else:
        raise ValueError(f"Unexpected alpha shape: {A_img.shape}")
    
    RGBA_img = np.concatenate([RGB_img, A_img], axis=-1)
    Image.fromarray(RGBA_img, mode='RGBA').save(output_path.replace('.png', '_no_bg.png'))
    prim_distribution = (rendered_alpha.detach().cpu().numpy()==0).astype(np.uint8)
    alpha_uint8 = (rendered_alpha.detach().cpu().numpy() * 255).astype(np.uint8)
    alpha_uint8 = 255 - alpha_uint8  # Invert alpha for visualization
    Image.fromarray(alpha_uint8, mode='L').save(output_path.replace('.png', '_alpha.png'))
    Image.fromarray(prim_distribution*255, mode='L').save(output_path.replace('.png', '_prim_distribution.png'))
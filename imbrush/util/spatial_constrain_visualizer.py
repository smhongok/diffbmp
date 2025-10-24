import numpy as np
from PIL import Image

def save_spatial_constraints(rendered_alpha, output_path):
    prim_distribution = (rendered_alpha.detach().cpu().numpy()>0).astype(np.uint8)
    alpha_uint8 = (rendered_alpha.detach().cpu().numpy() * 255).astype(np.uint8)
    alpha_uint8 = 255 - alpha_uint8  # Invert alpha for visualization
    Image.fromarray(alpha_uint8, mode='L').save(output_path.replace('.png', '_alpha.png'))
    Image.fromarray(prim_distribution*255, mode='L').save(output_path.replace('.png', '_prim_distribution.png'))
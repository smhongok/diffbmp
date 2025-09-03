import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer

class PSDExporter:
    """Exports individual transformed primitives as PSD layers using psd-tools."""
    
    def __init__(self, canvas_width: int, canvas_height: int,
                 alpha_upper_bound: float = 1.0):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.alpha_upper_bound = alpha_upper_bound
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create new PSD document with transparent background
        self.psd = PSDImage.new('RGBA', (canvas_width, canvas_height), color=(0, 0, 0, 0))
        
    def add_layer_from_primitive(self, primitive_template: torch.Tensor, 
                               x: float, y: float, r: float, theta: float,
                               v: float, c: torch.Tensor, name: str = None):
        """
        Add a layer by creating a PIL image from primitive template with transformations applied.
        
        Args:
            primitive_template: (H, W) primitive template
            x, y: Position
            r: Scale
            theta: Rotation
            v: Visibility logit
            c: (3,) RGB color logits
            name: Layer name
        """
        layer_name = name or f"primitive_{len(self.psd)}"
        
        # Apply geometric transformations to template
        transformed_template = self._apply_transformations(primitive_template, x, y, r, theta)
        
        # Convert transformed template to PIL Image with color and alpha
        pil_image = self._template_to_pil(transformed_template, c, v)
        
        # Create PixelLayer from PIL image
        layer = PixelLayer.frompil(pil_image, self.psd, layer_name)
        
        # Add layer to PSD
        self.psd.append(layer)
        
    def _apply_transformations(self, template: torch.Tensor, x: float, y: float, r: float, theta: float) -> torch.Tensor:
        """Apply geometric transformations using exactly the same logic as _batched_soft_rasterize."""
        import torch.nn.functional as F
        
        # Ensure template is on the correct device
        template = template.to(self.device)
        
        # Create pixel coordinate grids exactly like VectorRenderer._create_coordinate_grid
        H, W = self.canvas_height, self.canvas_width
        X, Y = torch.meshgrid(
            torch.arange(W, device=self.device, dtype=torch.float32),
            torch.arange(H, device=self.device, dtype=torch.float32),
            indexing='xy'
        )
        X = X.unsqueeze(0)  # (1, H, W)
        Y = Y.unsqueeze(0)  # (1, H, W)
        
        # Convert single primitive parameters to batch format (B=1)
        x_tensor = torch.tensor([x], device=self.device, dtype=torch.float32)
        y_tensor = torch.tensor([y], device=self.device, dtype=torch.float32)
        r_tensor = torch.tensor([r], device=self.device, dtype=torch.float32)
        theta_tensor = torch.tensor([theta], device=self.device, dtype=torch.float32)
        
        # Expand coordinates and parameters exactly like _batched_soft_rasterize
        B = 1
        X_exp = X.expand(B, H, W)
        Y_exp = Y.expand(B, H, W)
        x_exp = x_tensor.view(B, 1, 1).expand(B, H, W)
        y_exp = y_tensor.view(B, 1, 1).expand(B, H, W)
        r_exp = r_tensor.view(B, 1, 1).expand(B, H, W)
        
        # Normalize and rotate positions - EXACT same logic as _batched_soft_rasterize
        pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
        cos_t = torch.cos(theta_tensor)
        sin_t = torch.sin(theta_tensor)
        R_inv = torch.zeros(B, 2, 2, device=self.device)
        R_inv[:, 0, 0] = cos_t; R_inv[:, 0, 1] = sin_t
        R_inv[:, 1, 0] = -sin_t; R_inv[:, 1, 1] = cos_t
        uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
        grid = uv.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        # Prepare template for grid_sample - single primitive template
        bmp_exp = template.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
        
        # Sample masks via grid_sample - EXACT same as _batched_soft_rasterize
        sampled = F.grid_sample(bmp_exp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # Return single-channel mask
        return sampled.squeeze(1).squeeze(0)  # (H, W)
        
    def _template_to_pil(self, template: torch.Tensor, c: torch.Tensor, v: float) -> Image.Image:
        """Convert primitive template to PIL RGBA image with color and alpha."""
        # Convert to numpy
        template_np = template.detach().cpu().numpy()
        h, w = template_np.shape
        
        # Apply color (sigmoid activation)
        rgb = torch.sigmoid(c).detach().cpu().numpy()
        visibility = self.alpha_upper_bound * (1 / (1 + np.exp(-v))).item()
        
        # Create RGBA array
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Set RGB channels with template as mask
        for i in range(3):
            rgba[:, :, i] = ((template_np>0).astype(np.float32) * rgb[i] * 255).astype(np.uint8)
            
        # Set alpha channel (template acts as alpha mask, modulated by visibility)
        rgba[:, :, 3] = (template_np * visibility * 255).astype(np.uint8)
        
        # Convert to PIL Image
        return Image.fromarray(rgba, 'RGBA')
        
    def export_psd(self, filepath: str):
        """Export as PSD file."""
        # Ensure .psd extension
        if not filepath.endswith('.psd'):
            filepath = filepath.replace('.tiff', '.psd').replace('.png', '.psd')
            
        self.psd.save(filepath)
        print(f"✅ Exported PSD with {len(self.psd)} layers to {filepath}")
        
    def export_individual_layers(self, output_dir: str):
        """Export each layer as individual PNG files for debugging."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, layer in enumerate(self.psd):
            if hasattr(layer, 'topil'):
                layer_path = os.path.join(output_dir, f"{layer.name}.png")
                try:
                    layer_img = layer.topil()
                    if layer_img:
                        layer_img.save(layer_path)
                except Exception as e:
                    print(f"Warning: Could not export layer {layer.name}: {e}")
                    
        print(f"✅ Exported individual layers to {output_dir}/")

import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
import time
import os
import sys

class PSDExporter:
    """Exports individual transformed primitives as PSD layers using psd-tools."""
    
    def __init__(self, canvas_width: int, canvas_height: int,
                 alpha_upper_bound: float = 1.0, scale_factor: float = 1.0, use_cuda: bool = True,
                 c_blend: float = 0.0, primitive_colors: Optional[torch.Tensor] = None):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.alpha_upper_bound = alpha_upper_bound
        self.scale_factor = scale_factor
        self.use_cuda = use_cuda
        self.c_blend = c_blend
        self.primitive_colors = primitive_colors  # c_o colors for blending
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate scaled dimensions for high-resolution export
        self.export_width = int(canvas_width * scale_factor)
        self.export_height = int(canvas_height * scale_factor)
        
        # Pre-compute coordinate grids for performance optimization
        self._precompute_coordinate_grids()
        
        # Create new PSD document with scaled dimensions
        self.psd = PSDImage.new('RGBA', (self.export_width, self.export_height), color=(0, 0, 0, 0))

        
    def add_layers_batch_optimized(self, primitive_templates: torch.Tensor,
                                  x_tensor: torch.Tensor, y_tensor: torch.Tensor, r_tensor: torch.Tensor,
                                  theta_tensor: torch.Tensor, v_tensor: torch.Tensor, c_tensor: torch.Tensor,
                                  names=None, reverse_order: bool = True):
        """
        Add multiple layers using batched F.grid_sample for maximum efficiency.
        Handles all data preparation internally for cleaner main.py code.
        
        Args:
            primitive_templates: (p, H, W) or (H, W) primitive templates from renderer.S
            x_tensor, y_tensor, r_tensor, theta_tensor: (N,) primitive parameter tensors
            v_tensor: (N,) visibility logit tensor
            c_tensor: (N, 3) RGB color logit tensor
            reverse_order: Whether to reverse layer order for correct layering (default: True)
        """
        N = len(x_tensor)
        if names is None:
            names = [f"primitive_{i}" for i in range(N)]
        
        # Handle primitive templates preparation
        start_total = time.time()
        print(f"🎨 Starting PSD export for {N} primitives...")
        
        try:
            # Try CUDA implementation first - add path for import
            cuda_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cuda_tile_rasterizer')
            if cuda_dir not in sys.path:
                sys.path.insert(0, cuda_dir)
            
            from cuda_tile_rasterizer.psd_export_uint8 import export_psd_layers_cuda_uint8, create_pixel_layers_from_numpy
            
            # Generate global_bmp_sel (template selection indices)
            if primitive_templates.dim() == 2:
                # Single template for all primitives
                global_bmp_sel = torch.zeros(N, dtype=torch.int32, device=x_tensor.device)
                primitive_templates = primitive_templates.unsqueeze(0)
            else:
                # Multiple templates - cycle through them
                p = primitive_templates.shape[0]
                global_bmp_sel = (-torch.arange(N, device=x_tensor.device) + N - 1) % p
                global_bmp_sel = global_bmp_sel.int()
            
            # Apply sigmoid to tensors
            v_tensor = torch.sigmoid(v_tensor)
            c_tensor = torch.sigmoid(c_tensor)
            
            # Export layers using CUDA with c_blend support
            numpy_arrays, bounds_list, valid_indices = export_psd_layers_cuda_uint8(
                primitive_templates, x_tensor, y_tensor, r_tensor, theta_tensor,
                v_tensor, c_tensor, global_bmp_sel,
                self.export_height, self.export_width, self.scale_factor, self.alpha_upper_bound,
                self.c_blend, self.primitive_colors
            )
            numpy_arrays.reverse()
            bounds_list.reverse()
            valid_indices.reverse()
            
            # Create PixelLayer objects and add to PSD
            reversed_names = names[::-1]
            layers = create_pixel_layers_from_numpy(
                numpy_arrays, bounds_list, valid_indices, reversed_names, self.psd
            )
            self.psd.extend(layers)
            
            total_time = time.time() - start_total
            print(f"✅ PSD generation {len(layers)}/{N} layers: wating for saving...")
            
        except (ImportError, RuntimeError) as e:
            # Fallback to PyTorch implementation
            print(f"⚠️  CUDA unavailable, using PyTorch fallback...")
            
            # Apply transformations in batch
            transformed_templates = self._apply_transformations_batch(
                primitive_templates, x_tensor, y_tensor, r_tensor, theta_tensor
            )
            
            # Generate global_bmp_sel
            if primitive_templates.dim() == 2:
                primitive_templates = primitive_templates.unsqueeze(0)
            P = primitive_templates.shape[0]
            if P == 1:
                template_indices = np.zeros(N, dtype=np.int32)
            else:
                template_indices = ((-np.arange(N) + N - 1) % P).astype(np.int32)
            
            # Create layers in batch
            layers = []
            for i in range(N):
                template = transformed_templates[i]
                c_i = torch.sigmoid(c_tensor[i]).detach().cpu().numpy()  # (3,)
                vis = torch.sigmoid(v_tensor[i]).item() * self.alpha_upper_bound
                
                # Apply color blending per-pixel if c_blend > 0
                if self.primitive_colors is not None and self.c_blend > 0.0:
                    template_idx = template_indices[i]
                    c_o_map = self.primitive_colors[template_idx].detach().cpu().numpy()  # (H_t, W_t, 3)
                    
                    # Sample c_o for each pixel
                    H_export, W_export = template.shape
                    x_scaled = x_tensor[i].item() * self.scale_factor
                    y_scaled = y_tensor[i].item() * self.scale_factor
                    r_scaled = r_tensor[i].item() * self.scale_factor
                    theta_val = theta_tensor[i].item()
                    
                    rgb_map = self._sample_color_map_numpy(
                        c_o_map, x_scaled, y_scaled, r_scaled, theta_val, H_export, W_export
                    )  # (H, W, 3)
                    
                    # Blend c_i and c_o per-pixel
                    rgb_blended = (1.0 - self.c_blend) * c_i[None, None, :] + self.c_blend * rgb_map
                    
                    # Create RGBA image with per-pixel colors
                    rgba = self._create_rgba_image_with_color_map(template, rgb_blended, vis)
                else:
                    # No color blending - use c_i
                    rgb = torch.tensor(c_i)
                    rgba = self._create_rgba_image(template, rgb, vis)
                
                # Convert to PIL and create layer
                pil_image = Image.fromarray(rgba, 'RGBA')
                layer = PixelLayer.frompil(pil_image, self.psd, names[i])
                layers.append(layer)
            
            # Add all layers to PSD in batch
            self.psd.extend(layers)
            
            total_time = time.time() - start_total
            print(f"✅ PSD export completed: {len(layers)}/{N} layers in {total_time:.3f}s")

        
    def _apply_transformations_batch(self, primitive_templates: torch.Tensor,
                                   x_tensor: torch.Tensor, y_tensor: torch.Tensor, r_tensor: torch.Tensor, 
                                   theta_tensor: torch.Tensor) -> torch.Tensor:
        """Apply geometric transformations to multiple primitives at once using batched F.grid_sample."""
        import torch.nn.functional as F
        
        N = len(x_tensor)
        H, W = self.export_height, self.export_width
        
        # Handle template selection
        if primitive_templates.dim() == 2:
            # Single template for all primitives
            templates = primitive_templates.unsqueeze(0).expand(N, -1, -1)
        else:
            # Multiple templates - cycle through them
            p = primitive_templates.shape[0]
            template_indices = torch.arange(N, device=self.device) % p
            templates = primitive_templates[template_indices]
        
        # Apply scaling to parameters (tensors already on correct device)
        x_tensors = x_tensor * self.scale_factor
        y_tensors = y_tensor * self.scale_factor
        r_tensors = r_tensor * self.scale_factor
        theta_tensors = theta_tensor  # rotation unchanged
        
        # Expand parameters for batch processing (N, H, W)
        x_exp = x_tensors.view(N, 1, 1).expand(N, H, W)
        y_exp = y_tensors.view(N, 1, 1).expand(N, H, W)
        r_exp = r_tensors.view(N, 1, 1).expand(N, H, W)
        
        # Expand coordinate grids for batch (N, H, W)
        X_batch = self.X_exp.expand(N, H, W)
        Y_batch = self.Y_exp.expand(N, H, W)
        
        # Normalize and rotate positions for all primitives
        pos = torch.stack([X_batch - x_exp, Y_batch - y_exp], dim=1) / r_exp.unsqueeze(1)
        
        # Batch rotation matrices (N, 2, 2)
        cos_t = torch.cos(theta_tensors)
        sin_t = torch.sin(theta_tensors)
        R_inv = torch.zeros(N, 2, 2, device=self.device)
        R_inv[:, 0, 0] = cos_t; R_inv[:, 0, 1] = sin_t
        R_inv[:, 1, 0] = -sin_t; R_inv[:, 1, 1] = cos_t
        
        uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
        grid = uv.permute(0, 2, 3, 1)  # (N, H, W, 2)
        
        # Prepare templates for batched grid_sample (N, 1, template_H, template_W)
        templates_4d = templates.unsqueeze(1)
        
        # Ensure both tensors have the same dtype for grid_sample
        if templates_4d.dtype != grid.dtype:
            if templates_4d.dtype == torch.float16:
                grid = grid.half()
            else:
                templates_4d = templates_4d.float()
        
        # Batched grid sample - this is the key optimization!
        sampled = F.grid_sample(templates_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # Return (N, H, W)
        return sampled.squeeze(1)
        
    def _batch_templates_to_pil_with_bounds(self, templates: torch.Tensor, c_tensor: torch.Tensor, 
                                          v_tensor: torch.Tensor, x_tensor: torch.Tensor, y_tensor: torch.Tensor,
                                          r_tensor: torch.Tensor, theta_tensor: torch.Tensor,
                                          primitive_templates: torch.Tensor) -> tuple[List[Image.Image], List[tuple[int, int, int, int]]]:
        """
        Batch convert multiple templates to PIL images with bounds using vectorized operations.
        
        Args:
            templates: (N, H, W) transformed templates
            c_tensor: (N, 3) RGB color tensor
            v_tensor: (N,) visibility logit tensor
            x_tensor, y_tensor, r_tensor, theta_tensor: primitive parameters for color sampling
            primitive_templates: original templates for template selection
            
        Returns:
            List of PIL Images and list of bounds tuples
        """
        N, H, W = templates.shape
        
        # Convert all templates to numpy at once
        templates_np = templates.detach().cpu().numpy()  # (N, H, W)
        
        # Convert colors and visibility to numpy (already batched)
        c_i = torch.sigmoid(c_tensor).detach().cpu().numpy()  # (N, 3) - c_i colors
        v_array = v_tensor.detach().cpu().numpy()  # (N,)
        visibility_batch = self.alpha_upper_bound * (1 / (1 + np.exp(-v_array)))  # (N,)
        
        # Create batch RGBA array (N, H, W, 4)
        rgba_batch = np.zeros((N, H, W, 4), dtype=np.uint8)
        
        # Apply color blending per-pixel (matching PNG rendering)
        if self.primitive_colors is not None and self.c_blend > 0.0:
            # Get primitive parameters for color sampling
            x_np = (x_tensor * self.scale_factor).detach().cpu().numpy()
            y_np = (y_tensor * self.scale_factor).detach().cpu().numpy()
            r_np = (r_tensor * self.scale_factor).detach().cpu().numpy()
            theta_np = theta_tensor.detach().cpu().numpy()
            
            # Generate global_bmp_sel
            P = primitive_templates.shape[0] if primitive_templates.dim() > 2 else 1
            if P == 1:
                template_indices = np.zeros(N, dtype=np.int32)
            else:
                template_indices = ((-np.arange(N) + N - 1) % P).astype(np.int32)
            
            # Process each primitive
            for i in range(N):
                template_idx = template_indices[i]
                c_o_map = self.primitive_colors[template_idx].detach().cpu().numpy()  # (H_t, W_t, 3)
                
                # Sample c_o for each pixel using same logic as PNG rendering
                rgb_map = self._sample_color_map_numpy(
                    c_o_map, x_np[i], y_np[i], r_np[i], theta_np[i], H, W
                )  # (H, W, 3)
                
                # Blend c_i and c_o per-pixel
                c_i_expanded = c_i[i][None, None, :]  # (1, 1, 3)
                rgb_blended = (1.0 - self.c_blend) * c_i_expanded + self.c_blend * rgb_map
                
                # Apply to RGBA batch
                template_mask = templates_np[i] > 0  # (H, W)
                for c in range(3):
                    rgba_batch[i, :, :, c] = (template_mask * rgb_blended[:, :, c] * 255).astype(np.uint8)
        else:
            # No color blending - use c_i for all pixels
            template_mask = templates_np > 0  # (N, H, W)
            for i in range(3):
                rgba_batch[:, :, :, i] = (template_mask * c_i[:, i:i+1, None] * 255).astype(np.uint8)
        
        # Vectorized alpha channel assignment
        rgba_batch[:, :, :, 3] = (templates_np * visibility_batch[:, None, None] * 255).astype(np.uint8)
        
        # Batch bounding box calculation
        pil_images = []
        bounds_list = []
        
        for i in range(N):
            rgba = rgba_batch[i]
            alpha_mask = rgba[:, :, 3] > 0
            
            if not alpha_mask.any():
                # Completely transparent
                pil_images.append(Image.fromarray(np.zeros((1, 1, 4), dtype=np.uint8), 'RGBA'))
                bounds_list.append((0, 0, 1, 1))
                continue
            
            # Find bounds
            rows = np.any(alpha_mask, axis=1)
            cols = np.any(alpha_mask, axis=0)
            top, bottom = np.where(rows)[0][[0, -1]]
            left, right = np.where(cols)[0][[0, -1]]
            
            # Add 1 to bottom and right for inclusive bounds
            bottom += 1
            right += 1
            
            # Crop and create PIL image
            cropped_rgba = rgba[top:bottom, left:right]
            pil_images.append(Image.fromarray(cropped_rgba, 'RGBA'))
            bounds_list.append((left, top, right, bottom))
        
        return pil_images, bounds_list
        
    def _create_rgba_image(self, template: torch.Tensor, rgb: torch.Tensor, vis: torch.Tensor) -> np.ndarray:
        """
        Create RGBA image from template, RGB color, and visibility.
        
        Args:
            template: (H, W) transformed template
            rgb: (3,) RGB color tensor
            vis: scalar visibility value
            
        Returns:
            RGBA numpy array (H, W, 4) with dtype uint8
        """
        H, W = template.shape
        
        # Convert to numpy
        template_np = template.detach().cpu().numpy()
        rgb_np = rgb.detach().cpu().numpy()
        vis_np = vis.detach().cpu().numpy()
        
        # Create RGBA array
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        
        # Set RGB channels where template is non-zero
        template_mask = template_np > 0
        for i in range(3):
            rgba[:, :, i] = (template_mask * rgb_np[i] * 255).astype(np.uint8)
        
        # Set alpha channel
        rgba[:, :, 3] = (template_np * vis_np * 255).astype(np.uint8)
        
        return rgba
    
    def _create_rgba_image_with_color_map(self, template: torch.Tensor, rgb_map: np.ndarray, vis: float) -> np.ndarray:
        """Create RGBA image from template and per-pixel color map"""
        H, W = template.shape
        template_np = template.detach().cpu().numpy()
        vis_np = vis
        
        # Create RGBA array
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        
        # Set RGB channels where template is non-zero (using per-pixel colors)
        template_mask = template_np > 0
        for i in range(3):
            rgba[:, :, i] = (template_mask * rgb_map[:, :, i] * 255).astype(np.uint8)
        
        # Set alpha channel
        rgba[:, :, 3] = (template_np * vis_np * 255).astype(np.uint8)
        
        return rgba
        
    def _precompute_coordinate_grids(self):
        """Pre-compute coordinate grids and expansions to avoid redundant calculations."""
        H, W = self.export_height, self.export_width
        
        # Create pixel coordinate grids
        X, Y = torch.meshgrid(
            torch.arange(W, device=self.device, dtype=torch.float32),
            torch.arange(H, device=self.device, dtype=torch.float32),
            indexing='xy'
        )
        
        # Store base grids
        self.X = X.unsqueeze(0)  # (1, H, W)
        self.Y = Y.unsqueeze(0)  # (1, H, W)
        
        # Pre-compute expanded grids for B=1 (since we always process one primitive at a time)
        self.X_exp = self.X.expand(1, H, W)
        self.Y_exp = self.Y.expand(1, H, W)
    
    def _sample_color_map_numpy(self, color_map: np.ndarray, center_x: float, center_y: float,
                                radius: float, rotation: float, H: int, W: int) -> np.ndarray:
        """
        Sample color from color map for each pixel (matching PNG rendering).
        
        Args:
            color_map: (H_t, W_t, 3) color map
            center_x, center_y: primitive center position
            radius: primitive radius
            rotation: primitive rotation
            H, W: output dimensions
            
        Returns:
            (H, W, 3) sampled colors
        """
        H_t, W_t = color_map.shape[:2]
        
        # Create pixel coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # Relative coordinates to primitive center
        rel_x = x_coords - center_x
        rel_y = y_coords - center_y
        
        # Normalize by radius
        r_inv = 1.0 / radius if radius > 1e-6 else 1e6
        norm_x = rel_x * r_inv
        norm_y = rel_y * r_inv
        
        # Apply rotation
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)
        rot_x = cos_theta * norm_x + sin_theta * norm_y
        rot_y = -sin_theta * norm_x + cos_theta * norm_y
        
        # Convert to color map coordinates [0, W_t-1], [0, H_t-1]
        coord_x = (rot_x + 1.0) * 0.5 * (W_t - 1)
        coord_y = (rot_y + 1.0) * 0.5 * (H_t - 1)
        
        # Clamp coordinates
        coord_x = np.clip(coord_x, 0, W_t - 1)
        coord_y = np.clip(coord_y, 0, H_t - 1)
        
        # Bilinear interpolation
        x0 = np.floor(coord_x).astype(np.int32)
        y0 = np.floor(coord_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, W_t - 1)
        y1 = np.minimum(y0 + 1, H_t - 1)
        
        fx = coord_x - x0
        fy = coord_y - y0
        
        # Sample colors at four corners
        c00 = color_map[y0, x0]
        c01 = color_map[y1, x0]
        c10 = color_map[y0, x1]
        c11 = color_map[y1, x1]
        
        # Bilinear interpolation
        c0 = c00 * (1 - fx)[:, :, None] + c10 * fx[:, :, None]
        c1 = c01 * (1 - fx)[:, :, None] + c11 * fx[:, :, None]
        sampled_colors = c0 * (1 - fy)[:, :, None] + c1 * fy[:, :, None]
        
        return sampled_colors  # (H, W, 3)
    
    def export_psd(self, filepath: str):
        """Export as PSD file."""
        # Ensure .psd extension
        if not filepath.endswith('.psd'):
            filepath = filepath.replace('.tiff', '.psd').replace('.png', '.psd')
            
        self.psd.save(filepath)
        print(f"💾 Exported PSD with {len(self.psd)} layers to {filepath}")

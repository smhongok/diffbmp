import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
import time

class PSDExporter:
    """Exports individual transformed primitives as PSD layers using psd-tools."""
    
    def __init__(self, canvas_width: int, canvas_height: int,
                 alpha_upper_bound: float = 1.0, scale_factor: float = 1.0):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.alpha_upper_bound = alpha_upper_bound
        self.scale_factor = scale_factor
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
        
        try:
            # Try CUDA implementation first - add path for import
            import sys
            import os
            cuda_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cuda_tile_rasterizer')
            if cuda_dir not in sys.path:
                sys.path.insert(0, cuda_dir)
            
            from psd_export_uint8 import export_psd_layers_cuda_uint8
            
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
            
            # Single CUDA kernel call to generate PIL images and bounding boxes directly
            start_cuda = time.time()
            v_tensor = torch.sigmoid(v_tensor)
            c_tensor = torch.sigmoid(c_tensor)
            pil_images, bounds_list = export_psd_layers_cuda_uint8(
                primitive_templates, x_tensor, y_tensor, r_tensor, theta_tensor,
                v_tensor, c_tensor, global_bmp_sel,
                self.export_height, self.export_width, self.scale_factor, self.alpha_upper_bound
            )
            pil_images.reverse()
            bounds_list.reverse()
            cuda_time = time.time() - start_cuda
            print(f"   🚀 CUDA processing: {cuda_time:.3f}s ({cuda_time/N*1000:.2f}ms per primitive)")
            
            # Add layers to PSD with optimized batching
            start_layers = time.time()
            
            # Filter out empty layers to reduce processing
            valid_layers = []
            for i in range(N):
                left, top, right, bottom = bounds_list[i]
                img = pil_images[i]
                if img.size[0] > 1 and img.size[1] > 1:  # Skip 1x1 empty layers
                    # Check if layer has any non-transparent pixels
                    img_array = np.array(img)
                    if img_array[:,:,3].max() > 0:  # Has some alpha
                        valid_layers.append((i, img, left, top, right, bottom))
            
            print(f"   📊 Processing {len(valid_layers)}/{N} non-empty layers...")
            
            # Process valid layers only
            for i, img, left, top, right, bottom in valid_layers:
                layer = PixelLayer.frompil(img, self.psd, names[i], top=top, left=left)
                self.psd.append(layer)
            
            layers_time = time.time() - start_layers
            print(f"   🔧 Layer creation: {layers_time:.3f}s ({layers_time/len(valid_layers)*1000:.2f}ms per valid layer)")
            
            total_time = time.time() - start_total
            print(f"✅ CUDA Total time: {total_time:.3f}s")
            print(f"   🔄 Breakdown: CUDA={cuda_time/total_time*100:.1f}%, Layer={layers_time/total_time*100:.1f}%")
            
        except (ImportError, RuntimeError) as e:
            # Fallback to PyTorch implementation
            print(f"⚠️  CUDA implementation failed: {e}")
            print("   🔄 Falling back to PyTorch implementation...")
            
            # Apply transformations in batch
            start_batch = time.time()
            transformed_templates = self._apply_transformations_batch(
                primitive_templates, x_tensor, y_tensor, r_tensor, theta_tensor
            )
            batch_time = time.time() - start_batch
            print(f"   ⏱️  Batch transformations: {batch_time:.3f}s ({batch_time/N*1000:.2f}ms per primitive)")
            
            # Convert to PIL images and add to PSD
            start_pil = time.time()
            
            # Batch convert all templates to numpy
            start_numpy = time.time()
            pil_images, bounds_list = self._batch_templates_to_pil_with_bounds(
                transformed_templates, c_tensor, v_tensor
            )
            numpy_time = time.time() - start_numpy
            print(f"      🔧 Batch numpy conversion: {numpy_time:.3f}s ({numpy_time/N*1000:.2f}ms per primitive)")
            
            # Create layers with same optimization as CUDA path
            start_layers = time.time()
            
            # Filter out empty layers to reduce processing
            valid_layers = []
            for i in range(N):
                left, top, right, bottom = bounds_list[i]
                img = pil_images[i]
                if img.size[0] > 1 and img.size[1] > 1:  # Skip 1x1 empty layers
                    # Check if layer has any non-transparent pixels
                    img_array = np.array(img)
                    if img_array[:,:,3].max() > 0:  # Has some alpha
                        valid_layers.append((i, img, left, top, right, bottom))
            
            print(f"      📊 Processing {len(valid_layers)}/{N} non-empty layers...")
            
            # Process valid layers only
            for i, img, left, top, right, bottom in valid_layers:
                layer = PixelLayer.frompil(img, self.psd, names[i], top=top, left=left)
                self.psd.append(layer)
            
            layers_time = time.time() - start_layers
            print(f"      🔧 Layer creation: {layers_time:.3f}s ({layers_time/len(valid_layers)*1000:.2f}ms per valid layer)")
            
            pil_time = time.time() - start_pil
            print(f"   ⏱️  Total PIL + layer creation: {pil_time:.3f}s ({pil_time/N*1000:.2f}ms per primitive)")
            
            total_time = time.time() - start_total
            print(f"✅ PyTorch Total time: {total_time:.3f}s")
            print(f"   🔄 Breakdown: Batch={batch_time/total_time*100:.1f}%, PIL+Layer={pil_time/total_time*100:.1f}%")

        
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
                                          v_tensor: torch.Tensor) -> tuple[List[Image.Image], List[tuple[int, int, int, int]]]:
        """
        Batch convert multiple templates to PIL images with bounds using vectorized operations.
        
        Args:
            templates: (N, H, W) transformed templates
            c_tensor: (N, 3) RGB color tensor
            v_tensor: (N,) visibility logit tensor
            
        Returns:
            List of PIL Images and list of bounds tuples
        """
        N, H, W = templates.shape
        
        # Convert all templates to numpy at once
        templates_np = templates.detach().cpu().numpy()  # (N, H, W)
        
        # Convert colors and visibility to numpy (already batched)
        rgb_batch = torch.sigmoid(c_tensor).detach().cpu().numpy()  # (N, 3)
        v_array = v_tensor.detach().cpu().numpy()  # (N,)
        visibility_batch = self.alpha_upper_bound * (1 / (1 + np.exp(-v_array)))  # (N,)
        
        # Create batch RGBA array (N, H, W, 4)
        rgba_batch = np.zeros((N, H, W, 4), dtype=np.uint8)
        
        # Vectorized RGB channel assignment
        template_mask = templates_np > 0  # (N, H, W)
        for i in range(3):
            rgba_batch[:, :, :, i] = (template_mask * rgb_batch[:, i:i+1, None] * 255).astype(np.uint8)
        
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
        
        
    
        
    def export_psd(self, filepath: str):
        """Export as PSD file."""
        # Ensure .psd extension
        if not filepath.endswith('.psd'):
            filepath = filepath.replace('.tiff', '.psd').replace('.png', '.psd')
            
        self.psd.save(filepath)
        print(f"✅ Exported PSD with {len(self.psd)} layers to {filepath}")
        

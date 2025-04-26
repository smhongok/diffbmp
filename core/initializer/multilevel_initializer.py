import torch
import gc
from typing import Tuple, List
from core.initializer.base_initializer import BaseInitializer
from core.renderer.vector_renderer import VectorRenderer
import numpy as np

class MultiLevelInitializer(BaseInitializer):
    """
    A multi-level initializer that applies SVG splatting at multiple levels:
      - Level 0: Divide into 4 overlapping quadrants, initialize & render each, then blend.
      - Levels 1..N-1: On the residual (target – current reconstruction), initialize & render full image.
    """
    def __init__(self,
                 num_init=100, alpha=0.3, min_distance=20,
                 peak_threshold=0.5, radii_min=2, radii_max=None,
                 v_init_bias=-5.0, v_init_slope=0.0,
                 keypoint_extracting=False, debug_mode=False):
        super().__init__(num_init, alpha, min_distance,
                          peak_threshold, radii_min,
                          radii_max, v_init_bias, v_init_slope,
                          keypoint_extracting, debug_mode)
        self.overlap_pixels = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.num_levels = 4
        #self.level_fracs = [0.2, 0.2, 0.2, 0.2]  
        #self.per_quad_frac = self.level_fracs[0] / 4
        self.num_levels = 2
        self.level_fracs = [0.4, 0.4]  
        self.per_quad_frac = self.level_fracs[0] / 2
        self.residual_frac = 0.2

    def _divide_into_quadrants(self, image: torch.Tensor):
        H, W, C = image.shape
        h2, w2 = H // 2, W // 2
        o = self.overlap_pixels
        coords = [
            (0,      h2+o, 0,      w2+o),
            (0,      h2+o, w2-o,   W),
            (h2-o,   H,    0,      w2+o),
            (h2-o,   H,    w2-o,   W),
        ]
        quads = []
        for ys, ye, xs, xe in coords:
            quads.append((image[ys:ye, xs:xe].clone(), (ys, ye, xs, xe)))
        return quads

    def _merge_quadrants(self,
                          rendered_quads: List[Tuple[torch.Tensor, Tuple[int,int,int,int]]],
                          original_shape: Tuple[int,int,int]) -> torch.Tensor:
        H, W, C = original_shape
        merged = torch.zeros((H, W, C), device=self.device)
        weight = torch.zeros((H, W), device=self.device)

        for img_q, (ys, ye, xs, xe) in rendered_quads:
            # Get actual dimensions of the quadrant image
            hq, wq, _ = img_q.shape
            
            # Debug prints to identify the issue
            print(f"Quadrant shape: {img_q.shape}, Region: ({ys}:{ye}, {xs}:{xe})")
            print(f"Expected region size: ({ye-ys}, {xe-xs})")
            
            # Create weight mask with the exact same dimensions as the quadrant image
            wm = torch.ones((hq, wq), device=self.device)
            o = self.overlap_pixels
            
            # feather edges
            if ys>0:
                for i in range(o): wm[i,:]   *= (i/o)
            if ye< H:
                for i in range(o): wm[-1-i,:]*= (i/o)
            if xs>0:
                for i in range(o): wm[:,i]   *= (i/o)
            if xe< W:
                for i in range(o): wm[:,-1-i]*= (i/o)

            # Expand weight mask to match image channels
            wm = wm.unsqueeze(-1).expand(-1,-1,C)
            
            # Check if dimensions match
            if wm.shape[0] != img_q.shape[0] or wm.shape[1] != img_q.shape[1]:
                print(f"Dimension mismatch: wm={wm.shape}, img_q={img_q.shape}")
                # Resize weight mask to match image dimensions if needed
                wm = wm[:img_q.shape[0], :img_q.shape[1], :]
            
            # Add weighted quadrant to merged image
            merged[ys:ye, xs:xe] += img_q * wm
            weight[ys:ye, xs:xe] += wm[:,:,0]

        # Normalize by weights
        weight = weight.clamp(min=1e-8).unsqueeze(-1).expand(-1,-1,C)
        return merged / weight
    
    def _render_image(self, 
                     renderer: VectorRenderer, 
                     params: Tuple[torch.Tensor, ...], 
                     bmp_image: torch.Tensor,
                     target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Render an image using the given parameters.
        
        Args:
            renderer: Vector renderer
            params: Rendering parameters
            bmp_image: Base bitmap image
            target_size: Optional target size for the rendered image (H, W)
            
        Returns:
            Rendered image tensor
        """
        x, y, r, v, theta, c = params
        
        # Generate masks
        cached_masks = renderer._batched_soft_rasterize(
            bmp_image, x, y, r, theta,
            sigma=0.0
        )
        
        # Render image
        rendered = renderer.render(cached_masks, v, c)
        
        # Resize if target size is provided and different from current size
        if target_size is not None:
            H, W = target_size
            if rendered.shape[0] != H or rendered.shape[1] != W:
                # Use interpolation to resize the rendered image
                rendered = torch.nn.functional.interpolate(
                    rendered.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)
        
        # Clear memory
        del cached_masks
        torch.cuda.empty_cache()
        
        return rendered
    
    def _random_initialize(self, image: torch.Tensor, num_init=None) -> Tuple[torch.Tensor, ...]:
        """
        Initialize parameters using random initialization.
        
        Args:
            image: Target image to match
            num_init: Number of circles to initialize (if None, use self.num_init)
            
        Returns:
            Tuple of initialized parameters (x, y, r, v, theta, c)
        """
        if num_init is None:
            num_init = self.num_init
            
        H, W, C = image.shape
        
        # Convert image to grayscale if it's color
        if C > 1:
            gray_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray_image = image[:, :, 0]
        
        # Normalize to [0, 1]
        gray_image = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min() + 1e-8)
        
        # Generate random positions
        x = torch.rand(num_init, device=self.device) * W
        y = torch.rand(num_init, device=self.device) * H
        
        # Generate random radii
        if self.radii_max is None:
            self.radii_max = min(H, W) // 4
        
        r = torch.rand(num_init, device=self.device) * (self.radii_max - self.radii_min) + self.radii_min
        
        # Generate random angles
        theta = torch.rand(num_init, device=self.device) * 2 * np.pi
        
        # Initialize v based on image intensity at the positions
        v = torch.zeros(num_init, device=self.device)
        for i in range(num_init):
            x_idx = min(int(x[i]), W-1)
            y_idx = min(int(y[i]), H-1)
            v[i] = self.v_init_bias + self.v_init_slope * gray_image[y_idx, x_idx]
        
        # Initialize c based on image color at the positions
        c = torch.zeros(num_init, C, device=self.device)
        for i in range(num_init):
            x_idx = min(int(x[i]), W-1)
            y_idx = min(int(y[i]), H-1)
            c[i] = image[y_idx, x_idx]
        
        return x, y, r, v, theta, c

    def initialize(self, target_image: torch.Tensor) -> Tuple[torch.Tensor,...]:
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        H, W, C = target_image.shape
        renderer = VectorRenderer((H, W), device=self.device)
        bmp_whole = torch.ones((H, W), device=self.device)

        # Accumulate all levels' parameters here
        all_x, all_y, all_r = [], [], []
        all_v, all_theta, all_c = [], [], []

        # CURRENT reconstruction starts at zero
        current_recon = torch.zeros_like(target_image)

        # --- Level 0: quadrants ---
        quads = self._divide_into_quadrants(target_image)
        rendered_quads = []
        per_quad = max(1, int(self.num_init * self.per_quad_frac))

        for quad_img, (ys, ye, xs, xe) in quads:
            # Get the expected size for this quadrant
            expected_h, expected_w = ye - ys, xe - xs
            
            # initialize this quadrant
            x, y, r, v, theta, c = self._random_initialize(quad_img, num_init=per_quad)
            # shift into full-image coords
            x = x + xs; y = y + ys

            # render this quadrant (local bmp)
            bmp_q = torch.ones((quad_img.shape[0], quad_img.shape[1]), device=self.device)
            rendered_q = self._render_image(
                renderer, 
                (x, y, r, v, theta, c), 
                bmp_q,
                target_size=(expected_h, expected_w)
            )

            rendered_quads.append((rendered_q, (ys, ye, xs, xe)))

            # collect params
            all_x.append(x); all_y.append(y); all_r.append(r); all_v.append(v); all_theta.append(theta); all_c.append(c)

        # blend quadrants back into full image
        merged = self._merge_quadrants(rendered_quads, (H, W, C))
        current_recon = merged

        # stop if residual is negligible
        residual = (target_image - current_recon).clamp(0.0,1.0)
        if residual.norm() >= self.residual_frac:
            # primitives for this level: decay by quadrant_scale_factor each level
            nr = max(1, int(self.num_init * self.residual_frac))
            x, y, r, v, theta, c = self._random_initialize(residual, num_init=nr)

            # render full-resolution residual primitives
            rendered_res = self._render_image(renderer, (x, y, r, v, theta, c), bmp_whole)
            current_recon = (current_recon + rendered_res).clamp(0.0, 1.0)

            # collect these level's params
            all_x.append(x); all_y.append(y); all_r.append(r); all_v.append(v); all_theta.append(theta); all_c.append(c)

        # concatenate all
        x_final     = torch.cat(all_x,     dim=0)
        y_final     = torch.cat(all_y,     dim=0)
        r_final     = torch.cat(all_r,     dim=0)
        v_final     = torch.cat(all_v,     dim=0)
        theta_final = torch.cat(all_theta, dim=0)
        c_final     = torch.cat(all_c,     dim=0)
        print(f"x_final: {x_final.shape}, y_final: {y_final.shape}, r_final: {r_final.shape}, v_final: {v_final.shape}, theta_final: {theta_final.shape}, c_final: {c_final.shape}")

        return x_final, y_final, r_final, v_final, theta_final, c_final
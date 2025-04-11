import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

class Initializer:
    def __init__(self, template_matching=False, N=10000, 
                 min_distance=5, peak_threshold=0.5,
                 radii_min=2, radii_max=None,
                 arc_thickness=1, v_init_mean=-5.0):
        """
        Initialize the parameters for the model.

        Parameters:
        - template_matching: bool, whether to use template matching for initialization
        - N: int, number of primitives (circles or arbitrary shapes) to initialize
        - min_distance: int, minimum distance between detected peaks
        - peak_threshold: float, threshold for peak detection
        - radii_min: float, minimum radius (or scale) for initialization
        - radii_max: float, maximum radius (or scale) for initialization
        - arc_thickness: float, thickness of the circle arc in pixels (for circle initialization)
        - v_init_mean: float, mean for the initial intensity parameter (pre-sigmoid)
        """
        self.template_matching = template_matching
        self.N = N
        self.min_distance = min_distance
        self.peak_threshold = peak_threshold
        self.radii_min = radii_min
        self.radii_max = radii_max
        self.arc_thickness = arc_thickness
        self.v_init_mean = v_init_mean

    def initialize_circles_with_template_matching(self, I_target):
        """
        Initialize circle parameters (x, y, r, v) using template matching with circle arcs.

        Parameters:
          - I_target: torch.Tensor, the target grayscale image (H x W)
        Returns:
          - x_init: torch.Tensor, initialized x-coordinates of circle centers
          - y_init: torch.Tensor, initialized y-coordinates of circle centers
          - r_init: torch.Tensor, initialized radii of circles
          - v_init: torch.Tensor, initialized intensity parameters (pre-sigmoid)
        """
        device = I_target.device
        H, W = I_target.shape
        if self.radii_max is None:
            self.radii_max = min(H, W) // 2
        
        radii = range(self.radii_min, min(H,W)//2 if self.radii_max is None else self.radii_max + 1)

        x_list, y_list, r_list, v_list = [], [], [], []

        for r in radii:
            kernel_size = int(2 * r + 1)
            y, x = torch.meshgrid(
                torch.arange(-r, r + 1, device=device),
                torch.arange(-r, r + 1, device=device),
                indexing='ij'
            )
            dist = torch.sqrt(x**2 + y**2)
            kernel = ((dist >= r - self.arc_thickness / 2) & (dist <= r + self.arc_thickness / 2)).float()
            kernel /= kernel.sum()

            C_r = F.conv2d(
                I_target.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=int(r)
            ).squeeze()

            C_r_np = C_r.cpu().numpy()
            max_filtered = maximum_filter(C_r_np, size=self.min_distance)
            peaks = (C_r_np == max_filtered) & (C_r_np > self.peak_threshold)
            peak_coords = np.argwhere(peaks)

            for py, px in peak_coords:
                x_list.append(px)
                y_list.append(py)
                r_list.append(r)
                I_bar = C_r[py, px].item()
                if I_bar <= 0:
                    v_i = -10.0
                elif I_bar >= 1:
                    v_i = 10.0
                else:
                    I_bar = np.clip(I_bar, 1e-3, 1 - 1e-3, dtype=np.float32)
                    v_i = np.log(I_bar / (1 - I_bar), dtype=np.float32)
                v_list.append(v_i)

        x_init = torch.tensor(x_list, device=device)
        y_init = torch.tensor(y_list, device=device)
        r_init = torch.tensor(r_list, device=device)
        v_init = torch.tensor(v_list, device=device)

        if len(v_init) > self.N:
            print(f"Warning: More than {self.N} circles detected. Selecting the top {self.N} based on intensity.")
            v_init, indices = torch.sort(v_init, descending=True)
            v_init = v_init[:self.N]
            x_init = x_init[indices[:self.N]]
            y_init = y_init[indices[:self.N]]
            r_init = r_init[indices[:self.N]]

        return x_init, y_init, r_init, v_init

    def initialize_circles(self, I_target):
        device = I_target.device
        H, W = I_target.shape
        if self.radii_max is None:
            self.radii_max = min(H, W) // 2
        N = self.N
        if self.template_matching:
            x, y, r, v = self.initialize_circles_with_template_matching(I_target)
            torch.manual_seed(0)
            if len(x) < N:
                additional = N - len(x)
                x = torch.cat((x, torch.rand(additional, device=device) * W))
                y = torch.cat((y, torch.rand(additional, device=device) * H))
                r = torch.cat((r, torch.rand(additional, device=device) * (self.radii_max - self.radii_min) + self.radii_min))
                v = torch.cat((v, torch.randn(additional, device=device) + self.v_init_mean))
        else:
            torch.manual_seed(0)
            x = torch.rand(N, device=device) * W
            y = torch.rand(N, device=device) * H
            r = torch.rand(N, device=device) * (self.radii_max - self.radii_min) + self.radii_min
            v = torch.randn(N, device=device) + self.v_init_mean
        
        return x, y, r, v

    def initialize_circles_color(self, I_target):
        """
        When I_target is a (H, W, 3) color image.
        """
        device = I_target.device
        H, W, _ = I_target.shape
        if self.radii_max is None:
            self.radii_max = min(H, W) // 2
        N = self.N

        if self.template_matching:
            I_target_gray = I_target.mean(dim=-1)
            x, y, r, v = self.initialize_circles_with_template_matching(I_target_gray)
            torch.manual_seed(0)
            if len(x) < N:
                additional = N - len(x)
                x = torch.cat((x, torch.rand(additional, device=device) * W))
                y = torch.cat((y, torch.rand(additional, device=device) * H))
                r = torch.cat((r, torch.rand(additional, device=device) * (self.radii_max - self.radii_min) + self.radii_min))
                v = torch.cat((v, torch.randn(additional, device=device) + self.v_init_mean))
        else:
            torch.manual_seed(0)
            x = torch.rand(N, device=device) * W
            y = torch.rand(N, device=device) * H
            r = torch.rand(N, device=device) * (self.radii_max - self.radii_min) + self.radii_min
            v = torch.randn(N, device=device) + self.v_init_mean

        return x, y, r, v

    def initialize_triangles(self, I_target):
        """
        Initialize parameters for equilateral triangles.
        Geometry is fixed: for each triangle, randomly generate a center (x, y), 
        a scale factor (r, representing the circumradius), and a rotation angle.
        Then, compute the 3 vertices of the triangle.
        Only the intensity parameter v (to be passed through sigmoid to get alpha) is learnable.

        Parameters:
          - I_target: torch.Tensor, target image; used only for extracting dimensions.
                     Acceptable shape: (H, W) for grayscale or (H, W, 3) for color.
        
        Returns:
          - triangles: a list of torch.Tensor, each of shape (3, 2) representing vertices of a triangle.
          - v_init: a torch.Tensor of shape (N,) with the initialized intensity parameters.
        """
        device = I_target.device
        # Extract image size: if I_target is color, use first two dims.
        if I_target.ndim == 3:
            H, W, _ = I_target.shape
        else:
            H, W = I_target.shape
        N = self.N
        # Define default radii_max if not provided
        if self.radii_max is None:
            self.radii_max = min(H, W) / 4  # 예: 이미지 크기의 1/4 이내로
        
        torch.manual_seed(0)
        # Random centers uniformly in the image bounds
        x_centers = torch.rand(N, device=device) * W
        y_centers = torch.rand(N, device=device) * H
        # Random scale (circumradius) between radii_min and radii_max
        r_vals = torch.rand(N, device=device) * (self.radii_max - self.radii_min) + self.radii_min
        # Random rotation angle in [0, 2*pi)
        rotations = torch.rand(N, device=device) * 2 * np.pi
        
        triangles = []
        for i in range(N):
            # For an equilateral triangle, vertices are separated by 120°.
            center = torch.tensor([x_centers[i], y_centers[i]], device=device)
            r_val = r_vals[i]
            theta = rotations[i]
            vertices = []
            for j in range(3):
                angle = theta + 2 * np.pi * j / 3
                vertex = center + r_val * torch.tensor([torch.cos(angle), torch.sin(angle)], device=device)
                vertices.append(vertex.unsqueeze(0))  # shape (1,2)
            # Concatenate vertices -> (3,2)
            triangle = torch.cat(vertices, dim=0)
            triangles.append(triangle)
        
        # Initialize intensity parameter v for each triangle
        v_init = torch.randn(N, device=device) + self.v_init_mean
        
        return triangles, v_init
    
    def initialize_for_svg(self, I_target):

        device = I_target.device
        # Extract image size: if I_target is color, use first two dims.
        if I_target.ndim == 3:
            H, W, _ = I_target.shape
        else:
            H, W = I_target.shape
        N = self.N
        # Define default radii_max if not provided
        if self.radii_max is None:
            self.radii_max = min(H, W) / 4  # 예: 이미지 크기의 1/4 이내로

        x = torch.rand(N, device=device) * W  # 중심 x 좌표
        y = torch.rand(N, device=device) * H  # 중심 y 좌표
        r = torch.rand(N, device=device) * min(H,W) / 8 + min(H,W) / 32  # 배율 (초기값 1)
        v = torch.full((N,), self.v_init_mean, device=device)  # alpha 초깃값
        theta = torch.rand(N, device=device) * 2 * np.pi  # 회전각 (0~2π)
        return x, y, r, v, theta

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
        Initialize the circle parameters for the model.

        Parameters:
        - template_matching: bool, whether to use template matching for initialization
        - N: int, number of circles to initialize
        - min_distance: int, minimum distance between detected peaks
        - peak_threshold: float, threshold for peak detection
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
        - radii: list of floats, the radii to consider for circle arcs
        - arc_thickness: float, thickness of the circle arc in pixels (default: 1)
        - peak_threshold: float, threshold for peak detection in the response map (default: 0.5)
        - min_distance: int, minimum distance between peaks to avoid overlapping circles (default: 5)

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

        # Lists to store initialized parameters
        x_list, y_list, r_list, v_list = [], [], [], []

        # Iterate over each radius
        for r in radii:
            kernel_size = int(2 * r + 1)
            y, x = torch.meshgrid(
                torch.arange(-r, r + 1, device=device),
                torch.arange(-r, r + 1, device=device),
                indexing='ij'
            )
            dist = torch.sqrt(x**2 + y**2)
            kernel = ((dist >= r - self.arc_thickness / 2) & (dist <= r + self.arc_thickness / 2)).float()
            kernel /= kernel.sum()  # Normalize kernel

            # Convolution
            C_r = F.conv2d(
                I_target.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=int(r)
            ).squeeze()

            # Peak detection
            C_r_np = C_r.cpu().numpy()
            max_filtered = maximum_filter(C_r_np, size=self.min_distance)
            peaks = (C_r_np == max_filtered) & (C_r_np > self.peak_threshold)
            peak_coords = np.argwhere(peaks)

            # Store parameters for each detected circle
            for py, px in peak_coords:
                x_list.append(px)
                y_list.append(py)
                r_list.append(r)
                # Compute v based on average intensity at the peak
                I_bar = C_r[py, px].item()
                if I_bar <= 0:
                    v_i = -10.0
                elif I_bar >= 1:
                    v_i = 10.0
                else:
                    I_bar = np.clip(I_bar, 1e-3, 1 - 1e-3, dtype=np.float32)
                    v_i = np.log(I_bar / (1 - I_bar), dtype=np.float32)
                v_list.append(v_i)

        # Convert lists to tensors
        x_init = torch.tensor(x_list, device=device)
        y_init = torch.tensor(y_list, device=device)
        r_init = torch.tensor(r_list, device=device)
        v_init = torch.tensor(v_list, device=device)

        # v_init의 길이가 self.N보다 길면 큰 순서대로 self.N개만 선택
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
        # Initialize circle parameters
        if self.template_matching:
            x, y, r, v = self.initialize_circles_with_template_matching(I_target)
            torch.manual_seed(0)
            # Pad with random values if fewer than N circles are detected
            if len(x) < N:
                additional = N - len(x)
                x = torch.cat((x, torch.rand(additional, device=device) * W))
                y = torch.cat((y, torch.rand(additional, device=device) * H))
                r = torch.cat((r, torch.rand(additional, device=device) * (self.radii_max - self.radii_min) + self.radii_min))
                v = torch.cat((v, torch.randn(additional, device=device) + self.v_init_mean))
        else:
            # Random initialization
            torch.manual_seed(0)
            x = torch.rand(N, device=device) * W
            y = torch.rand(N, device=device) * H
            r = torch.rand(N, device=device) * (self.radii_max - self.radii_min) + self.radii_min
            v = torch.randn(N, device=device) + self.v_init_mean
        
        return x,y,r,v
    

    def initialize_circles_color(self, I_target):
        """
        I_target이 (H, W, 3) 형태의 컬러 이미지일 때 동작하도록 수정된 함수.
        """
        device = I_target.device
        H, W, _ = I_target.shape  # 컬러 이미지의 크기 추출
        if self.radii_max is None:
            self.radii_max = min(H, W) // 2
        N = self.N

        # Initialize circle parameters
        if self.template_matching:
            # I_target을 (H, W, 3)에서 (H, W)로 변환 (예: 밝기 채널 사용)
            I_target_gray = I_target.mean(dim=-1)  # RGB 평균값으로 흑백 변환
            x, y, r, v = self.initialize_circles_with_template_matching(I_target_gray)
            torch.manual_seed(0)
            # Pad with random values if fewer than N circles are detected
            if len(x) < N:
                additional = N - len(x)
                x = torch.cat((x, torch.rand(additional, device=device) * W))
                y = torch.cat((y, torch.rand(additional, device=device) * H))
                r = torch.cat((r, torch.rand(additional, device=device) * (self.radii_max - self.radii_min) + self.radii_min))
                v = torch.cat((v, torch.randn(additional, device=device) + self.v_init_mean))
        else:
            # Random initialization
            torch.manual_seed(0)
            x = torch.rand(N, device=device) * W
            y = torch.rand(N, device=device) * H
            r = torch.rand(N, device=device) * (self.radii_max - self.radii_min) + self.radii_min
            v = torch.randn(N, device=device) + self.v_init_mean

        return x, y, r, v
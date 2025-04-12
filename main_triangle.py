import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import imageio
from tqdm import tqdm
from PIL import Image
import json
import argparse

# Preprocessor와 Initializer는 사용자의 환경에 맞게 정의되어 있다고 가정합니다.
from preprocessing import Preprocessor
from initialization import Initializer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser setup
parser = argparse.ArgumentParser(description="Process some images.")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()
config_path = args.config

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

pp_conf = config["preprocessing"]
preprocessor = Preprocessor(
    final_width=pp_conf.get("final_width", 128),
    trim=pp_conf.get("trim", False),
    FM_halftone=pp_conf.get("FM_halftone", False),
    transform_mode=pp_conf.get("transform", "none"),
)
# 컬러 이미지 로드
I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
I_target = torch.tensor(I_target, device=device)  # (H, W, 3)
H = preprocessor.final_height
W = preprocessor.final_width

# 벡터 그래픽 초기화 (예: 정삼각형)
print("---Initializing vector graphics (triangles)---")
initializer = Initializer(
    template_matching=config["initialization"].get("template_matching", False),
    N=config["initialization"].get("N", 10000),
    min_distance=config["initialization"].get("min_distance", 5),
    peak_threshold=config["initialization"].get("peak_threshold", 0.5),
    radii_min=config["initialization"].get("radii_min", 2),
    radii_max=config["initialization"].get("radii_max", None),
    arc_thickness=config["initialization"].get("arc_thickness", 1),
    v_init_mean=config["initialization"].get("v_init_mean", -5.0),
)
# initialize_triangles() 반환: (paths, v_init)
paths, v = initializer.initialize_triangles(I_target)  
# paths: list of torch.Tensor, 각 텐서의 shape은 (n_points, 2)
N = len(paths)

# learnable 파라미터 v (scalar)와 c (3-vector) 초기화
v = v.clone().detach().requires_grad_(True)
c = torch.rand(N, 3, device=device, requires_grad=True)  # 색상, 초기 [0,1] 범위

# Precompute pixel coordinates (for image grid)
X, Y = torch.meshgrid(torch.arange(W, device=device),
                      torch.arange(H, device=device), indexing='xy')
# X,Y shape: (H, W) -> (1, H, W) for broadcasting
X, Y = X.unsqueeze(0), Y.unsqueeze(0)

alpha_upper_bound = config["optimization"].get("alpha_upper_bound", 0.5)

#############################################
# Cached Soft Rasterization of Fixed Geometry
#############################################
def soft_rasterize_shape(vertices, X, Y, thickness=1.0, sigma=1.0):
    """
    vertices: (n_points, 2) tensor of the polygon boundary (in (x,y) order)
    X, Y: (1, H, W) coordinate grids (with X as x-coordinate, Y as y-coordinate)
    반환: mask (1, H, W): 1에 가까울수록 경계에 가까움.
    
    각 픽셀에 대해, 각 선분의 point-to-segment 거리를 계산하고,
    최소 거리에 대해 exp(- (dist/sigma)**2)로 soft mask를 생성.
    (thickness 내에 있으면 유지)
    """
    n_points = vertices.shape[0]
    H = X.shape[1]
    W = X.shape[2]
    d_min = torch.full((1, H, W), float('inf'), device=vertices.device)
    
    for i in range(n_points):
        p1 = vertices[i]
        p2 = vertices[(i+1) % n_points]  # closed polygon
        
        v_seg = p2 - p1  # (2,)
        p1_exp = p1.view(1,1,2).expand(H, W, 2)
        # grid in (x,y) order: X is x, Y is y.
        grid = torch.stack([X.squeeze(0), Y.squeeze(0)], dim=-1)  # (H, W, 2)
        v_seg_exp = v_seg.view(1,1,2).expand(H, W, 2)
        
        w = grid - p1_exp
        t = (w * v_seg_exp).sum(dim=-1) / (v_seg_exp.norm(dim=-1)**2 + 1e-6)
        t = t.clamp(0,1).unsqueeze(-1)
        proj = p1_exp + t * v_seg_exp
        d = (grid - proj).norm(dim=-1, keepdim=True)
        d = d.squeeze(-1).unsqueeze(0)  # (1,H,W)
        d_min = torch.min(d_min, d)
    
    mask = torch.exp(- (d_min / sigma)**2)
    mask = mask * (d_min < thickness).float()
    return mask  # (1, H, W)

#############################################
# Over Operator & Tree-based Reduction Functions (동일)
#############################################
def over_pair(m1, a1, m2, a2):
    m_out = m1 + (1 - a1).unsqueeze(-1) * m2
    a_out = a1 + (1 - a1) * a2
    return m_out, a_out

def tree_over(m, a):
    while m.size(0) > 1:
        n = m.size(0)
        if n % 2 == 1:
            pad_m = torch.zeros((1, *m.shape[1:]), device=m.device, dtype=m.dtype)
            pad_a = torch.zeros((1, *a.shape[1:]), device=a.device, dtype=a.dtype)
            m = torch.cat([m, pad_m], dim=0)
            a = torch.cat([a, pad_a], dim=0)
            n = m.size(0)
        new_n = n // 2
        m = m.reshape(new_n, 2, m.size(1), m.size(2), 3)
        a = a.reshape(new_n, 2, a.size(1), a.size(2))
        m = m[:, 0] + (1 - a[:, 0]).unsqueeze(-1) * m[:, 1]
        a = a[:, 0] + (1 - a[:, 0]) * a[:, 1]
    return m.squeeze(0), a.squeeze(0)

# 캐시: 각 도형별 mask는 geometry가 고정이므로, 한 번 계산 후 재사용.
_cached_masks = []
for i in range(N):
    mask_i = soft_rasterize_shape(paths[i], X, Y, thickness=0.5, sigma=1.0)
    _cached_masks.append(mask_i)
# _cached_masks: Tensor of shape (N, H, W)
_cached_masks = torch.cat(_cached_masks, dim=0)

#############################################
# Render Image Function (Using Cached Masks)
#############################################
def render_image_vector_cached(cached_masks, v, c, X, Y):
    """
    cached_masks: precomputed masks for each shape, shape (N, H, W)
    v: learnable parameter for each shape (N,), will be passed through sigmoid and scaled
    c: learnable color for each shape (N, 3), passed through sigmoid to be in [0,1]
    X, Y: (1, H, W) pixel coordinate grids.
    
    각 도형의 effective alpha = alpha_upper_bound * sigmoid(v) * cached_mask
    effective premultiplied color = effective_alpha * sigmoid(c)
    모든 도형을 over operator (tree_over)로 합성한 후, 흰색 배경과 compositing.
    """
    N = v.shape[0]
    v_alpha = alpha_upper_bound * torch.sigmoid(v).view(N, 1, 1)
    # Effective alpha per shape (N, H, W)
    a = v_alpha * cached_masks
    # Compute effective colors in [0,1]
    c_eff = torch.sigmoid(c)
    # Reshape to (N, 1, 1, 3)
    c_reshaped = c_eff.view(N, 1, 1, 3)
    # Premultiplied color: (N, H, W, 3)
    m = a.unsqueeze(-1) * c_reshaped
    # Composite with tree-based over operator
    comp_m, comp_a = tree_over(m, a)
    background = torch.ones_like(comp_m)  # opaque white background
    final = comp_m + (1 - comp_a).unsqueeze(-1) * background
    return final  # (H, W, 3)

#############################################
# Training Loop (using cached masks)
#############################################
num_iterations = config["optimization"].get("num_iterations", 300)
learning_rate = config["optimization"].get("learning_rate", 0.1)
optimizer = torch.optim.Adam([v, c], lr=learning_rate)

for epoch in range(num_iterations):
    optimizer.zero_grad()
    I_hat = render_image_vector_cached(_cached_masks, v, c, X, Y)
    loss = F.mse_loss(I_hat, I_target)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#############################################
# Visualization Function
#############################################
def visualize_results(I_target, I_hat):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(I_target.cpu().detach().numpy())
    ax[0].set_title("Target Image")
    ax[0].axis('off')
    
    ax[1].imshow(I_hat.cpu().detach().numpy())
    ax[1].set_title("Reconstructed Image (Vector Graphics)")
    ax[1].axis('off')
    plt.show()

I_hat = render_image_vector_cached(_cached_masks, v, c, X, Y)
visualize_results(I_target, I_hat)

#############################################
def export_vector_graphics_to_pdf(paths, v, c,
                                  pdf_filename="vector_art.pdf",
                                  canvas_size=(8, 8)):
    # PDF 저장용 디렉토리 생성
    directory = os.path.dirname(pdf_filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Figure와 axis 생성
    fig, ax = plt.subplots(figsize=canvas_size)
    
    # 현재 learnable 파라미터 값 (sigmoid 적용 후)
    alpha_vals = (alpha_upper_bound * torch.sigmoid(v)).detach().cpu().numpy()
    colors = torch.sigmoid(c).detach().cpu().numpy()
    
    from matplotlib.path import Path
    import matplotlib.patches as mpatches
    
    # 도형들을 추가 (맨 밑 레이어부터 그리기 위해 reversed 순서로)
    patches_list = list(zip(paths, alpha_vals, colors))
    for vertices, alpha, col in reversed(patches_list):
        verts = vertices.cpu().detach().numpy()
        # Path 코드: 첫 점은 MOVETO, 나머지는 LINETO, 마지막에 CLOSEPOLY
        codes = [Path.MOVETO] + [Path.LINETO]*(verts.shape[0]-1) + [Path.CLOSEPOLY]
        verts = np.vstack([verts, verts[0]])
        path_obj = Path(verts, codes)
        patch = mpatches.PathPatch(path_obj,
                                   linewidth=config["postprocessing"].get("linewidth", 3.0),
                                   edgecolor=(col[0], col[1], col[2], alpha),
                                   facecolor='none')
        ax.add_patch(patch)
    
    # 축 설정: 좌표계 보존 (y축 반전 포함)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    # PDF 저장 (단일 페이지 PDF로)
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"PDF saved as {pdf_filename}")

# 실제 PDF export 실행:
output_pdf = config["postprocessing"].get("output_path", "vector_art.pdf")
export_vector_graphics_to_pdf(paths, v, c, pdf_filename=output_pdf, canvas_size=(8,8))
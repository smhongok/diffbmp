import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.patches import Circle
import imageio
from tqdm import tqdm
from PIL import Image, ImageOps
import json
import argparse

# Assuming Preprocessor is defined in preprocessing.py
from preprocessing import Preprocessor
from initialization import Initializer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser setup
parser = argparse.ArgumentParser(description="Process some images.")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()
config_path = args.config

# 1) Load JSON config
print(f"---Loading config from {config_path}---")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 2) Preprocessing setup
print("---Preprocessing---")
pp_conf = config["preprocessing"]

preprocessor = Preprocessor(
    final_width=pp_conf.get("final_width", 128),
    trim=pp_conf.get("trim", False),
    FM_halftone=pp_conf.get("FM_halftone", False),
    transform_mode=pp_conf.get("transform", "none"),
)

# 여기서는 흑백 대신 컬러 이미지를 로드하도록 합니다.
# (Preprocessor 내부에 load_image_8bit_color 메서드가 있거나,
# PIL을 사용하여 RGB로 변환하도록 수정할 수 있음)
I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
# I_target 의 shape는 (H, W, 3)
I_target = torch.tensor(I_target, device=device)

H = preprocessor.final_height
W = preprocessor.final_width

initizlier = Initializer(
    template_matching=config["initialization"].get("template_matching", False),
    N=config["initialization"].get("N", 10000),
    min_distance=config["initialization"].get("min_distance", 5),
    peak_threshold=config["initialization"].get("peak_threshold", 0.5),
    radii_min=config["initialization"].get("radii_min", 2),
    radii_max=config["initialization"].get("radii_max", None),
    arc_thickness=config["initialization"].get("arc_thickness", 1),
    v_init_mean=config["initialization"].get("v_init_mean", -5.0),
)

# 3) Initialize circle parameters (x, y, r, v)
print("---Initializing circles---")
x, y, r, v = initizlier.initialize_circles_color(I_target)

# 새로운 learnable parameter: 각 원의 색상 (N, 3)
# 초기값은 무작위로 [0, 1] 사이 값으로 설정 (또는 원하는 초기화 기법 사용)
c = torch.rand(x.shape[0], 3, device=device)

# 4) Convert to tensors and move to device
v = v.clone().detach().requires_grad_(True)
c = c.clone().detach().requires_grad_(True)

# Precompute pixel coordinates
X, Y = torch.meshgrid(torch.arange(W, device=device),
                      torch.arange(H, device=device), indexing='xy')
# X, Y shape: (H, W); unsqueeze to (1, H, W) for broadcasting
X, Y = X.unsqueeze(0), Y.unsqueeze(0)

# 5) Set up the optimizer
num_iterations = config["optimization"].get("num_iterations", 300)
learning_rate = config["optimization"].get("learning_rate", 0.1)
alpha_upper_bound = config["optimization"].get("alpha_upper_bound", 0.5)
optimizer = torch.optim.Adam([v, c], lr=learning_rate)

#############################################
# Over Operator & Tree-based Reduction
#############################################
def over_pair(m1, a1, m2, a2):
    """
    Over operator for two layers (in premultiplied form).
    m1, m2: shape (H, W, 3)
    a1, a2: shape (H, W), representing alpha
    Returns:
      m_out = m1 + (1 - a1) * m2
      a_out = a1 + (1 - a1) * a2
    """
    m_out = m1 + (1 - a1).unsqueeze(-1) * m2
    a_out = a1 + (1 - a1) * a2
    return m_out, a_out

def tree_over(m, a):
    """
    m: Tensor of shape (N, H, W, 3) -- premultiplied colors for each layer
    a: Tensor of shape (N, H, W)   -- effective alpha per layer
    Return: composite m (H, W, 3), a (H, W)
    계산은 over_pair를 벡터화한 트리 기반 reduction으로 수행.
    """
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
        # Pairwise over
        m = m[:, 0] + (1 - a[:, 0]).unsqueeze(-1) * m[:, 1]
        a = a[:, 0] + (1 - a[:, 0]) * a[:, 1]
    return m.squeeze(0), a.squeeze(0)

#############################################
# Render image function (Color Version)
#############################################
def render_image(x, y, r, v, c, X, Y):
    """
    각 원마다:
      - v_alpha = sigmoid(v) (shape: (N, 1, 1))
      - circle_mask: (N, H, W) as before
      - effective alpha: a_i = v_alpha * circle_mask
      - effective premultiplied color: m_i = a_i * sigmoid(c)
        (여기서 c는 sigmoid를 통해 [0, 1] 범위로 제한됨)
    모든 원을 over blending (트리 기반 reduction)하여 최종 색상을 계산합니다.
    최종 배경은 흰색 (1,1,1)으로 처리합니다.
    """
    # Reshape parameters
    x = x.view(-1, 1, 1)
    y = y.view(-1, 1, 1)
    r = r.view(-1, 1, 1)
    v_alpha = alpha_upper_bound * torch.sigmoid(v).view(-1, 1, 1)  # (N,1,1)

    # Compute distance and mask (same as before)
    dist = torch.sqrt((X - x)**2 + (Y - y)**2)
    thickness = 0.5
    circle_mask = ((dist >= (r - thickness)) & (dist <= (r + thickness))).float()  # (N, H, W)

    # Effective alpha per pixel for each circle: (N, H, W)
    a = v_alpha * circle_mask

    # Ensure color values c are in [0,1] via sigmoid
    c_eff = torch.sigmoid(c)
    # c_eff: (N, 3) -> reshape to (N, 1, 1, 3)
    c_reshaped = c_eff.view(-1, 1, 1, 3)
    # Premultiplied color: m_i = a_i * c_i, broadcast a to (N,H,W,1)
    m = a.unsqueeze(-1) * c_reshaped  # (N, H, W, 3)

    # Composite all circles using tree-based over operator reduction
    comp_m, comp_a = tree_over(m, a)  # comp_m: (H, W, 3), comp_a: (H, W)

    # Composite with background (opaque white: (1,1,1))
    background = torch.ones_like(comp_m)  # (H, W, 3)
    final = comp_m + (1 - comp_a).unsqueeze(-1) * background
    # 최종 합성 후 alpha는 1 (배경 불투명)
    return final  # shape (H, W, 3)

# 6) Training loop
for epoch in range(num_iterations):
    optimizer.zero_grad()
    I_hat = render_image(x, y, r, v, c, X, Y)  # I_hat: (H, W, 3)
    loss = F.mse_loss(I_hat, I_target)  # target is also (H, W, 3)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#############################################
# Visualization & PDF Export Functions
#############################################
def visualize_results(I_target, I_hat):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # I_target: (H, W, 3) - numpy conversion
    ax[0].imshow(I_target.cpu().detach().numpy())
    ax[0].set_title("Target Image")
    ax[0].axis('off')
    
    ax[1].imshow(I_hat.cpu().detach().numpy())
    ax[1].set_title("Reconstructed Image (Circles)")
    ax[1].axis('off')
    plt.show()

# 최종 결과 시각화
I_hat = render_image(x, y, r, v, c, X, Y)
visualize_results(I_target, I_hat)

def export_circles_to_pdf(x, y, r, v, c, filename="circle_art.pdf", canvas_size=(8, 8)):
    # 디렉토리 생성
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    fig, ax = plt.subplots(figsize=canvas_size)
    
    # alpha 값을 [0,1]로 정규화
    alpha_vals = alpha_upper_bound * torch.sigmoid(v).detach().cpu().numpy()
    # c: (N, 3)
    colors = torch.sigmoid(c).detach().cpu().numpy()
    
    for xi, yi, ri, alpha, col in reversed(list(zip(x.cpu(), y.cpu(), r.cpu(), alpha_vals, colors))):
        circle = patches.Circle((xi, yi), ri,
                                linewidth=config["postprocessing"].get("linewidth", 3.0),  
                                edgecolor=(col[0], col[1], col[2], alpha),  
                                facecolor='none')
        ax.add_patch(circle)
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # y축 반전
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# 실제 PDF export 실행
# output_path = config["postprocessing"].get("output_path", "circle_art.pdf")
# export_circles_to_pdf(x, y, r, v, c, filename=output_path)


#############################################
# PDF & GIF Export 함수: layer-wise 누적 결과를 캡처
#############################################
def export_circles_to_pdf_and_gif(x, y, r, v, c,
                                  pdf_filename="circle_art.pdf",
                                  gif_filename="layer_progress.gif",
                                  canvas_size=(8, 8),
                                  frame_interval=200):
    # 디렉토리 생성
    directory_pdf = os.path.dirname(pdf_filename)
    if directory_pdf and not os.path.exists(directory_pdf):
        os.makedirs(directory_pdf)
    directory_gif = os.path.dirname(gif_filename)
    if directory_gif and not os.path.exists(directory_gif):
        os.makedirs(directory_gif)
    
    # figure와 axis 생성
    fig, ax = plt.subplots(figsize=canvas_size)
    
    # PDF용 전체 도형은 밑쪽부터 쌓기 위해 reversed 순서로 처리
    alpha_vals = torch.sigmoid(v).detach().cpu().numpy()
    colors = torch.sigmoid(c).detach().cpu().numpy()
    
    # GIF 프레임 저장 리스트
    frames = []
    
    # for문을 통해 각 도형을 하나씩 add_patch
    patches_list = list(zip(x.cpu(), y.cpu(), r.cpu(), alpha_vals, colors))
    # 일반적으로 PDF에서는 맨 밑 레이어부터 그리므로 reversed 순서로 처리합니다.
    for idx, (xi, yi, ri, alpha, col) in enumerate(reversed(patches_list)):
        circle = patches.Circle((xi, yi), ri,
                                linewidth=config["postprocessing"].get("linewidth", 3.0),
                                edgecolor=(col[0], col[1], col[2], alpha),
                                facecolor='none')
        ax.add_patch(circle)
        
        # 매 frame_interval마다 현재 canvas 상태를 프레임으로 캡처
        if (idx + 1) % frame_interval == 0 or idx == len(patches_list) - 1:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # 해상도를 낮추기 위해 PIL을 사용하여 이미지 크기를 조절
            pil_img = Image.fromarray(image)
            new_size = (pp_conf.get("final_width", 128), int(pp_conf.get("final_width", 128)*H /W))
            pil_img = pil_img.resize(new_size, resample=Image.BILINEAR)
            image = np.array(pil_img)
            frames.append(image.copy())
    
    # figure 설정
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # y축 상하 반전 (일반 이미지 좌표계)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    # PDF 저장
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # GIF 저장 (duration은 프레임 간 간격, 0.1초 정도 권장)
    imageio.mimsave(gif_filename, frames, duration=0.1)
    print(f"PDF saved as {pdf_filename} and GIF saved as {gif_filename}")

# 실제 PDF와 GIF export 실행
output_pdf = config["postprocessing"].get("output_path", "circle_art.pdf")
output_gif = config["postprocessing"].get("gif_output_path", "layer_progress.gif")
export_circles_to_pdf_and_gif(x, y, r, v, c, pdf_filename=output_pdf, gif_filename=output_gif, canvas_size=(8, 8), frame_interval=200)
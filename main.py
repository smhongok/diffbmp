import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle

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

# Load and preprocess the target image
I_target = preprocessor.load_image_8bit_gray(config["preprocessing"]).astype(np.float32) / 255.0
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
# 3) Initialize circle parameters
print("---Initializing circles---")
x, y, r, v = initizlier.initialize_circles(I_target)

# 4) Convert to tensors and move to device
# Make v a learnable parameter
v = v.clone().detach().requires_grad_(True)

# Precompute pixel coordinates
X, Y = torch.meshgrid(torch.arange(W, device=device),
                      torch.arange(H, device=device), indexing='xy')
X, Y = X.unsqueeze(0), Y.unsqueeze(0)  # shape [1, W, H]

# 5) Set up the optimizer
num_iterations = config["optimization"].get("num_iterations", 300)
learning_rate = config["optimization"].get("learning_rate", 0.1)

# Optimization setup
optimizer = torch.optim.Adam([v], lr=learning_rate)

# Render image function (unchanged)
def render_image(x, y, r, v, X, Y):
    x = x.view(-1, 1, 1)
    y = y.view(-1, 1, 1)
    r = r.view(-1, 1, 1)
    v_alpha = torch.sigmoid(v).view(-1, 1, 1)

    dist = torch.sqrt((X - x)**2 + (Y - y)**2)
    thickness = 0.5
    circle_mask = ((dist >= (r - thickness)) & (dist <= (r + thickness))).float()

    img = torch.prod(1 - v_alpha * circle_mask, dim=0)
    img = 1 - img

    return img

# Training loop
for epoch in range(num_iterations):
    optimizer.zero_grad()
    I_hat = render_image(x, y, r, v, X, Y)
    loss = F.mse_loss(I_hat, I_target)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 결과 시각화
def visualize_results(I_target, I_hat):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(I_target.cpu().detach(), cmap='gray')
    ax[0].set_title("Target Image")
    ax[0].axis('off')

    ax[1].imshow(I_hat.cpu().detach(), cmap='gray')
    ax[1].set_title("Reconstructed Image (Circles)")
    ax[1].axis('off')

    plt.show()

# 최종 결과 시각화
visualize_results(I_target, I_hat)

# PDF로 export하는 함수
def export_circles_to_pdf(x, y, r, v, filename="circle_art.pdf", canvas_size=(8, 8)):
    # 디렉토리 경로 추출
    directory = os.path.dirname(filename)
    
    # 디렉토리가 존재하지 않으면 생성
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plt.subplots(figsize=canvas_size)

    # 강도를 [0,1] 사이로 정규화 (sigmoid 사용)
    alpha_vals = torch.sigmoid(v).detach().cpu().numpy()

    # 원들을 PDF에 추가
    for xi, yi, ri, alpha in zip(x.cpu(), y.cpu(), r.cpu(), alpha_vals):
        circle = patches.Circle((xi, yi), ri,
                                linewidth=3.,  # 적당한 선
                                edgecolor=(0,0,0,alpha),  # 검은색 + 알파값 (투명도)
                                facecolor='none')
        ax.add_patch(circle)

    # 이미지 범위 설정
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # y축 상하 반전 (일반 이미지 좌표계)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')

    # PDF로 저장
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

    plt.close(fig)

# 실제 PDF export 실행
output_path = config["postprocessing"].get("output_path", "circle_art.pdf")
export_circles_to_pdf(x, y, r, v, filename=output_path)



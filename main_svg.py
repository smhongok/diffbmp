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
import tempfile
import argparse
import svgpathtools
from svgpathtools import svg2paths

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
# 타겟 컬러 이미지 로드
I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
I_target = torch.tensor(I_target, device=device)  # (H, W, 3)
H = preprocessor.final_height
W = preprocessor.final_width

# SVG 파일 로드 및 처리
svg_file = config["svg"].get("svg_file", "images/apple_logo_hollow.svg")
paths, attributes = svg2paths(svg_file)

# SVG를 고화질 bmp로 변환 (실제로는 cairosvg 등으로 렌더링 필요)
# 여기서는 예시로 PIL을 사용해 임시 bmp를 생성한다고 가정
from cairosvg import svg2png

# 임시 파일 생성
with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
    # SVG를 PNG로 변환
    svg2png(url=svg_file, write_to=temp_file.name)
    
    # RGBA 이미지로 읽기
    bmp_image = Image.open(temp_file.name).convert('RGBA')
    bmp_image = np.array(bmp_image)
    
# 원본 이미지 크기 얻기
orig_h, orig_w, _ = bmp_image.shape
new_size = max(orig_h, orig_w)
old_size = min(orig_h, orig_w)

# 상하, 좌우에 추가할 패딩 계산 (이미지 중앙 정렬)
pad_top = (new_size - orig_h) // 2
pad_bottom = new_size - orig_h - pad_top
pad_left = (new_size - orig_w) // 2
pad_right = new_size - orig_w - pad_left

# numpy의 pad 함수 사용하여 정사각형으로 만들기 (세 번째 차원(RGBA)는 패딩하지 않음)
padded_bmp = np.pad(bmp_image, 
                    pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant', constant_values=0)

# 알파 채널 추출: padded_bmp의 4번째 채널, (new_size, new_size)
alpha_channel = padded_bmp[:, :, 3]

# 알파 채널을 기준으로 이진화 (투명 배경은 0, 객체는 1)
binary_image = (alpha_channel > 0).astype(np.float32)  # (new_size, new_size)

# 텐서로 변환 (device는 미리 지정한 device 변수 사용)
bmp_image_tensor = torch.tensor(binary_image, device=device)  # (new_size, new_size)

# 새로운 initializer method
print("---Initializing vector graphics from SVG---")
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
# initialize_from_svg()는 사용자가 정의해야 함 (아래 예시 참조)
x, y, r, v, theta = initializer.initialize_for_svg(I_target)
N = len(x)

# learnable 파라미터 v (scalar)와 c (3-vector) 초기화
v = v.clone().detach().requires_grad_(True)
c = torch.rand(N, 3, device=device, requires_grad=True)  # 색상, 초기 [0,1] 범위

# 픽셀 좌표 사전 계산
X, Y = torch.meshgrid(torch.arange(W, device=device),
                      torch.arange(H, device=device), indexing='xy')
X, Y = X.unsqueeze(0), Y.unsqueeze(0)  # (1, H, W)

alpha_upper_bound = config["optimization"].get("alpha_upper_bound", 0.5)


def soft_rasterize_shape(bmp_image, X, Y, x, y, r, theta):
    """
    bmp_image를 [-1,1]^2에서 정의된 continuous function으로 보고,
    이를 r 배 scale, theta 회전, (x,y) 이동한 뒤,
    (X,Y) 좌표에 따라 샘플링한 결과 mask를 생성.
    
    Args:
        bmp_image: (H_bmp, W_bmp) torch tensor, value in [0,1]
        X, Y: (1, H, W) 좌표 grid
        x, y, r, theta: 위치, 크기, 회전
    Returns:
        mask: (1, H, W)
    """
    _, H, W = X.shape

    # (1) bmp_image를 [-1, 1]^2 continuous function으로 보기 위해 4D로 reshape
    bmp_image = bmp_image.unsqueeze(0).unsqueeze(0)  # (1,1,H_bmp,W_bmp)

    # (2) (X,Y) 그리드 → (u,v)로 변환 (inverse transform)
    # 회전 행렬 (theta: 반시계 회전)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R_inv = torch.stack([
        torch.stack([cos_t, sin_t], dim=0),
        torch.stack([-sin_t, cos_t], dim=0)
    ], dim=0)  # (2, 2)

    # 출력 그리드 좌표 (1, H, W) → (H, W)
    X_flat = X.squeeze(0)
    Y_flat = Y.squeeze(0)

    # (X - x, Y - y): 위치 정규화
    pos = torch.stack([X_flat - x, Y_flat - y], dim=0)  # (2, H, W)
    pos = pos / r  # scale 보정

    # 회전 역변환
    uv = torch.einsum('ij,jhw->ihw', R_inv, pos)  # (2, H, W)

    # (3) uv ∈ [-1,1]^2를 (H, W, 2)로 reshape → grid_sample용
    u = uv[0]  # (H, W)
    v = uv[1]  # (H, W)
    grid = torch.stack([u, v], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0)  # (1, H, W, 2)

    # (4) 샘플링 (bilinear interpolation over [-1, 1])
    sampled = F.grid_sample(bmp_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (1,1,H,W)

    # (5) optional: 부드럽게 하고 싶으면 Gaussian blur 등을 추가해도 됨
    return sampled.squeeze(1)  # (1, H, W)


# Over Operator 함수 (기존과 동일)
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

# 마스크 캐싱
_cached_masks = []
for i in range(N):
    mask_i = soft_rasterize_shape(bmp_image_tensor, X, Y, x[i], y[i], r[i], theta[i])
    _cached_masks.append(mask_i)
_cached_masks = torch.cat(_cached_masks, dim=0)  # (N, H, W)

# 렌더링 함수 (기존과 동일)
def render_image_vector_cached(cached_masks, v, c, X, Y):
    N = v.shape[0]
    v_alpha = alpha_upper_bound * torch.sigmoid(v).view(N, 1, 1)
    a = v_alpha * cached_masks
    c_eff = torch.sigmoid(c).view(N, 1, 1, 3)
    m = a.unsqueeze(-1) * c_eff
    comp_m, comp_a = tree_over(m, a)
    background = torch.ones_like(comp_m)
    final = comp_m + (1 - comp_a).unsqueeze(-1) * background
    return final  # (H, W, 3)

# 학습 루프
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

# 시각화 함수 (기존과 동일)
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

import os
import xml.etree.ElementTree as ET
from copy import deepcopy

from cairosvg import svg2pdf

def remove_svg_styles(root):
    """
    주어진 SVG XML 트리의 모든 엘리먼트에서 style 관련 속성과
    <style> 태그를 제거합니다.
    """
    # 1. 재귀적으로 모든 엘리먼트에 대해 'style' 속성, 'stroke', 'fill', 
    #    'stroke-opacity', 'fill-opacity' 같은 속성을 삭제
    for elem in root.iter():
        # inline style attribute가 존재하면 삭제
        if 'style' in elem.attrib:
            del elem.attrib['style']
        # SVG 기본 스타일 속성도 삭제 (원하는 경우)
        for attr in ['stroke', 'fill', 'stroke-opacity', 'fill-opacity']:
            if attr in elem.attrib:
                del elem.attrib[attr]
    
    # 2. <style> 태그 자체를 제거
    # xml.etree.ElementTree는 getparent()가 없으므로,
    # 전체 자식 리스트에서 직접 찾아서 삭제
    remove_children = []
    for elem in root.iter():
        # 태그 이름에 namespace가 포함되어 있을 수 있으므로 endswith로 체크
        if elem.tag.endswith('style'):
            remove_children.append(elem)
    for child in remove_children:
        parent = _find_parent(root, child)
        if parent is not None:
            parent.remove(child)
            
def _find_parent(root, child):
    """
    xml.etree.ElementTree에는 getparent()가 없으므로, 수동으로 부모를 찾는 헬퍼 함수.
    """
    for elem in root.iter():
        for sub in elem:
            if sub == child:
                return elem
    return None

def export_vector_graphics_to_pdf(x, y, r, v, theta, c,
                                  pdf_filename="vector_art.pdf",
                                  new_size=None, svg_hollow=False):
    """
    원본 SVG 파일(전역변수 svg_file로 지정된 파일)을 불러와서,
    각 vector element에 대해 최종 PDF 캔버스 좌표에 맞게
    정규화 및 파라미터 변환 (x, y, r, theta)을 적용한 후
    모든 레이어를 누적하여 PDF로 내보낸다.
    
    입력 파라미터:
      paths: svgpathtools.svg2paths로 얻은 원본 SVG 경로 리스트 (바운딩 박스 중심 계산용)
      x, y, r, v, theta, c: 각 vector element에 대한 1D torch.Tensor (길이 N)
         - x, y: 변환 후의 위치 (canvas 좌표, 픽셀 단위)
         - r: 배율
         - theta: 회전각 (라디안)
         - v: 투명도 조절 파라미터 (sigmoid 후 alpha_upper_bound 배)
         - c: 색상 파라미터 (RGB; sigmoid 적용하여 0~1 값)
      pdf_filename: 출력 PDF 파일명
      canvas_size: PDF 캔버스 크기 (inch 단위; 여기서는 SVG의 viewBox/width/height와 연관)
      
    참고:
      - 전역변수 svg_file: 원본 SVG 파일 경로 (예: "images/apple.svg")
      - 전역변수 W, H: 캔버스 너비와 높이 (예: preprocessor.final_width, final_height)
      - 전역변수 alpha_upper_bound: 알파 값 상한
    """

    # (2) 원본 SVG XML 로드
    tree = ET.parse(svg_file)
    root = tree.getroot()
    remove_svg_styles(root)
    
    # (3) 정규화 scale 계산.
    norm_scale = 2 / new_size
    
    # (4) 최종 PDF 캔버스 크기 지정 (W, H는 전역에서 정의됨)
    root.attrib['width'] = str(W)
    root.attrib['height'] = str(H)
    root.attrib['viewBox'] = f"{-0.5*W} {-0.5*H} {W} {H}"
    
    # (5) 원본 SVG 콘텐츠(내부 자식 엘리먼트들)를 추출한 후 제거
    inner_elements = list(root)
    for elem in inner_elements:
        root.remove(elem)
    
    # (6) torch.Tensor들을 numpy array로 변환
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    r_np = r.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()  # 라디안 단위
    v_np = v.detach().cpu().numpy()
    c_np = torch.sigmoid(c).detach().cpu().numpy()  # shape (N, 3)
    alpha_vals = alpha_upper_bound * (1 / (1 + np.exp(-v_np)))
    
    N = len(x_np)
    # (7) 각 vector element마다, (역순으로) 원본 SVG 콘텐츠 복사본을 <g> 그룹으로 감싸고,
    #     새 transform 적용.
    # 최종 transform은 아래와 같이 구성:
    # T_total = translate(x - W/2, y - H/2)
    #           rotate(theta)
    #           scale(r)
    #           scale(norm_scale)
    #           translate({-orig_w/2},{-orig_h/2})
    for i in reversed(range(N)):
        theta_deg = np.degrees(theta_np[i])
        transform_str = (
            f"translate({x_np[i] - (W/2)},{y_np[i] - (H/2)}) "  # 이동: x,y를 기준점으로 재정렬
            f"rotate({theta_deg}) "                                # 회전
            f"scale({r_np[i]}) "                                  # vector별 스케일
            # f"translate({W/2},{H/2}) "                             # 캔버스 중앙으로 이동
            f"scale({norm_scale}) "                               # 원본 SVG 크기 정규화
            f"translate({-orig_w/2},{-orig_h/2})"         # 원본 SVG의 중심을 0,0으로
        )
        g = ET.Element("g")
        g.attrib["transform"] = transform_str
        # 색상 설정 (stroke), RGB는 [0,1] -> 정수 0~255 변환
        r_color, g_color, b_color = c_np[i]
        r_int = int(np.clip(r_color * 255, 0, 255))
        g_int = int(np.clip(g_color * 255, 0, 255))
        b_int = int(np.clip(b_color * 255, 0, 255))
        if svg_hollow == True:
            g.attrib["stroke"] = f"rgb({r_int},{g_int},{b_int})"
            g.attrib["stroke-opacity"] = str(alpha_vals[i])
            g.attrib["stroke-width"] = str(config["postprocessing"].get("linewidth", 3.0))
            g.attrib["fill"] = f"rgb({r_int},{g_int},{b_int})"
            g.attrib["fill-opacity"] = str(0.0)
        else:
            g.attrib["stroke"] = f"rgb({r_int},{g_int},{b_int})"
            g.attrib["stroke-opacity"] = str(alpha_vals[i])
            g.attrib["fill"] = f"rgb({r_int},{g_int},{b_int})"
            g.attrib["fill-opacity"] = str(alpha_vals[i])
        
        for elem in inner_elements:
            g.append(deepcopy(elem))
        root.append(g)
    
    # (8) 수정된 SVG XML을 임시 파일에 저장한 후, cairosvg로 PDF로 변환
    tmp_svg = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
    tmp_svg_name = tmp_svg.name
    tree.write(tmp_svg_name)
    tmp_svg.close()
    
    svg2pdf(url=tmp_svg_name, write_to=pdf_filename)
    os.remove(tmp_svg_name)

# PDF 내보내기 실행
output_pdf = config["postprocessing"].get("output_path", "vector_art.pdf")
export_vector_graphics_to_pdf(x, y, r, v, theta, c, 
                              pdf_filename=output_pdf, 
                              new_size=new_size, 
                              svg_hollow=config["svg"].get("svg_hollow", False))
import time
from datetime import timedelta
# 시작 시간 기록
start_time = time.time()

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

# Import our modules
from preprocessing import Preprocessor
from svgsplat_initialization import StructureAwareInitializer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser setup
parser = argparse.ArgumentParser(description="Process images with Structure-Aware Graphics Synthesis")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()
config_path = args.config

# Load configuration
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Initialize preprocessor
pp_conf = config["preprocessing"]
preprocessor = Preprocessor(
    final_width=pp_conf.get("final_width", 128),
    trim=pp_conf.get("trim", False),
    FM_halftone=pp_conf.get("FM_halftone", False),
    transform_mode=pp_conf.get("transform", "none"),
)

# Load target color image
I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
I_target = torch.tensor(I_target, device=device)  # (H, W, 3)
H = preprocessor.final_height
W = preprocessor.final_width

# Load SVG file
svg_file = config["svg"].get("svg_file", "images/tesla_logo.svg")
paths, attributes = svg2paths(svg_file)

# Convert SVG to high-quality bitmap
from cairosvg import svg2png

# Create temporary file
with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
    # Convert SVG to PNG
    svg2png(url=svg_file, write_to=temp_file.name, output_width=config["svg"].get("output_width", 128))
    
    # Read as RGBA image
    bmp_image = Image.open(temp_file.name).convert('RGBA')
    bmp_image = np.array(bmp_image)
    
# Get original image size
orig_h, orig_w, _ = bmp_image.shape
new_size = max(orig_h, orig_w)
old_size = min(orig_h, orig_w)

# Calculate padding for centering the image
pad_top = (new_size - orig_h) // 2
pad_bottom = new_size - orig_h - pad_top
pad_left = (new_size - orig_w) // 2
pad_right = new_size - orig_w - pad_left

# Pad the image to create a square
padded_bmp = np.pad(bmp_image, 
                    pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant', constant_values=0)

# Extract alpha channel
alpha_channel = padded_bmp[:, :, 3]

# Binarize using alpha channel (transparent background is 0, object is 1)
binary_image = (alpha_channel > 0).astype(np.float32)  # (new_size, new_size)

# Convert to tensor
bmp_image_tensor = torch.tensor(binary_image, device=device)  # (new_size, new_size)

# Initialize with Structure-Aware method
print("---Initializing vector graphics with Structure-Aware method---")
init_conf = config["initialization"]
initializer = StructureAwareInitializer(
    num_init=init_conf.get("N", 10000),
    alpha=init_conf.get("alpha", 0.3),  # Structure adjustment strength from config
    min_distance=init_conf.get("min_distance", 5),
    peak_threshold=init_conf.get("peak_threshold", 0.5),
    radii_min=init_conf.get("radii_min", 2),
    radii_max=init_conf.get("radii_max", None),
    v_init_mean=init_conf.get("v_init_mean", -5.0),
    keypoint_extracting=init_conf.get("keypoint_extracting", False)
)

# Initialize from SVG - all parameters are now learnable
x, y, r, v, theta = initializer.initialize_for_svg(I_target)
N = len(x)

# Convert to leaf tensors for optimization
x = x.detach().clone().requires_grad_(True)
y = y.detach().clone().requires_grad_(True)
r = r.detach().clone().requires_grad_(True)
v = v.detach().clone().requires_grad_(True)
theta = theta.detach().clone().requires_grad_(True)

x_init = x.clone().detach()
y_init = y.clone().detach()
r_init = r.clone().detach()
v_init = v.clone().detach()
theta_init = theta.clone().detach()

# Initialize learnable parameters c (3-vector color)
c = torch.rand(N, 3, device=device, requires_grad=True)  # Color, initially in [0,1] range

# Pre-compute pixel coordinates
X, Y = torch.meshgrid(torch.arange(W, device=device),
                      torch.arange(H, device=device), indexing='xy')
X, Y = X.unsqueeze(0), Y.unsqueeze(0)  # (1, H, W)

alpha_upper_bound = config["optimization"].get("alpha_upper_bound", 0.5)
delta      = config["optimization"].get("delta", 1.0)
kappa      = config["optimization"].get("kappa", 2000)   # 남길 splat 수
prune_itv  = config["optimization"].get("prune_interval", 20)
warmup     = config["optimization"].get("warmup_iter", 100)

z      = torch.zeros_like(v, requires_grad=False)      # auxiliary sparse var
lam    = torch.zeros_like(v, requires_grad=False)      # λ in paper

def batched_soft_rasterize(bmp_image, X, Y, x, y, r, theta):
    B = len(x)
    _, H, W = X.shape
    
    # Reshape X and Y to match batch dimension
    X_exp = X.expand(B, H, W)
    Y_exp = Y.expand(B, H, W)
    
    # Reshape x, y, r, theta to match grid dimensions
    x_exp = x.view(B, 1, 1).expand(B, H, W)
    y_exp = y.view(B, 1, 1).expand(B, H, W)
    r_exp = r.view(B, 1, 1).expand(B, H, W)
    
    # Calculate position
    pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
    
    # Calculate rotation for each batch element
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # Create rotation matrices for each batch element
    R_inv = torch.zeros(B, 2, 2, device=X.device)
    R_inv[:, 0, 0] = cos_t
    R_inv[:, 0, 1] = sin_t
    R_inv[:, 1, 0] = -sin_t
    R_inv[:, 1, 1] = cos_t
    
    # Apply rotation
    uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)
    
    # Reshape for grid_sample
    grid = uv.permute(0, 2, 3, 1)  # (B,H,W,2)
    
    # Expand bmp_image to match batch dimension
    bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Sample
    sampled = F.grid_sample(bmp_exp, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
    
    return sampled.squeeze(1)  # (B, H, W)

# Over Operator functions
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

# Rendering function
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

# Training loop
num_iterations = config["optimization"].get("num_iterations", 300)
learning_rate = config["optimization"].get("learning_rate", 0.1)

# Create optimizer for all learnable parameters
optimizer = torch.optim.Adam([
    {'params': x, 'lr': learning_rate*10},
    {'params': y, 'lr': learning_rate*10},
    {'params': r, 'lr': learning_rate},  # Smaller learning rate for radius
    {'params': v, 'lr': learning_rate},  # Smaller learning rate for visibility
    {'params': theta, 'lr': learning_rate},  # Smaller learning rate for rotation
    {'params': c, 'lr': learning_rate}  # Color parameters
])

print(f"Starting optimization for {num_iterations} iterations...")
for epoch in tqdm(range(num_iterations)):
    optimizer.zero_grad()
    
    # Recompute masks for current parameters
    _cached_masks = batched_soft_rasterize(bmp_image_tensor, X, Y, x, y, r, theta)

    I_hat = render_image_vector_cached(_cached_masks, v, c, X, Y)

    _alpha      = alpha_upper_bound * torch.sigmoid(v)
    loss   = F.mse_loss(I_hat, I_target)
    loss  += 0.5 * delta * F.mse_loss(_alpha, z - lam)  # ❋ 추가

    loss.backward()
    optimizer.step()
    
    # Clamp parameters to valid ranges
    with torch.no_grad():
        x.clamp_(0, W)
        y.clamp_(0, H)
        r.clamp_(init_conf.get("radii_min", 2), init_conf.get("radii_max", min(H, W) // 4))
        theta.clamp_(0, 2 * np.pi)
    
    if (epoch >= warmup and epoch % prune_itv == 0):
        with torch.no_grad():
            _alpha = alpha_upper_bound * torch.sigmoid(v.detach())
            # 중요도 = α 자체(혹은 |∂loss/∂α| 곱)  ↔ 실험 후 선택
            keep_idx = torch.topk(_alpha, kappa).indices
            mask     = torch.zeros_like(_alpha, dtype=torch.bool)
            mask[keep_idx] = True
            
            z.zero_(); z[mask] = _alpha[mask]        # z ← sparse projection
            lam += (_alpha - z)                      # λ ← λ + α - z

            # *** 실제 파라미터 수를 줄이고 싶다면 ***  
            if epoch % (10*prune_itv) == 0:     # 느슨한 빈도로 detach
                # ------------------------  pruning  ------------------------ #
                # 1) helper: 기존 tensor -> 잘라낸 tensor(requires_grad 유지)
                def _prune(t):
                    t_new = t[keep_idx].clone().detach()
                    if t.requires_grad:
                        t_new.requires_grad_(True)
                    return t_new

                # 2) 실제 파라미터·보조변수 잘라내기
                x, y, r, theta, v, c = map(_prune, (x, y, r, theta, v, c))
                z, lam               = map(_prune, (z, lam))

                # 3) optimizer 재생성 (Adam state 깔끔히 초기화)
                optimizer = torch.optim.Adam([
                    {'params': x, 'lr': learning_rate*10},
                    {'params': y, 'lr': learning_rate*10},
                    {'params': r, 'lr': learning_rate},
                    {'params': v, 'lr': learning_rate},
                    {'params': theta, 'lr': learning_rate},
                    {'params': c, 'lr': learning_rate},
                ])

                # 4) sparsity 목표를 조금 더 줄여 나가고 싶다면
                kappa = max(int(kappa * 0.8), 500)   # 예: 최저 500개까지


    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# RNMSE of x,y,r,v,theta
# print("RNMSE of x:", torch.norm(x - x_init) / torch.norm(x_init))
# print("RNMSE of y:", torch.norm(y - y_init) / torch.norm(y_init))
# print("RNMSE of r:", torch.norm(r - r_init) / torch.norm(r_init))
# print("RNMSE of v:", torch.norm(v - v_init) / torch.norm(v_init))
# print("RNMSE of theta:", torch.norm(theta - theta_init) / torch.norm(theta_init))

# 1) v 자체 분포
alpha_cpu = alpha_upper_bound * torch.sigmoid(v.detach()).cpu().numpy()          # (N,)
plt.figure(figsize=(6, 4))
plt.hist(alpha_cpu, bins=50)
plt.xlabel("alpha")
plt.ylabel("Count")
plt.title("Histogram of alpha after optimization, with sparsifying")
plt.tight_layout()
plt.show()



# Export vector graphics to PDF
def remove_svg_styles(root):
    """
    Remove style-related attributes and <style> tags from SVG XML tree.
    """
    # Remove style attributes from all elements
    for elem in root.iter():
        if 'style' in elem.attrib:
            del elem.attrib['style']
        for attr in ['stroke', 'fill', 'stroke-opacity', 'fill-opacity']:
            if attr in elem.attrib:
                del elem.attrib[attr]
    
    # Remove <style> tags
    remove_children = []
    for elem in root.iter():
        if elem.tag.endswith('style'):
            remove_children.append(elem)
    for child in remove_children:
        parent = _find_parent(root, child)
        if parent is not None:
            parent.remove(child)
            
def _find_parent(root, child):
    """
    Helper function to find parent in xml.etree.ElementTree.
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
    Load original SVG file, adjust vector elements to match final PDF canvas coordinates,
    and export as PDF.
    """
    import xml.etree.ElementTree as ET
    from copy import deepcopy
    from cairosvg import svg2pdf

    # Load original SVG XML
    tree = ET.parse(svg_file)
    root = tree.getroot()
    remove_svg_styles(root)
    
    # Calculate normalization scale
    norm_scale = 2 / new_size
    
    # Set final PDF canvas size
    root.attrib['width'] = str(W)
    root.attrib['height'] = str(H)
    root.attrib['viewBox'] = f"{-0.5*W} {-0.5*H} {W} {H}"
    
    # Extract and remove original SVG content
    inner_elements = list(root)
    for elem in inner_elements:
        root.remove(elem)
    
    # Convert torch.Tensor to numpy array
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    r_np = r.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()  # radians
    v_np = v.detach().cpu().numpy()
    c_np = torch.sigmoid(c).detach().cpu().numpy()  # shape (N, 3)
    alpha_vals = alpha_upper_bound * (1 / (1 + np.exp(-v_np)))
    
    N = len(x_np)
    # For each vector element (in reverse order), copy the original SVG content,
    # wrap in a <g> group, and apply the new transform
    for i in reversed(range(N)):
        theta_deg = np.degrees(theta_np[i])
        transform_str = (
            f"translate({x_np[i] - (W/2)},{y_np[i] - (H/2)}) "  # Move to x,y as reference point
            f"rotate({theta_deg}) "                             # Rotate
            f"scale({r_np[i]}) "                                # Scale per vector
            f"scale({norm_scale}) "                             # Normalize original SVG size
            f"translate({-orig_w/2},{-orig_h/2})"               # Center original SVG at 0,0
        )
        g = ET.Element("g")
        g.attrib["transform"] = transform_str
        # Set color (stroke), convert RGB from [0,1] to integers 0-255
        r_color, g_color, b_color = c_np[i]
        r_int = int(np.clip(r_color * 255, 0, 255))
        g_int = int(np.clip(g_color * 255, 0, 255))
        b_int = int(np.clip(b_color * 255, 0, 255))
        if svg_hollow:
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
    
    # Save modified SVG XML to temporary file, then convert to PDF
    tmp_svg = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
    tmp_svg_name = tmp_svg.name
    tree.write(tmp_svg_name)
    tmp_svg.close()
    
    svg2pdf(url=tmp_svg_name, write_to=pdf_filename)
    os.remove(tmp_svg_name)
    print(f"Exported vector graphics to {pdf_filename}")

# Export PDF
output_pdf = config["postprocessing"].get("output_path", "vector_art.pdf")
export_vector_graphics_to_pdf(x, y, r, v, theta, c, 
                              pdf_filename=output_pdf, 
                              new_size=new_size, 
                              svg_hollow=config["svg"].get("svg_hollow", False))

print("Done!") 

end_time = time.time()
formatted_time = str(timedelta(seconds=int(end_time - start_time)))
# 수행 시간 출력
print(f"total_cost_time: {formatted_time}")
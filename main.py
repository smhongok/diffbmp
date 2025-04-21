import time
from datetime import timedelta
# 시작 시간 기록
start_time = time.time()

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import json
import argparse
import cv2
from datetime import datetime

# Import our modules
from preprocessing import Preprocessor
from svgsplat_initialization import StructureAwareInitializer
from utils import set_global_seed, gaussian_blur, compute_psnr
from svg_loader import SVGLoader
from pdf_exporter import PDFExporter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser setup
parser = argparse.ArgumentParser(description="Process images with Structure-Aware Graphics Synthesis")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()
config_path = args.config

# Load configuration
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# import 뒤 혹은 config 로드 직후
set_global_seed(config.get("seed", 42))

# Initialize preprocessor
pp_conf = config["preprocessing"]
opt_conf = config["optimization"]

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
svg_loader = SVGLoader(
    svg_path=config["svg"].get("svg_file", "images/tesla_logo.svg"),
    output_width=config["svg"].get("output_width", 128),
    device=device
)

bmp_image_tensor = svg_loader.load_alpha_bitmap()

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
    keypoint_extracting=init_conf.get("keypoint_extracting", False),
    debug_mode=init_conf.get("debug_mode", False)  # Add debug mode parameter
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

# Initialize learnable parameters c (3-vector color)
c = torch.rand(N, 3, device=device, requires_grad=True)  # Color, initially in [0,1] range

# Pre-compute pixel coordinates
X, Y = torch.meshgrid(torch.arange(W, device=device),
                      torch.arange(H, device=device), indexing='xy')
X, Y = X.unsqueeze(0), Y.unsqueeze(0)  # (1, H, W)

alpha_upper_bound = opt_conf.get("alpha_upper_bound", 0.5)

def batched_soft_rasterize(bmp_image, X, Y, x, y, r, theta, sigma=0.0):
    """
    bmp_image: (H_bmp, W_bmp) tensor in [0,1]
    X, Y: (1, H, W) coordinate grids
    x, y, r, theta: each is tensor of shape (B,)
    sigma: Gaussian blur standard deviation for softening the mask
    Returns:
        (B, H, W) soft masks
    """
    B = len(x)
    _, H, W = X.shape

    # 1) 입력 bmp_image에 Gaussian blur 적용 (soft rasterization 효과)
    if sigma > 0.0:
        # bmp_image: (H_bmp, W_bmp) -> (1,H_bmp,W_bmp)
        bmp = bmp_image.unsqueeze(0)
        # gaussian_blur: (N, H, W) 형태를 기대하므로 N=1
        bmp = gaussian_blur(bmp, sigma)  # (N, H_bmp, W_bmp)
        bmp_image = bmp.squeeze(0)       # (H_bmp, W_bmp)

    # 2) 그리드와 파라미터를 배치 크기에 맞춰 확장
    X_exp = X.expand(B, H, W)
    Y_exp = Y.expand(B, H, W)
    x_exp = x.view(B,1,1).expand(B, H, W)
    y_exp = y.view(B,1,1).expand(B, H, W)
    r_exp = r.view(B,1,1).expand(B, H, W)

    # 3) 위치 정규화 및 회전
    pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R_inv = torch.zeros(B, 2, 2, device=X.device)
    R_inv[:,0,0] = cos_t; R_inv[:,0,1] = sin_t
    R_inv[:,1,0] = -sin_t; R_inv[:,1,1] = cos_t
    uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)

    # 4) grid_sample을 위한 형태 조정
    grid = uv.permute(0,2,3,1)  # (B, H, W, 2)
    bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,1,H_bmp,W_bmp)

    # 5) bilinear sampling
    sampled = F.grid_sample(bmp_exp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

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
num_iterations = opt_conf.get("num_iterations", 300)
learning_rate = opt_conf.get("learning_rate", 0.1)

# sigma 스케줄: config 에서 초기와 최종 값 가져옴 (기본값: 2.0 -> 1.0)
do_gaussian_blur = opt_conf.get("do_gaussian_blur", False)
sigma_start = opt_conf.get("blur_sigma_start", 2.0) # 
sigma_end = opt_conf.get("blur_sigma_end", 1.0)

# Create optimizer for all learnable parameters
lr_conf = opt_conf["learning_rate"]
lr = lr_conf.get("default", 0.1)
do_decay = opt_conf.get("do_decay", False)
optimizer_xyr = torch.optim.Adam([
    {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
    {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
    {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},  
])

optimizer_rest = torch.optim.Adam([
    {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},  
    {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
    {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)}, 
])
sched_xyr = torch.optim.lr_scheduler.ExponentialLR(
    optimizer_xyr, gamma=opt_conf.get("decay_rate", 0.99)) if do_decay else None


sparsify_conf = opt_conf["sparsifying"]
do_sparsify = sparsify_conf.get("do_sparsify", False)

if do_sparsify:
    z = torch.zeros_like(v, requires_grad=False)
    lam = torch.zeros_like(v, requires_grad=False)
    iters_warmup = sparsify_conf.get("sparsify_warmup", num_iterations//3)
    sparsify_duration = sparsify_conf.get("sparsify_duration", num_iterations//3)
    sparsify_loss_coeff = sparsify_conf.get("sparsify_loss_coeff", 0.5)
    sparsifying_period = sparsify_conf.get("sparsifying_period", 20)
    sparsified_N = int(sparsify_conf.get("sparsified_N", 0.6 * N))
    assert sparsified_N < N, "sparsified_N must be less than N"
    assert num_iterations > iters_warmup + sparsify_duration, "num_iterations must be greater than warmup + duration"

print(f"Starting optimization for {num_iterations} iterations...")
for epoch in tqdm(range(num_iterations)):
    optimizer_xyr.zero_grad()
    optimizer_rest.zero_grad()

    # 현재 epoch에 따른 sigma (선형 보간)
    sigma = sigma_start * (1 - epoch / num_iterations) + sigma_end * (epoch / num_iterations) if do_gaussian_blur else 0.0
    
    # Recompute masks for current parameters
    _cached_masks = batched_soft_rasterize(bmp_image_tensor, X, Y, x, y, r, theta, sigma=sigma)

    I_hat = render_image_vector_cached(_cached_masks, v, c, X, Y)
    loss = F.mse_loss(I_hat, I_target)
    if do_sparsify and epoch >= iters_warmup and epoch < iters_warmup+sparsify_duration:
        _alpha      = alpha_upper_bound * torch.sigmoid(v)
        loss  += 0.5 * sparsify_loss_coeff * F.mse_loss(_alpha, z - lam) # Sparsification loss
    loss.backward()

    # step the optimizers
    optimizer_xyr.step()
    optimizer_rest.step()
    if do_decay and sched_xyr is not None:
        sched_xyr.step()
    
    # Clamp parameters to valid ranges
    with torch.no_grad():
        x.clamp_(0, W)
        y.clamp_(0, H)
        r.clamp_(init_conf.get("radii_min", 2), init_conf.get("radii_max", min(H, W) // 4))
        theta.clamp_(0, 2 * np.pi)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    if do_sparsify:
        with torch.no_grad():
            if (epoch>= iters_warmup and epoch <= iters_warmup+sparsify_duration):
                if epoch % sparsifying_period == 0:
                    # Sparsification step
                    _alpha = alpha_upper_bound * torch.sigmoid(v.detach())
                    keep_idx = torch.topk(_alpha, sparsified_N).indices
                    mask     = torch.zeros_like(_alpha, dtype=torch.bool)
                    mask[keep_idx] = True
                    
                    z.zero_(); z[mask] = _alpha[mask]        # z ← sparse projection
                    lam += (_alpha - z)             
                
                if epoch == iters_warmup+sparsify_duration:
                    # actual sparsification
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
                    optimizer_xyr = torch.optim.Adam([
                        {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
                        {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
                        {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},  
                    ])

                    optimizer_rest = torch.optim.Adam([
                        {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},  
                        {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
                        {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)}, 
                    ])
                    sched_xyr = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer_xyr, gamma=opt_conf.get("decay_rate", 0.99)) if do_decay else None

# Save the final rendered image
final_render = render_image_vector_cached(_cached_masks, v, c, X, Y)
final_render_np = final_render.detach().cpu().numpy()
final_render_np = (final_render_np * 255).astype(np.uint8)

# Create combined visualization with point debug if debug mode was enabled
if init_conf.get("debug_mode", False):
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Save the final render
    cv2.imwrite('outputs/final_render.png', cv2.cvtColor(final_render_np, cv2.COLOR_RGB2BGR))
    
    # Load point debug visualization
    point_debug = cv2.imread('outputs/point_debug.png')
    
    # Resize if dimensions don't match
    if point_debug.shape[:2] != final_render_np.shape[:2]:
        point_debug = cv2.resize(point_debug, (final_render_np.shape[1], final_render_np.shape[0]))
    
    # Load original visualization if it exists
    if os.path.exists('outputs/side_by_side_debug.png'):
        side_by_side = cv2.imread('outputs/side_by_side_debug.png')
        
        # Create a new row with final render
        final_render_bgr = cv2.cvtColor(final_render_np, cv2.COLOR_RGB2BGR)
        
        # Extract original and point debug from side_by_side
        mid_point = side_by_side.shape[1] // 2
        original = side_by_side[:, :mid_point]
        points = side_by_side[:, mid_point:]
        
        # Resize final render to match original and points
        final_render_bgr = cv2.resize(final_render_bgr, (mid_point, side_by_side.shape[0]))
        
        # Create three-panel image: original, points, final render
        combined = np.hstack((original, points, final_render_bgr))
        
        timestamp_str = datetime.now().strftime("%m-%d-%H-%M-%S")
        print(timestamp_str)
        cv2.imwrite('outputs/combined_visualization_' + timestamp_str + '.png', combined)
        print("Combined visualization saved to outputs/combined_visualization.png")

# Save the final rendered image
final_render = render_image_vector_cached(_cached_masks, v, c, X, Y)
final_render_np = final_render.detach().cpu().numpy()
final_render_np = (final_render_np * 255).astype(np.uint8)

# Create combined visualization with point debug if debug mode was enabled
if init_conf.get("debug_mode", False):
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Save the final render
    cv2.imwrite('outputs/final_render.png', cv2.cvtColor(final_render_np, cv2.COLOR_RGB2BGR))
    
    # Load point debug visualization
    point_debug = cv2.imread('outputs/point_debug.png')
    
    # Resize if dimensions don't match
    if point_debug.shape[:2] != final_render_np.shape[:2]:
        point_debug = cv2.resize(point_debug, (final_render_np.shape[1], final_render_np.shape[0]))
    
    # Load original visualization if it exists
    if os.path.exists('outputs/side_by_side_debug.png'):
        side_by_side = cv2.imread('outputs/side_by_side_debug.png')
        
        # Create a new row with final render
        final_render_bgr = cv2.cvtColor(final_render_np, cv2.COLOR_RGB2BGR)
        
        # Extract original and point debug from side_by_side
        mid_point = side_by_side.shape[1] // 2
        original = side_by_side[:, :mid_point]
        points = side_by_side[:, mid_point:]
        
        # Resize final render to match original and points
        final_render_bgr = cv2.resize(final_render_bgr, (mid_point, side_by_side.shape[0]))
        
        # Create three-panel image: original, points, final render
        combined = np.hstack((original, points, final_render_bgr))
        
        timestamp_str = datetime.now().strftime("%m-%d-%H-%M-%S")
        print(timestamp_str)
        cv2.imwrite('outputs/combined_visualization_' + timestamp_str + '.png', combined)
        print("Combined visualization saved to outputs/combined_visualization.png")

exporter = PDFExporter(svg_loader.svg_path, canvas_size=(W, H), viewbox_size=(svg_loader.get_svg_size()),
                       alpha_upper_bound=alpha_upper_bound, stroke_width=config["postprocessing"].get("linewidth", 3.0))

exporter.export(x, y, r, theta, v, c,
                output_path=config['postprocessing']['output_path'], 
                svg_hollow=config['svg'].get('svg_hollow',False))

end_time = time.time()
formatted_time = str(timedelta(seconds=int(end_time - start_time)))
# 수행 시간 출력
print(f"total_cost_time: {formatted_time}")

do_compute_psnr = config['postprocessing'].get('compute_psnr', False)

# compute PSNR, SSIM, LPIPS between exported PDF and target image
if do_compute_psnr:
    try:
        from pdf2image import convert_from_path
        import piq
        # Convert first page of PDF to image at same resolution
        pages = convert_from_path(config['postprocessing']['output_path'], dpi=300)
        export_img_pil = pages[0].resize((W, H))
        export_arr = np.array(export_img_pil).astype(np.float32) / 255.0
        # If RGBA, drop alpha
        if export_arr.shape[2] == 4:
            export_arr = export_arr[..., :3]
        export_tensor = torch.tensor(export_arr, device=device)
        # reshape to (1,3,H,W)
        out = export_tensor.permute(2,0,1).unsqueeze(0)
        tgt = I_target.permute(2,0,1).unsqueeze(0)
        # Compute metrics
        psnr_val = piq.psnr(out, tgt, data_range=1.0)
        ssim_val = piq.ssim(out, tgt, data_range=1.0)
        vif_val = piq.vif_p(out, tgt, data_range=1.0)
        lpips_val = piq.LPIPS()(out, tgt)
        print(f"PSNR: {psnr_val.item():.2f} dB")
        print(f"SSIM: {ssim_val.item():.4f}")
        print(f"VIF: {vif_val.item():.4f}")
        print(f"LPIPS: {lpips_val.item():.4f}")
        print(f"Number of splats: {len(x)}")
    except ImportError as e:
        print(f"Required library missing: {e}. Cannot compute metrics.")


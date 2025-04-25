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
from core.renderer.mse_renderer import MseRenderer
from core.renderer.lpips_renderer import LpipsRenderer
from core.renderer.mix_renderer import MixRenderer
from util.svg_loader import SVGLoader
from util.font_to_svg import FontParser
from core.initializer.svgsplat_initializater import StructureAwareInitializer
from core.initializer.random_initializater import RandomInitializer

# Import our modules
from preprocessing import Preprocessor
from util.utils import set_global_seed, gaussian_blur, compute_psnr
from util.pdf_exporter import PDFExporter

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

# Handle SVG file loading
svg_ext = os.path.splitext(config["svg"].get("svg_file"))[1].lower()
if (svg_ext in (".otf", ".ttf")) and ("text" in config["svg"]):
    font_parser = FontParser(config["svg"].get("svg_file"))
    svg_path = str(font_parser.text_to_svg(config["svg"].get("text"), mode="opt-path"))
else:
    svg_path = config["svg"].get("svg_file", "assets/svg/MaruBuri-Bold_HELLO.svg")

# Load SVG file
svg_loader = SVGLoader(
    svg_path=svg_path,
    output_width=config["svg"].get("output_width", 128),
    device=device
)

# Initialize renderer based on loss type
renderer_type = opt_conf.get("renderer_type", "mse")
renderer_class = {
    "mse": MseRenderer,
    "lpips": LpipsRenderer,
    "mix": MixRenderer
}.get(renderer_type.lower())

if renderer_class is None:
    raise ValueError(f"Invalid renderer type: {renderer_type}")

renderer = renderer_class(
    canvas_size=(H, W),
    alpha_upper_bound=opt_conf.get("alpha_upper_bound", 0.5),
    device=device
)

print(f"Using {renderer_class.__name__} for optimization")

# Initialize parameters
print("---Initializing vector graphics with Structure-Aware method---")
init_conf = config["initialization"]
if init_conf.get("initializer", "none") == "structure_aware":
    initializer = StructureAwareInitializer(
        num_init=init_conf.get("N", 10000),
        alpha=init_conf.get("alpha", 0.3),
        min_distance=init_conf.get("min_distance", 5),
        peak_threshold=init_conf.get("peak_threshold", 0.5),
        radii_min=init_conf.get("radii_min", 2),
        radii_max=init_conf.get("radii_max", None),
        v_init_bias=init_conf.get("v_init_bias", -5.0),
        v_init_slope=init_conf.get("v_init_slope", 10.0),
        keypoint_extracting=init_conf.get("keypoint_extracting", False),
        debug_mode=init_conf.get("debug_mode", False)
    )
elif init_conf.get("initializer", "none") == "random":
    initializer = RandomInitializer(
        num_init=init_conf.get("N", 10000),
        alpha=init_conf.get("alpha", 0.3),
        min_distance=init_conf.get("min_distance", 5),
        peak_threshold=init_conf.get("peak_threshold", 0.5),
        radii_min=init_conf.get("radii_min", 2),
        radii_max=init_conf.get("radii_max", None),
        v_init_bias=init_conf.get("v_init_bias", -5.0),
        v_init_slope=init_conf.get("v_init_slope", 10.0),
        keypoint_extracting=init_conf.get("keypoint_extracting", False),
        debug_mode=init_conf.get("debug_mode", False)
    )
else:
    raise ValueError(f"Invalid initializer: {init_conf.get('initializer', 'none')}")

# Initialize parameters
x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target)

bmp_image_tensor = svg_loader.load_alpha_bitmap()

# Optimize parameters
x, y, r, v, theta, c = renderer.optimize_parameters(
    x, y, r, v, theta, c,
    I_target, bmp_image_tensor,
    opt_conf
)

# Generate final masks and render
cached_masks = renderer._batched_soft_rasterize(
    bmp_image_tensor, x, y, r, theta,
    sigma=opt_conf.get("blur_sigma_end", 1.0)
)

# Save the final rendered image
output_path = config["postprocessing"].get("output_folder", "./outputs/")
output_path = output_path + "output.png"
renderer.save_rendered_image(cached_masks, v, c, output_path)

# Create combined visualization with point debug if debug mode was enabled
if init_conf.get("debug_mode", False):
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Save the final render
    cv2.imwrite('outputs/final_render.png', cv2.cvtColor(cached_masks.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
    
    # Load point debug visualization
    point_debug = cv2.imread('outputs/point_debug.png')
    
    # Resize if dimensions don't match
    if point_debug.shape[:2] != cached_masks.shape[:2]:
        point_debug = cv2.resize(point_debug, (cached_masks.shape[1], cached_masks.shape[0]))
    
    # Load original visualization if it exists
    if os.path.exists('outputs/side_by_side_debug.png'):
        side_by_side = cv2.imread('outputs/side_by_side_debug.png')
        
        # Create a new row with final render
        final_render_bgr = cv2.cvtColor(cached_masks.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR)
        
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

filename_only = os.path.splitext(os.path.basename(config['preprocessing']['img_path']))[0]
output_path=config['postprocessing']['output_folder'] + filename_only + "_N" + str(init_conf.get("N", 1000)) + "_ITER" + str(config["optimization"]["num_iterations"]) \
    + "_" + str(config['optimization']['sparsifying']['do_sparsify'])[0] + "_SPN" + str(config['optimization']['sparsifying']['sparsified_N']) \
    + "_" + str(config['initialization']['initializer'])[0:2] + "_" + str(config['optimization']['renderer_type']) + ".pdf"
config['postprocessing']['output_path'] = output_path

exporter = PDFExporter(svg_loader.svg_path, canvas_size=(W, H), viewbox_size=(svg_loader.get_svg_size()),
                       alpha_upper_bound=config["optimization"]["alpha_upper_bound"], stroke_width=config["postprocessing"].get("linewidth", 3.0))

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


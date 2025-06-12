import time
from datetime import timedelta
# Record the start time
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
from util.svg_converter import FontParser
from core.initializer.svgsplat_initializater import StructureAwareInitializer
from core.initializer.random_initializater import RandomInitializer

# Import our modules
from core.preprocessing import Preprocessor
from util.utils import set_global_seed, gaussian_blur, compute_psnr
from util.pdf_exporter import PDFExporter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser setup
parser = argparse.ArgumentParser(description="Process images with Structure-Aware Graphics Synthesis")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
parser.add_argument('--initializer', type=str, default='StructureAwareInitializer', help='StructureAwareInitializer, RandomInitializer, MultiLevelInitializer, ...')
parser.add_argument('--renderer', type=str, default='MseRenderer', help='MseRenderer, ...')
parser.add_argument('--svg_text', type=str, default='', help='G, B, M, ...')
parser.add_argument('--svg_path', type=str, default='', help='LOVE.svg, ...')
parser.add_argument('--img_path', type=str, default='', help='images/HighFreq/0831.png, images/LowFreq/0831.png, ...')
args = parser.parse_args()

# Load configuration
with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)

# Set random seed
set_global_seed(config.get("seed", 42))

# Initialize preprocessor
pp_conf = config["preprocessing"]
preprocessor = Preprocessor(
    final_width=pp_conf.get("final_width", 128),
    trim=pp_conf.get("trim", False),
    FM_halftone=pp_conf.get("FM_halftone", False),
    transform_mode=pp_conf.get("transform", "none"),
)

# Handle SVG file loading
svg_ext = os.path.splitext(config["svg"].get("svg_file"))[1].lower()
if svg_ext == ".svg":
    svg_path = os.path.join("assets/svg", config["svg"].get("svg_file"))
elif svg_ext in (".png", ".jpg", ".jpeg"):
    img_converter = ImageToSVG()
    svg_path = img_converter.extract_filled_outlines(config["svg"].get("svg_file"), threshold=100, min_area_ratio=0.000001)
    del img_converter
elif (svg_ext in (".otf", ".ttf")) and ("text" in config["svg"]):
    font_parser = FontParser(config["svg"]["svg_file"])
    texts = config["svg"]["text"]
    if isinstance(texts, list):
        svg_paths = [str(font_parser.text_to_svg(t, mode="opt-path")) for t in texts]
    else:
        svg_paths = str(font_parser.text_to_svg(texts, mode="opt-path"))
    svg_path = svg_paths
    del font_parser
else:
    svg_path = config["svg"].get("svg_file", "assets/svg/MaruBuri-Bold_HELLO.svg")

print(f"SVG path: {svg_path}")

# Load SVG file
svg_loader = SVGLoader(
    svg_path=svg_path,
    output_width=config["svg"].get("output_width", 128),
    device=device
)
classify_svg = svg_loader.classify_svg()
print(f"SVG is classified as: {classify_svg}")

# Load and preprocess target image
I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
I_target = torch.tensor(I_target, device=device)
H = preprocessor.final_height
W = preprocessor.final_width
config['canvas_size'] = (H, W)

# Create initializer
if args.initializer == "StructureAwareInitializer":
    from core.initializer.svgsplat_initializater import StructureAwareInitializer
    initializer = StructureAwareInitializer(config["initialization"])
elif args.initializer == "RandomInitializer":
    from core.initializer.random_initializater import RandomInitializer
    initializer = RandomInitializer(config["initialization"])
elif args.initializer == "MultiLevelInitializer":
    from core.initializer.multilevel_initializer import MultiLevelInitializer
    initializer = MultiLevelInitializer(config["initialization"])
else:
    raise ValueError(f"Unknown initializer: {args.initializer}")

# Create renderer
if args.renderer == "MseRenderer":
    from core.renderer.mse_renderer import MseRenderer
    renderer = MseRenderer((H, W), S=svg_loader.load_alpha_bitmap(), 
                        alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                        device=device,
                        use_fp16=config["optimization"].get("use_fp16", False),
                        output_path=config["postprocessing"].get("output_folder", "./outputs/"))
else:
    raise ValueError(f"Unknown renderer: {args.renderer}")

# Generate initialization parameters
if "LevelInitializer" in args.initializer:
    params = initializer.initialize(I_tar=I_target, renderer=renderer, opt_conf=config["optimization"])
else:
    params = initializer.initialize(I_target)

x, y, r, v, theta, c = params

# Optimize parameters
x, y, r, v, theta, c = renderer.optimize_parameters(
    x, y, r, v, theta, c,
    I_target, 
    opt_conf=config["optimization"]
)

# Generate final render
cached_masks = renderer._batched_soft_rasterize(
    x, y, r, theta,
    sigma=config["optimization"].get("blur_sigma_end", 1.0)
)
rendered = renderer.render(cached_masks, v, c)

# Save the final rendered image
output_dir = config["postprocessing"].get("output_folder", "./outputs/")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(output_dir, f'output_{timestamp}.png')
renderer.save_rendered_image(cached_masks, v, c, output_path)

# Export PDF
pdf_path = os.path.join(output_dir, f'output_{timestamp}.pdf')
exporter = PDFExporter(
    svg_loader.svg_path, 
    canvas_size=(W, H),
    viewbox_size=svg_loader.get_svg_size(),
    alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5),
    stroke_width=config["postprocessing"].get("linewidth", 3.0)
)

exporter.export(x, y, r, theta, v, c,
                output_path=pdf_path,
                svg_hollow=config["svg"].get("svg_hollow", False))

# Compute metrics if requested
if config['postprocessing'].get('compute_psnr', False):
    try:
        import piq
        # Convert rendered image to tensor format for metrics
        rendered_t = rendered.permute(2, 0, 1).unsqueeze(0)
        target_t = I_target.permute(2, 0, 1).unsqueeze(0)
        
        # Compute metrics
        psnr_val = piq.psnr(rendered_t, target_t, data_range=1.0)
        ssim_val = piq.ssim(rendered_t, target_t, data_range=1.0)
        vif_val = piq.vif_p(rendered_t, target_t, data_range=1.0)
        lpips_val = piq.LPIPS()(rendered_t, target_t)
        
        print(f"PSNR: {psnr_val.item():.2f} dB")
        print(f"SSIM: {ssim_val.item():.4f}")
        print(f"VIF: {vif_val.item():.4f}")
        print(f"LPIPS: {lpips_val.item():.4f}")
        print(f"Number of splats: {len(x)}")
    except ImportError as e:
        print(f"Required library missing: {e}. Cannot compute metrics.")

end_time = time.time()
formatted_time = str(timedelta(seconds=int(end_time - start_time)))
print(f"total_cost_time: {formatted_time}")

if "extra_postprocessing" in config:
    # ------------------------------------------------------------------
    # Extra post-processing : Figure-1 (PDF)  ─ original | export | dropout
    # ------------------------------------------------------------------
    if config.get("extra_postprocessing", {}).get("make_fig1_pdf", False):

        #right_pdf = 'outputs/_fig1_right.pdf'
        if pdf_path.endswith(".pdf"):
            extra_output_path = pdf_path.replace(".pdf", "_extra.pdf")
        else:
            raise("Output path must end with .pdf")
            extra_output_path = pdf_path + "_extra.pdf"

        exporter.export_dropout_right_third(x, y, r, theta, v, c,
                        output_path=extra_output_path,
                        svg_hollow=config['svg'].get('svg_hollow', False))


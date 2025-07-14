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
from util.svg_loader import SVGLoader
from util.svg_converter import FontParser, ImageToSVG
from core.initializer.svgsplat_initializater import StructureAwareInitializer
from core.initializer.random_initializater import RandomInitializer

# Import our modules
from core.preprocessing import Preprocessor
from util.utils import set_global_seed, gaussian_blur, compute_psnr, extract_chars_from_file
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
# After import or after loading config
set_global_seed(config.get("seed", 42))

# Force list attributes to single item
if type(config["preprocessing"]["img_path"]) is list:
    config["preprocessing"]["img_path"] = config["preprocessing"]["img_path"][0]
    print("Use only one file to inference")
if type(config["preprocessing"]["final_width"]) is list:
    config["preprocessing"]["final_width"] = config["preprocessing"]["final_width"][0]
    print("Use only one final_width to inference")
if type(config["initialization"]["N"]) is list: 
    config["initialization"]["N"] = config["initialization"]["N"][0]
    print("Use only one N to inference")

# Initialize preprocessor
pp_conf = config["preprocessing"]
opt_conf = config["optimization"]
use_fp16 = opt_conf.get("use_fp16", False)  # Default to False for CPU compatibility

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
if svg_ext == ".svg":
    svg_path = os.path.join("assets/svg", config["svg"].get("svg_file"))
elif svg_ext in (".png", ".jpg", ".jpeg"):
    img_converter = ImageToSVG()
    svg_path = img_converter.extract_filled_outlines(config["svg"].get("svg_file"), threshold=100, min_area_ratio=0.000001)
    del img_converter
elif svg_ext in (".otf", ".ttf"):
    # 텍스트 소스 결정
    texts = None
    if "text" in config["svg"]:
        texts = config["svg"]["text"]
    elif "text_file" in config["svg"]:
        # 파일에서 텍스트 추출 (예시: txt, lrc 등 처리)
        text_ext = os.path.splitext(config["svg"].get("text_file"))[1].lower()
        text_path = os.path.join("assets/texts", config["svg"].get("text_file")) 
        
        # 전용 파서 클래스 (여기서는 간단 예시)
        if text_ext == ".txt" or text_ext == ".lrc":
            texts, char_counts, word_lengths_per_line = extract_chars_from_file(text_path, text_ext, remove_punct=config["svg"].get("remove_punctuation", False), punct_to_remove=".,;:()\{\}[]\"\'")
            html_extra_path_special = "output_webpage/src_lyrics/index.html"
            config["initialization"]["N"] = sum(char_counts)  # N은 텍스트의 개수로 설정
        else:
            raise ValueError(f"Unsupported text_file type: {text_ext}")

    if texts is not None:
        font_parser = FontParser(config["svg"]["svg_file"])
        # texts = config["svg"]["text"]
        if isinstance(texts, list):
            svg_paths = [str(font_parser.text_to_svg(t, mode="opt-path")) for t in texts]
        else:
            svg_paths = str(font_parser.text_to_svg(texts, mode="opt-path"))
        svg_path = svg_paths
        del font_parser
    else:
        raise ValueError("No text source ('text' or 'text_file') provided in svg config.")

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
}.get(renderer_type.lower())

if renderer_class is None:
    raise ValueError(f"Invalid renderer type: {renderer_type}")

bmp_tensor = svg_loader.load_alpha_bitmap()
if use_fp16:
    bmp_tensor = bmp_tensor.to(dtype=torch.float16)
else:
    bmp_tensor = bmp_tensor.to(dtype=torch.float32)
    
H = preprocessor.final_height
W = preprocessor.final_width

# Create renderer only when needed - defer instantiation
renderer = renderer_class((H, W), S=bmp_tensor, 
                        alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                        device=device,
                        use_fp16=use_fp16,
                        output_path=config["postprocessing"].get("output_folder", "./outputs/"))
print(f"Using {renderer_class.__name__} for optimization")

# Initialize parameters
print("---Initializing vector graphics with Structure-Aware method---")
init_conf = config["initialization"]
if init_conf.get("initializer", "none") == "structure_aware":
    initializer = StructureAwareInitializer(init_conf)
elif init_conf.get("initializer", "none") == "random":
    initializer = RandomInitializer(init_conf)
else:
    raise ValueError(f"Invalid initializer: {init_conf.get('initializer', 'none')}")

# Initialize parameters
x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target)

bmp_image_tensor = svg_loader.load_alpha_bitmap()

# Optimize parameters
x, y, r, v, theta, c = renderer.optimize_parameters(
    x, y, r, v, theta, c,
    I_target, 
    opt_conf=opt_conf
)

# Generate final masks and render
cached_masks = renderer._batched_soft_rasterize(
    x, y, r, theta,
    sigma=0
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
                svg_hollow=config["svg"].get("svg_hollow", False),
                html_extra_path = "output_webpage/src/index.html" if html_extra_path_special is None else html_extra_path_special,
                export_pdf=True,
                html_extra_meta={"char_counts": json.dumps(char_counts), "word_lengths_per_line": json.dumps(word_lengths_per_line)} if 'char_counts' in locals() else {}
)
with torch.no_grad():
    video_path = os.path.join(output_dir, f'output_{timestamp}.mp4')
    renderer.render_export_mp4(cached_masks, v, c, video_path=video_path)

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

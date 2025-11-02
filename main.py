import time
from datetime import timedelta
# Record the start time
start_time = time.time()

import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import json
import argparse
import cv2
from datetime import datetime
from imbrush.core.renderer.sequential_renderer import SequentialFrameRenderer
from imbrush.core.renderer.simple_tile_renderer import SimpleTileRenderer
from imbrush.util.svg_loader import SVGLoader
from imbrush.util.spatial_constrain_visualizer import save_spatial_constraints
from imbrush.util.primitive_loader import PrimitiveLoader
from imbrush.util.svg_converter import FontParser, ImageToSVG
from imbrush.core.initializer.svgsplat_initializater import StructureAwareInitializer
from imbrush.core.initializer.random_initializater import RandomInitializer

# Route visualization flag - set to True to enable primitive movement visualization
ENABLE_ROUTE_VISUALIZATION = False

# Import our modules
from imbrush.core.preprocessing import Preprocessor
from imbrush.util.utils import set_global_seed, gaussian_blur, compute_psnr, extract_chars_from_file
from imbrush.util.pdf_exporter import PDFExporter
import imbrush.util.target_masks as target_masks

# Conditional import for route visualization
if ENABLE_ROUTE_VISUALIZATION:
    from imbrush.util.route_visualizer import create_route_visualization



html_extra_path_special = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser setup
parser = argparse.ArgumentParser(description="Process images with Structure-Aware Graphics Synthesis")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()
config_path = args.config

# Load configuration
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Apply constants as default values
from imbrush.util.constants import apply_constants_to_config, PRIMITIVE_HOLLOW_DEFAULT
config = apply_constants_to_config(config)
# After import or after loading config
set_global_seed(config["seed"])

# Handle multiple images - convert to list if single path
# For CLIP-only mode, img_path can be None
img_paths = config["preprocessing"]["img_path"]
if img_paths is None:
    # Text-to-drawing mode (CLIPDraw style)
    img_paths = [None]
    print("Running in text-to-drawing mode (no target image)")
elif not isinstance(img_paths, list):
    img_paths = [img_paths]
    print(f"Processing {len(img_paths)} image(s)")
else:
    print(f"Processing {len(img_paths)} image(s)")

# Force list attributes to single item
if type(config["preprocessing"]["final_width"]) is list:
    config["preprocessing"]["final_width"] = config["preprocessing"]["final_width"][0]
    print("Use only one final_width to inference")
if type(config["initialization"]["N"]) is list: 
    config["initialization"]["N"] = config["initialization"]["N"][0]
    print("Use only one N to inference")

# Initialize preprocessor configuration (same for all images)
pp_conf = config["preprocessing"]
opt_conf = config["optimization"]
use_fp16 = opt_conf.get("use_fp16", False)  # Default to False for CPU compatibility

exist_bg = pp_conf.get("exist_bg", True)

# Handle primitive file loading (same for all images)
primitive_file_config = config["primitive"].get("primitive_file")

# Handle list of files (multiple primitives)
if isinstance(primitive_file_config, list):
    # Multiple primitives - process each one
    svg_path = []
    for file_item in primitive_file_config:
        file_ext = os.path.splitext(file_item)[1].lower()
        if file_ext == ".svg":
            svg_path.append(os.path.join("assets/svg", file_item))
        elif file_ext in (".png", ".jpg", ".jpeg"):
            if config["primitive"]["convert_to_svg"]:
                img_converter = ImageToSVG()
                converted_path = img_converter.extract_filled_outlines(file_item, threshold=100, min_area_ratio=0.000001)
                svg_path.append(converted_path)
                del img_converter
            else:
                svg_path.append(os.path.join("assets/primitives", file_item))
        else:
            raise ValueError(f"Unsupported file extension in list: {file_ext}")
else:
    # Single primitive - original logic
    primitive_ext = os.path.splitext(primitive_file_config)[1].lower()
    if primitive_ext == ".svg":
        svg_path = os.path.join("assets/svg", primitive_file_config)
    elif primitive_ext in (".png", ".jpg", ".jpeg"):
        if config["primitive"]["convert_to_svg"]:
            img_converter = ImageToSVG()
            svg_path = img_converter.extract_filled_outlines(primitive_file_config, threshold=100, min_area_ratio=0.000001)
            del img_converter
            svg_path = os.path.join("assets/primitives", primitive_file_config)
        else:
            svg_path = os.path.join("assets/primitives", primitive_file_config)
    elif primitive_ext in (".otf", ".ttf"):
        # 텍스트 소스 결정
        texts = None
        if "text" in config["primitive"]:
            texts = config["primitive"]["text"]
        elif "text_file" in config["primitive"]:
            text_ext = os.path.splitext(config["primitive"].get("text_file"))[1].lower()
            text_path = os.path.join("assets/texts", config["primitive"].get("text_file")) 
            
            # 전용 파서 클래스 (여기서는 간단 예시)
            if text_ext == ".txt" or text_ext == ".lrc":
                texts, char_counts, word_lengths_per_line = extract_chars_from_file(text_path, text_ext, remove_punct=config["primitive"]["remove_punctuation"], punct_to_remove=".,;:()\{\}[]\"\'")
                html_extra_path_special = "output_webpage/src_lyrics/index.html"
                config["initialization"]["N"] = sum(char_counts)  # N은 텍스트의 개수로 설정
            else:
                raise ValueError(f"Unsupported text_file type: {text_ext}")

        if texts is not None:
            font_parser = FontParser(config["primitive"]["primitive_file"])
            # texts = config["primitive"]["text"]
            if isinstance(texts, list):
                svg_paths = [str(font_parser.text_to_svg(t, mode="opt-path")) for t in texts]
            else:
                svg_paths = str(font_parser.text_to_svg(texts, mode="opt-path"))
            svg_path = svg_paths
            del font_parser
        else:
            raise ValueError("No text source ('text' or 'text_file') provided in primitive config.")

    else:
        svg_path = primitive_file_config if primitive_file_config else "assets/svg/MaruBuri-Bold_HELLO.svg"

# Load primitives (SVG, PNG, JPG) - same for all images
# Use PrimitiveLoader for hybrid support, fallback to SVGLoader for compatibility
try:
    primitive_loader = PrimitiveLoader(
        primitive_paths=svg_path,
        output_width=config["primitive"]["output_width"],
        device=device,
        bg_threshold=config["primitive"]["bg_threshold"]
    )
    # Keep reference for backward compatibility
    svg_loader = primitive_loader
    print(f"Loaded primitives: {len(primitive_loader.primitive_paths)} files")
    print(f"Primitive types: {primitive_loader.primitive_types}")
except Exception as e:
    print(f"PrimitiveLoader failed, falling back to SVGLoader: {e}")
    svg_loader = SVGLoader(
        svg_path=svg_path,
        output_width=config["primitive"]["output_width"],
        device=device
    )
    primitive_loader = None

# Process each image in the list
for img_idx, img_path in enumerate(img_paths):
    print(f"\n{'='*80}")
    if img_path is None:
        print(f"Text-to-drawing synthesis (no target image)")
    else:
        print(f"Processing image {img_idx + 1}/{len(img_paths)}: {img_path}")
    print(f"{'='*80}")
    
    # Update config with current image path
    config["preprocessing"]["img_path"] = img_path
    
    # Check if this is text-only mode (CLIPDraw style)
    is_text_only_mode = (img_path is None)
    
    if is_text_only_mode:
        # Text-to-drawing mode: create white canvas as placeholder
        H = pp_conf["final_width"]  # Use square canvas
        W = pp_conf["final_width"]
        I_target = torch.ones((H, W, 3), device=device, dtype=torch.float32)
        target_binary_mask_np = None
        print(f"Created white canvas: {W}x{H}")
    else:
        # Initialize preprocessor for current image (each image may have different dimensions)
        preprocessor = Preprocessor(
            final_width=pp_conf["final_width"],
            trim=pp_conf.get("trim", False),
            FM_halftone=pp_conf.get("FM_halftone", False),
            transform_mode=pp_conf.get("transform", "none"),
        )
        
        # Load single image
        if exist_bg:
            print("Target image has background")
            I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
        else:
            print("Target image has no background, using color and opacity")
            I_target,target_binary_mask_np = preprocessor.load_image_8bit_color_opacity(config["preprocessing"])
            I_target = I_target.astype(np.float32) / 255.0

        I_target = torch.tensor(I_target, device=device)  # (H, W, 3) or (H, W, 4) if no background
        H = preprocessor.final_height
        W = preprocessor.final_width

    # Initialize renderer (always use SimpleTileRenderer)
    renderer_class = SimpleTileRenderer

    bmp_tensor = svg_loader.load_alpha_bitmap()
    if use_fp16:
        bmp_tensor = bmp_tensor.to(dtype=torch.float16)
    else:
        bmp_tensor = bmp_tensor.to(dtype=torch.float32)

    # Extract primitive colors for c_o initialization
    if primitive_loader is not None:
        primitive_colors = primitive_loader.get_primitive_color_maps()  # (num_primitives, 3)
        print(f"Extracted primitive colors: {primitive_colors.shape}")
    else:
        # Fallback: use default colors if primitive_loader is not available
        num_primitives = bmp_tensor.shape[0] if bmp_tensor.ndim == 3 else 1
        primitive_colors = torch.zeros(num_primitives, 128, 128, 3, device=device)
        print("Using default colors for primitives")

    # Create renderer only when needed - defer instantiation
    renderer_kwargs = {
        "canvas_size": (H, W),
        "S": bmp_tensor,
        "alpha_upper_bound": config["optimization"]["alpha_upper_bound"],
        "device": device,
        "use_fp16": use_fp16,
        "output_path": config["postprocessing"]["output_folder"],
        "tile_size": opt_conf["tile_size"],
        "sigma": opt_conf["blur_sigma"] if opt_conf.get("do_gaussian_blur", False) else 0.0,
        "c_blend": config["optimization"].get("c_blend", 0.0),  # Pass c_blend from config
        "primitive_colors": primitive_colors,  # Pass primitive colors for c_o initialization
    }

    renderer = renderer_class(**renderer_kwargs)
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

    # Single image optimization
    # Initialize parameters

    # Generate distance masks if requested
    target_binary_mask = None # 1 at backgorund, 0 at foreground

    if not exist_bg:
        target_binary_mask = torch.from_numpy(target_binary_mask_np[:,:]>0).to(device)
        x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target, target_binary_mask)

    else:
        x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target, target_binary_mask)

    bmp_image_tensor = svg_loader.load_alpha_bitmap()
    
    # Optimize parameters
    x, y, r, v, theta, c = renderer.optimize_parameters(
        x, y, r, v, theta, c,
        I_target, 
        opt_conf=opt_conf,
        target_binary_mask=target_binary_mask,
        initializer=initializer
    )

    if not exist_bg:
        I_target = I_target[..., :3]  # Remove alpha channel if exists

    # Save the final rendered image
    output_dir = config["postprocessing"].get("output_folder", "./outputs/")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate unique output suffix for multiple images
    if len(img_paths) > 1:
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        output_suffix = f"_{img_basename}_{img_idx}"
    else:
        output_suffix = ""

    # Single image export
    output_path = os.path.join(output_dir, f'output_{timestamp}{output_suffix}.png')

    # Standard PDF export for SVG-only primitives (if no raster primitives)
    if not (primitive_loader and primitive_loader.has_raster_primitives()):
        # Single image PDF export
        pdf_path = os.path.join(output_dir, f'output_{timestamp}{output_suffix}.pdf')
        exporter = PDFExporter(
            svg_loader.svg_path, 
            canvas_size=(W, H),
            viewbox_size=svg_loader.get_svg_size(),
            alpha_upper_bound=config["optimization"]["alpha_upper_bound"],
            stroke_width=config["postprocessing"]["linewidth"]
        )
        
        exporter.export(x, y, r, theta, v, c,
                        output_path=pdf_path,
                        svg_hollow=config["primitive"]["primitive_hollow"],
                        html_extra_path = "output_webpage/src/index.html" if html_extra_path_special is None else html_extra_path_special,
                        export_pdf=True,
                        html_extra_meta={"char_counts": json.dumps(char_counts), "word_lengths_per_line": json.dumps(word_lengths_per_line)} if 'char_counts' in locals() else {}
        )
    # Single image final rendering and export
    with torch.no_grad():
        white_bg = torch.ones((renderer.H, renderer.W, 3), device=renderer.device)
        dense_mask_path = config['postprocessing'].get('dense_mask', None)
        if dense_mask_path is not None:
            dense_mask_img = cv2.imread(dense_mask_path, cv2.IMREAD_GRAYSCALE)
            dense_mask_img = cv2.resize(dense_mask_img, (renderer.W, renderer.H))
        import imbrush.util.app as app
        x, y, r, v, theta, c = app.mask_blur(x, y, r, v, theta, c,
                      I_target,
                      target_binary_mask=target_binary_mask,
                      dense_mask=dense_mask_img if dense_mask_path is not None else None)
        
        # Check if PSD export is requested
        psd_export = config.get('postprocessing', {}).get('export_psd', False)

        # Convert parameters to FP16 for final rendering if using FP16 renderer
        if renderer.use_fp16:
            from torch.amp import autocast
            with autocast('cuda'):
                rendered, rendered_alpha = renderer.render_from_params(x, y, r, theta, v, c, return_alpha=True, I_bg=white_bg, sigma=0.0, is_final=True)
                
                # Save rendered image directly from rendered tensor 
                rendered_np = rendered.detach().cpu().numpy()
                rendered_np = (rendered_np * 255).astype(np.uint8)
                Image.fromarray(rendered_np).save(output_path)
                if not exist_bg:
                    save_spatial_constraints(rendered, rendered_alpha, output_path)

        else:
            # Still render final PNG for preview/compatibility
            rendered, rendered_alpha = renderer.render_from_params(x, y, r, theta, v, c, return_alpha=True, I_bg=white_bg, sigma=0.0, is_final=True)
            if not exist_bg:
                save_spatial_constraints(rendered, rendered_alpha, output_path)
            # Save rendered image directly from rendered tensor 
            rendered_np = rendered.detach().cpu().numpy()
            rendered_np = (rendered_np * 255).astype(np.uint8)
            Image.fromarray(rendered_np).save(output_path)

        if psd_export:
            # Export PSD layers using util/psd_exporter.py with batched processing
            from imbrush.util.psd_exporter import PSDExporter
            
            psd_path = output_path.replace('.png', '.psd')
            psd_scale_factor = config['postprocessing']['psd_scale_factor']
            
            # Pass c_blend and primitive_colors to PSDExporter
            c_blend = config["optimization"].get("c_blend", 0.0)
            exporter = PSDExporter(
                renderer.W, renderer.H, 
                alpha_upper_bound=renderer.alpha_upper_bound, 
                scale_factor=psd_scale_factor,
                c_blend=c_blend,
                primitive_colors=primitive_colors
            )
            
            # Use batched processing - all data preparation handled internally
            exporter.add_layers_batch_optimized(
                renderer.S, x, y, r, theta, v, c
            )
                
            # Export PSD file
            exporter.export_psd(psd_path)
    

        if config['postprocessing'].get('export_mp4', False):
            video_path = os.path.join(output_dir, f'output_{timestamp}{output_suffix}.mp4')
            # Warning: this takes a long time. TODO: fix this
            renderer.render_export_mp4(x, y, r, theta, v, c, video_path=video_path)

    # Compute metrics if requested
    if config['postprocessing'].get('compute_psnr', False):
        try:
            import piq
            
            # Single image metrics
            # Convert rendered image to tensor format for metrics
                
            rendered_t = rendered.permute(2, 0, 1).unsqueeze(0)
            target_t = I_target.permute(2, 0, 1).unsqueeze(0)
            
            # If no background, apply mask
            # Evaluate only foreground pixels
            if not exist_bg:
                foreground_mask = (target_binary_mask_np ==0)
                mask_tensor = torch.tensor(foreground_mask, device=device, dtype=torch.float32)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                mask_tensor = mask_tensor.expand(-1, 3, -1, -1)  # (1, 3, H, W)

                rendered_t = rendered_t * mask_tensor
                target_t = target_t * mask_tensor

            # Compute metrics
            rendered_t_f32 = rendered_t.float()
            target_t_f32 = target_t.float()
            
            # Clip values to [0, 1] range to avoid PSNR calculation errors
            rendered_t_f32 = torch.clamp(rendered_t_f32, 0.0, 1.0)
            target_t_f32 = torch.clamp(target_t_f32, 0.0, 1.0)

            psnr_val = piq.psnr(rendered_t_f32, target_t_f32, data_range=1.0)
            ssim_val = piq.ssim(rendered_t_f32, target_t_f32, data_range=1.0)
            vif_val = piq.vif_p(rendered_t_f32, target_t_f32, data_range=1.0)
            lpips_val = piq.LPIPS()(rendered_t_f32, target_t_f32)
            
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

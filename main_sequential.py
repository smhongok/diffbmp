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
from imbrush.util.constants import apply_constants_to_config
config = apply_constants_to_config(config)
# After import or after loading config
set_global_seed(config["seed"])

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
    final_width=pp_conf["final_width"],
    trim=pp_conf.get("trim", False),
    FM_halftone=pp_conf.get("FM_halftone", False),
    transform_mode=pp_conf.get("transform", "none"),
)

# Sequential rendering mode - load frames
sequential_config = config["sequential"]
input_path = config["preprocessing"]["img_path"]
input_type = sequential_config["input_type"]
max_frames = sequential_config.get("max_frames", None)
sequential_config = apply_constants_to_config(sequential_config)

print(f"Loading {input_type} frames from: {input_path}")

if input_type == "gif":
    frames = preprocessor.load_gif_frames(input_path, config["preprocessing"])
elif input_type == "video":
    frames = preprocessor.load_video_frames(input_path, config["preprocessing"], max_frames)
elif input_type == "sequence":
    frames = preprocessor.load_image_sequence(input_path, config["preprocessing"])
else:
    raise ValueError(f"Unsupported input_type: {input_type}")

print(f"Loaded {len(frames)} frames")

# Convert frames to tensors
I_targets = []
for frame in frames:
    frame_tensor = torch.tensor(frame.astype(np.float32) / 255.0, device=device)  # (H, W, 3)
    I_targets.append(frame_tensor)
    print("There are {} frames in the input".format(len(I_targets)))

# Use first frame dimensions
H = preprocessor.final_height
W = preprocessor.final_width

# Handle primitive file loading
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

# Load primitives (SVG, PNG, JPG)
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

# Initialize renderer (always use SimpleTileRenderer)
renderer_class = SimpleTileRenderer

bmp_tensor = svg_loader.load_alpha_bitmap()
if use_fp16:
    bmp_tensor = bmp_tensor.to(dtype=torch.float16)
else:
    bmp_tensor = bmp_tensor.to(dtype=torch.float32)
    

H = preprocessor.final_height
W = preprocessor.final_width

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
    "c_blend": config["optimization"].get("c_blend", 0.0),
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

# Sequential frame-by-frame optimization
print("Starting sequential frame-by-frame optimization...")

# Create sequential renderer for subsequent frames
sequential_renderer = SequentialFrameRenderer(
    canvas_size=(H, W), 
    S=bmp_tensor,
    alpha_upper_bound=config["optimization"]["alpha_upper_bound"],
    device=device,
    use_fp16=use_fp16,
    gamma=config["optimization"].get("gamma", 1.0),
    output_path=config["postprocessing"]["output_folder"],
    tile_size=sequential_config["tile_size"],
    sigma = opt_conf["blur_sigma"] if opt_conf.get("do_gaussian_blur", False) else 0.0,
    c_blend=config["optimization"].get("c_blend", 0.0),
    primitive_colors=primitive_colors,
)
print(f"Using SequentialFrameRenderer with tile-based rendering (tile_size: {sequential_config['tile_size']})")

# Store optimized parameters for each frame
frame_results = []
prev_params = None

for frame_idx, I_target_frame in enumerate(I_targets):
    print(f"\nOptimizing frame {frame_idx + 1}/{len(I_targets)}...")
    
    if frame_idx == 0:
        # First frame: initialize from scratch using standard renderer
        print("First frame: initializing from scratch with SimpleTileRenderer")
        x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target_frame)
        
        # Optimize with standard renderer using optimization config
        start_time_frame = time.time()
        x, y, r, v, theta, c = renderer.optimize_parameters(
            x, y, r, v, theta, c,
            I_target_frame, 
            opt_conf=opt_conf
        )
        end_time_frame = time.time()
        
        # No need for neighbor computation with simple anchoring loss
        
    else:
        # Subsequent frames: use SequentialFrameRenderer with temporal consistency
        print(f"Subsequent frame: initializing from frame {frame_idx} with SequentialFrameRenderer")
        
        # Use sequential optimization settings and include adaptive control
        optimization_config = sequential_config.get("optimization", {}) 
        
        # Add adaptive control configuration to optimization config
        adaptive_control_config = sequential_config.get("adaptive_control", {})
        optimization_config["adaptive_control"] = adaptive_control_config

        # Add selective parameter optimization configuration to optimization config
        selective_parameter_optimization_config = sequential_config.get("selective_parameter_optimization", {})
        optimization_config["selective_parameter_optimization"] = selective_parameter_optimization_config
        

        if adaptive_control_config.get("enabled", False) and not sequential_config.get("seperate_init", False):
            print("Using adaptive control")
        
        if sequential_config.get("seperate_init", False):
            
            print("seperately initializing for every frame")
            
            x, y, r, v, theta, c = sequential_renderer.initialize_parameters(initializer, I_target_frame)
            
            start_time_frame = time.time()
            x, y, r, v, theta, c = renderer.optimize_parameters(
            x, y, r, v, theta, c,
            I_target_frame, 
            opt_conf=frame_opt_conf
            )

            end_time_frame = time.time()
        else:
            print("initializing from previous frame")
            # Choose optimization strategy

            if (frame_idx > 0) and sequential_config.get("selective_parameter_optimization").get("enabled", False):
                print("Using selective parameter optimization")
                previous_frame = I_targets[frame_idx-1]
            else:
                print("Using full parameter optimization")
                previous_frame = None

            start_time_frame = time.time()
            x, y, r, v, theta, c = sequential_renderer.optimize_parameters_full_temporal(
                x, y, r, v, theta, c, I_target_frame, prev_params, optimization_config, previous_frame
            )
            end_time_frame = time.time()
    
    # Render final frame for export
    with torch.no_grad():
        white_bg = torch.ones((sequential_renderer.H, sequential_renderer.W, 3), device=sequential_renderer.device)
        frame_rendered = sequential_renderer.render_from_params(x, y, r, theta, v, c, I_bg=white_bg, sigma=0.0, is_final=True)
    
    # Store results for this frame
    current_params = {
        'x': x.clone(),
        'y': y.clone(), 
        'r': r.clone(),
        'v': v.clone(),
        'theta': theta.clone(),
        'c': c.clone(),
        'rendered_frame': frame_rendered.clone()
    }
    
    frame_results.append({
        **current_params,
        'optimization_time': end_time_frame - start_time_frame
    })
    
    # Update previous parameters for next frame
    prev_params = current_params
    
    print(f"Frame {frame_idx + 1} optimization completed in {end_time_frame - start_time_frame:.2f}s")

print(f"\nSequential optimization completed for {len(frame_results)} frames")

# Save the final rendered image
output_dir = config["postprocessing"].get("output_folder", "./outputs/")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Sequential export: create frame sequence, GIF, and MP4
print("\nExporting sequential frames...")

# Create subdirectory for frame sequence
frames_dir = os.path.join(output_dir, f'frames_{timestamp}')
os.makedirs(frames_dir, exist_ok=True)

# Export individual frames
exported_frames = []
for frame_idx, frame_result in enumerate(frame_results):
    print(f"Exporting frame {frame_idx + 1}/{len(frame_results)}...")
    
    # Get the already rendered frame
    rendered_frame = frame_result['rendered_frame']
    
    # Extract parameters for individual frame saving
    x_frame = frame_result['x']
    y_frame = frame_result['y']
    r_frame = frame_result['r']
    v_frame = frame_result['v']
    theta_frame = frame_result['theta']
    c_frame = frame_result['c']
    

    # Still render final PNG for preview/compatibility using tile-based rendering
    with torch.no_grad():
        white_bg = torch.ones((sequential_renderer.H, sequential_renderer.W, 3), device=sequential_renderer.device)
        frame_rendered = sequential_renderer.render_from_params(x_frame, y_frame, r_frame, theta_frame, v_frame, c_frame, I_bg=white_bg, sigma=0.0, is_final=True)
    # Save rendered frame directly
    frame_rendered_np = frame_rendered.detach().cpu().numpy()
    frame_rendered_np = (frame_rendered_np * 255).astype(np.uint8)
    frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
    Image.fromarray(frame_rendered_np).save(frame_path)
    
    # Store rendered frame for GIF/MP4 export
    frame_np = rendered_frame.cpu().numpy()
    frame_np = (frame_np * 255).astype(np.uint8)
    exported_frames.append(frame_np)
    
    # Check if PSD export is requested
    psd_export = config.get('postprocessing', {}).get('export_psd', False)
    
    if psd_export:
        # Export PSD layers using util/psd_exporter.py with batched processing
        from imbrush.util.psd_exporter import PSDExporter
        
        psd_path = frame_path.replace('.png', '.psd')
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
    

# Generate route visualization if enabled (after all frames are processed)
if ENABLE_ROUTE_VISUALIZATION and len(frame_results) >= 2:
    print("\nGenerating primitive movement route visualization...")
    route_viz_path = os.path.join(output_dir, f'primitive_routes_{timestamp}.png')
    try:
        create_route_visualization(
            frame_results=frame_results,
            output_path=route_viz_path,
            line_width=1.5,
            alpha=0.7,
            max_primitives=1000,
            color_scheme='rainbow',
            interpolation_points=10
        )
        print(f"Route visualization saved to: {route_viz_path}")
    except Exception as e:
        print(f"Warning: Failed to generate route visualization: {e}")

# Export GIF
export_config = sequential_config.get("export", {})
if export_config.get("export_gif", True):
    gif_path = os.path.join(output_dir, f'output_{timestamp}.gif')
    print(f"Creating GIF: {gif_path}")
    
    # Convert frames to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in exported_frames]
    
    # Save as GIF
    frame_duration = export_config.get("frame_duration", 100)  # milliseconds
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration,
        loop=0
    )
    print(f"GIF saved: {gif_path}")

# Export MP4
if export_config.get("export_mp4", True):
    mp4_path = os.path.join(output_dir, f'output_{timestamp}.mp4')
    print(f"Creating MP4: {mp4_path}")
    
    # Use OpenCV to create MP4
    fps = 1000.0 / export_config.get("frame_duration", 100)  # Convert ms to fps
    height, width = exported_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
    
    for frame in exported_frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"MP4 saved: {mp4_path}")

print(f"Sequential export completed. {len(exported_frames)} frames exported.")

# Standard PDF export for SVG-only primitives (if no raster primitives)
if not (primitive_loader and primitive_loader.has_raster_primitives()):
    # For sequential processing, export PDF for the last frame
    if frame_results:
        pdf_path = os.path.join(output_dir, f'output_{timestamp}_final_frame.pdf')
        last_frame = frame_results[-1]
        exporter = PDFExporter(
            svg_loader.svg_path, 
            canvas_size=(W, H),
            viewbox_size=svg_loader.get_svg_size(),
            alpha_upper_bound=config["optimization"]["alpha_upper_bound"],
            stroke_width=config["postprocessing"]["linewidth"]
        )
        
        # Use sequential-specific HTML path and export sequence
        lastframe_html_path = "output_webpage/src/index.html"
        
        # Choose sequential HTML path based on text file type
        if 'text_ext' in locals() and (text_ext == ".txt" or text_ext == ".lrc"):
            sequential_html_path = "output_webpage/src_sequential_lyrics/index.html"
        else:
            sequential_html_path = "output_webpage/src_sequential/index.html"
        
        # Export PDF for the last frame
        exporter.export(last_frame['x'], last_frame['y'], last_frame['r'], 
                      last_frame['theta'], last_frame['v'], last_frame['c'],
                      output_path=pdf_path,
                      svg_hollow=config["primitive"].get("primitive_hollow", False),
                      html_extra_path=lastframe_html_path,
                      export_pdf=True,
                      html_extra_meta={"char_counts": json.dumps(char_counts), "word_lengths_per_line": json.dumps(word_lengths_per_line)} if 'char_counts' in locals() else {}
        )
        
        # Export HTML sequence animation
        export_config = sequential_config.get("export", {})
        sequence_fps = export_config.get("sequence_fps", 24)
        
        print(f"\nExporting HTML sequence animation...")
        exporter.export_sequence(
            frame_results=frame_results,
            output_html_path=sequential_html_path,
            svg_hollow=config["primitive"].get("primitive_hollow", False),
            fps=sequence_fps,
            html_extra_meta={"char_counts": json.dumps(char_counts), "word_lengths_per_line": json.dumps(word_lengths_per_line)} if 'char_counts' in locals() else {}
        )
        print(f"HTML sequence exported: {sequential_html_path}")

# Compute metrics if requested
if config['postprocessing'].get('compute_psnr', False):
    try:
        import piq
        
        # Compute metrics for each frame in sequential processing
        print("\nComputing metrics for sequential frames...")
        total_psnr = 0
        total_ssim = 0
        total_vif = 0
        total_lpips = 0
        
        # Prepare metrics txt output path under configured output folder
        metrics_txt_path = os.path.join(output_dir, f"metrics_{timestamp}.txt")
        try:
            metrics_fh = open(metrics_txt_path, 'w')
            metrics_fh.write("Sequential frame metrics\n")
            metrics_fh.write(f"Timestamp: {timestamp}\n")
            
            # Write adaptive_control configuration snapshot for record
            try:
                ac_conf = sequential_config.get("adaptive_control", {})
                metrics_fh.write("adaptive_control_config\n")
                metrics_fh.write(f"  enabled: {ac_conf.get('enabled', False)}\n")
                metrics_fh.write(f"  tile_rows: {ac_conf.get('tile_rows', '')}\n")
                metrics_fh.write(f"  tile_cols: {ac_conf.get('tile_cols', '')}\n")
                metrics_fh.write(f"  scale_threshold: {ac_conf.get('scale_threshold', '')}\n")
                metrics_fh.write(f"  opacity_threshold: {ac_conf.get('opacity_threshold', '')}\n")
                metrics_fh.write(f"  opacity_reduction_factor: {ac_conf.get('opacity_reduction_factor', '')}\n")
                metrics_fh.write(f"  max_primitives_per_tile: {ac_conf.get('max_primitives_per_tile', '')}\n")
                metrics_fh.write(f"  min_criteria_count: {ac_conf.get('min_criteria_count', '')}\n")
                metrics_fh.write(f"  front_primitives_percentile: {ac_conf.get('front_primitives_percentile', '')}\n")
                metrics_fh.write(f"  apply_epochs: {ac_conf.get('apply_epochs', '')}\n")
                gr_conf = ac_conf.get('gradient_ranking', {})
                if isinstance(gr_conf, dict):
                    metrics_fh.write("  gradient_ranking:\n")
                    metrics_fh.write(f"    enabled: {gr_conf.get('enabled', False)}\n")
                    metrics_fh.write(f"    process_all_pixels: {gr_conf.get('process_all_pixels', False)}\n")
                    metrics_fh.write(f"    pixels_per_tile: {gr_conf.get('pixels_per_tile', '')}\n")
            except Exception as e:
                print(f"Warning: Failed to write adaptive_control config to metrics file: {e}")
            
            # Write selective_parameter_optimization configuration snapshot for record
            try:
                spo_conf = sequential_config.get("selective_parameter_optimization", {})
                metrics_fh.write("selective_parameter_optimization_config\n")
                metrics_fh.write(f"  enabled: {spo_conf.get('enabled', False)}\n")
                metrics_fh.write(f"  freeze_distance_threshold: {spo_conf.get('freeze_distance_threshold', '')}\n")                
                metrics_fh.write(f"  diff_magnitude_threshold: {spo_conf.get('diff_magnitude_threshold', '')}\n")
                gf_conf = spo_conf.get('gradual_freeze', {})
                if isinstance(gf_conf, dict):
                    metrics_fh.write("  gradual_freeze:\n")
                    metrics_fh.write(f"    enabled: {gf_conf.get('enabled', False)}\n")
                    metrics_fh.write(f"    strength: {gf_conf.get('strength', '')}\n")
            except Exception as e:
                print(f"Warning: Failed to write selective_parameter_optimization config to metrics file: {e}")

            metrics_fh.write("frame,psnr,ssim,vif,lpips\n")
        except Exception as e:
            print(f"Warning: Could not open metrics file for writing: {e}")
            metrics_fh = None
        
        for frame_idx, (frame_result, I_target_frame) in enumerate(zip(frame_results, I_targets)):
            # Render frame for metrics using tile-based rendering
            with torch.no_grad():
                white_bg = torch.ones((renderer.H, renderer.W, 3), device=renderer.device)
                rendered_frame = renderer.render_from_params(
                    frame_result['x'], frame_result['y'], frame_result['r'], frame_result['theta'],
                    frame_result['v'], frame_result['c'], I_bg=white_bg, sigma=0.0, is_final=True
                )
            
            # Convert to tensor format for metrics
            rendered_t = rendered_frame.permute(2, 0, 1).unsqueeze(0)
            target_t = I_target_frame.permute(2, 0, 1).unsqueeze(0)
            
            rendered_t_f32 = rendered_t.float()
            target_t_f32 = target_t.float()

            # Compute metrics for this frame
            psnr_val = piq.psnr(rendered_t_f32, target_t_f32, data_range=1.0)
            ssim_val = piq.ssim(rendered_t_f32, target_t_f32, data_range=1.0)
            vif_val = piq.vif_p(rendered_t_f32, target_t_f32, data_range=1.0)
            lpips_val = piq.LPIPS()(rendered_t_f32, target_t_f32)
            
            print(f"Frame {frame_idx + 1}: PSNR: {psnr_val.item():.2f} dB, SSIM: {ssim_val.item():.4f}, VIF: {vif_val.item():.4f}, LPIPS: {lpips_val.item():.4f}")
            
            # Write per-frame metrics to txt file if available
            if metrics_fh is not None:
                try:
                    metrics_fh.write(f"{frame_idx + 1},{psnr_val.item():.4f},{ssim_val.item():.6f},{vif_val.item():.6f},{lpips_val.item():.6f}\n")
                except Exception as e:
                    print(f"Warning: Failed to write metrics for frame {frame_idx + 1}: {e}")
            
            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()
            total_vif += vif_val.item()
            total_lpips += lpips_val.item()
        
        # Print average metrics
        num_frames = len(frame_results)
        print(f"\nAverage metrics across {num_frames} frames:")
        print(f"PSNR: {total_psnr / num_frames:.2f} dB")
        print(f"SSIM: {total_ssim / num_frames:.4f}")
        print(f"VIF: {total_vif / num_frames:.4f}")
        print(f"LPIPS: {total_lpips / num_frames:.4f}")
        print(f"Number of splats: {len(frame_results[0]['x'])}")
        
        # Append averages to metrics txt and close file
        if metrics_fh is not None:
            try:
                metrics_fh.write("\nAverages\n")
                metrics_fh.write(f"avg_psnr,{total_psnr / num_frames:.4f}\n")
                metrics_fh.write(f"avg_ssim,{total_ssim / num_frames:.6f}\n")
                metrics_fh.write(f"avg_vif,{total_vif / num_frames:.6f}\n")
                metrics_fh.write(f"avg_lpips,{total_lpips / num_frames:.6f}\n")
                metrics_fh.close()
                print(f"Per-frame metrics saved to: {metrics_txt_path}")
            except Exception as e:
                print(f"Warning: Failed to finalize metrics file: {e}")
            
    except ImportError as e:
        print(f"Required library missing: {e}. Cannot compute metrics.")

end_time = time.time()
formatted_time = str(timedelta(seconds=int(end_time - start_time)))
print(f"total_cost_time: {formatted_time}")

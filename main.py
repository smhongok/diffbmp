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
from core.renderer.mse_renderer import MseRenderer
from core.renderer.sequential_renderer import SequentialFrameRenderer
from core.renderer.simple_tile_renderer import SimpleTileRenderer
from util.svg_loader import SVGLoader
from util.primitive_loader import PrimitiveLoader
from util.svg_converter import FontParser, ImageToSVG
from core.initializer.svgsplat_initializater import StructureAwareInitializer
from core.initializer.random_initializater import RandomInitializer

# Import our modules
from core.preprocessing import Preprocessor
from util.utils import set_global_seed, gaussian_blur, compute_psnr, extract_chars_from_file
from util.pdf_exporter import PDFExporter
import util.target_masks as target_masks



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

exist_bg = pp_conf.get("exist_bg", True)

# Check if sequential processing is enabled
sequential_config = config.get("sequential", {"enabled": False})

if not exist_bg and sequential_config.get("enabled", False):
    raise NotImplementedError("Sequential processing is not supported for images without background. Please set 'exist_bg' to True in the config.")

if sequential_config.get("enabled", False):
    # Load frames for sequential processing
    input_path = config["preprocessing"]["img_path"]
    input_type = sequential_config.get("input_type", "gif")
    max_frames = sequential_config.get("max_frames", None)
    
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
else:
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

# Handle SVG file loading
svg_ext = os.path.splitext(config["svg"].get("svg_file"))[1].lower()
if svg_ext == ".svg":
    svg_path = os.path.join("assets/svg", config["svg"].get("svg_file"))
elif svg_ext in (".png", ".jpg", ".jpeg"):
    if config["svg"].get("convert_to_svg", True):
        img_converter = ImageToSVG()
        svg_path = img_converter.extract_filled_outlines(config["svg"].get("svg_file"), threshold=100, min_area_ratio=0.000001)
        del img_converter
    else:
        svg_path = os.path.join("assets/primitives", config["svg"].get("svg_file"))
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

# Load primitives (SVG, PNG, JPG)
# Use PrimitiveLoader for hybrid support, fallback to SVGLoader for compatibility
try:
    primitive_loader = PrimitiveLoader(
        primitive_paths=svg_path,
        output_width=config["svg"].get("output_width", 128),
        device=device,
        bg_threshold=config["svg"].get("bg_threshold", 250)
    )
    # Keep reference for backward compatibility
    svg_loader = primitive_loader
    print(f"Loaded primitives: {len(primitive_loader.primitive_paths)} files")
    print(f"Primitive types: {primitive_loader.primitive_types}")
except Exception as e:
    print(f"PrimitiveLoader failed, falling back to SVGLoader: {e}")
    svg_loader = SVGLoader(
        svg_path=svg_path,
        output_width=config["svg"].get("output_width", 128),
        device=device
    )
    primitive_loader = None

# Initialize renderer based on loss type
renderer_type = opt_conf.get("renderer_type", "mse")
renderer_class = {
    "mse": MseRenderer,
    "tile": SimpleTileRenderer,
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
renderer_kwargs = {
    "canvas_size": (H, W),
    "S": bmp_tensor,
    "alpha_upper_bound": config["optimization"].get("alpha_upper_bound", 0.5),
    "device": device,
    "use_fp16": use_fp16,
    "output_path": config["postprocessing"].get("output_folder", "./outputs/")
}

# Add tile_size parameter for SimpleTileRenderer
if renderer_type.lower() == "tile":
    renderer_kwargs["tile_size"] = opt_conf.get("tile_size", 32)

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

if sequential_config.get("enabled", False):
    # Sequential frame-by-frame optimization
    print("Starting sequential frame-by-frame optimization...")
    
    # Create sequential renderer for subsequent frames
    sequential_renderer = SequentialFrameRenderer(
        (H, W), S=bmp_tensor,
        alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5),
        device=device,
        use_fp16=use_fp16,
        gamma=config["optimization"].get("gamma", 1.0)
    )
    
    # Store optimized parameters for each frame
    frame_results = []
    prev_params = None
    
    for frame_idx, I_target_frame in enumerate(I_targets):
        print(f"\nOptimizing frame {frame_idx + 1}/{len(I_targets)}...")
        
        if frame_idx == 0:
            # First frame: initialize from scratch using standard MseRenderer
            print("First frame: initializing from scratch with MseRenderer")
            x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target_frame)
            
            # Use first frame optimization settings
            frame_opt_conf = opt_conf.copy()
            first_frame_conf = config["optimization"].get("first_frame", {})
            frame_opt_conf.update(first_frame_conf)
            
            # Optimize with standard renderer
            start_time_frame = time.time()
            x, y, r, v, theta, c = renderer.optimize_parameters(
                x, y, r, v, theta, c,
                I_target_frame, 
                opt_conf=frame_opt_conf
            )
            end_time_frame = time.time()
            
            # No need for neighbor computation with simple anchoring loss
            
        else:
            # Subsequent frames: use SequentialFrameRenderer with temporal consistency
            print(f"Subsequent frame: initializing from frame {frame_idx} with SequentialFrameRenderer")
            
            # Use sequential optimization settings
            optimization_config = sequential_config.get("optimization", {})
            
            # Choose optimization strategy
            start_time_frame = time.time()
            x, y, r, v, theta, c = sequential_renderer.optimize_parameters_full_temporal(
                x, y, r, v, theta, c, I_target_frame, prev_params, optimization_config
            )
            end_time_frame = time.time()
        
        # Render final frame for export
        with torch.no_grad():
            cached_masks = sequential_renderer._batched_soft_rasterize(x, y, r, theta, sigma=0)
            rendered_frame = sequential_renderer.render(cached_masks, v, c)
        
        # Store results for this frame
        current_params = {
            'x': x.clone(),
            'y': y.clone(), 
            'r': r.clone(),
            'v': v.clone(),
            'theta': theta.clone(),
            'c': c.clone(),
            'rendered_frame': rendered_frame.clone()
        }
        
        frame_results.append({
            **current_params,
            'optimization_time': end_time_frame - start_time_frame
        })
        
        # Update previous parameters for next frame
        prev_params = current_params
        
        print(f"Frame {frame_idx + 1} optimization completed in {end_time_frame - start_time_frame:.2f}s")
    
    print(f"\nSequential optimization completed for {len(frame_results)} frames")
    
else:
    # Single image optimization (original behavior)
    # Initialize parameters

    # Generate distance masks if requested
    target_binary_mask = None # 1 at backgorund, 0 at foreground
    target_dist_mask = None # The distance based mask default : Skeleton - Aware Distance Transform

    if not exist_bg:
        target_binary_mask = torch.from_numpy(target_binary_mask_np[:,:]>0).to(device) 
        target_dist_mask = target_masks.SADT_L2(target_binary_mask_np, device)

    x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target, target_binary_mask)
    
    bmp_image_tensor = svg_loader.load_alpha_bitmap()
    
    # Optimize parameters
    x, y, r, v, theta, c = renderer.optimize_parameters(
        x, y, r, v, theta, c,
        I_target, 
        opt_conf=opt_conf,
        target_binary_mask=target_binary_mask,
        target_dist_mask=target_dist_mask,
    )

if not exist_bg:
    I_target = I_target[..., :3]  # Remove alpha channel if exists

# Save the final rendered image
output_dir = config["postprocessing"].get("output_folder", "./outputs/")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

if sequential_config.get("enabled", False):
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
        
        # Save individual frame if requested
        export_config = sequential_config.get("export", {})
        if export_config.get("export_frames", True):
            frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            with torch.no_grad():
                cached_masks = sequential_renderer._batched_soft_rasterize(x_frame, y_frame, r_frame, theta_frame, sigma=0)
                sequential_renderer.save_rendered_image(cached_masks, v_frame, c_frame, frame_path)
        
        # Store rendered frame for GIF/MP4 export
        frame_np = rendered_frame.cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        exported_frames.append(frame_np)
    
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
    
else:
    # Single image export (original behavior)
    output_path = os.path.join(output_dir, f'output_{timestamp}.png')

# Standard PDF export for SVG-only primitives (if no raster primitives)
if not (primitive_loader and primitive_loader.has_raster_primitives()):
    if sequential_config.get("enabled", False):
        # For sequential processing, export PDF for the last frame
        if frame_results:
            pdf_path = os.path.join(output_dir, f'output_{timestamp}_final_frame.pdf')
            last_frame = frame_results[-1]
            exporter = PDFExporter(
                svg_loader.svg_path, 
                canvas_size=(W, H),
                viewbox_size=svg_loader.get_svg_size(),
                alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5),
                stroke_width=config["postprocessing"].get("linewidth", 3.0)
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
                          svg_hollow=config["svg"].get("svg_hollow", False),
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
                svg_hollow=config["svg"].get("svg_hollow", False),
                fps=sequence_fps,
                html_extra_meta={"char_counts": json.dumps(char_counts), "word_lengths_per_line": json.dumps(word_lengths_per_line)} if 'char_counts' in locals() else {}
            )
            print(f"HTML sequence exported: {sequential_html_path}")
    else:
        # Single image PDF export (original behavior)
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
if not sequential_config.get("enabled", False):
    # Single image final rendering and export (original behavior)
    with torch.no_grad():
        # Generate final masks and render
        cached_masks = renderer._batched_soft_rasterize(
            x, y, r, theta,
            sigma=0
        )
        if not exist_bg:
            alpha_loss = (cached_masks * target_binary_mask.unsqueeze(0)).sum(dim=0).mean()

        white_bg = torch.ones((renderer.H, renderer.W, 3), device=cached_masks.device)
        rendered = renderer.render(cached_masks, v, c, I_bg=white_bg)
        renderer.save_rendered_image(cached_masks, v, c, output_path)
        # High-resolution export configuration (recommended only when you have raster primitives)
        hires_enabled = config["postprocessing"].get("hires_export", False)
        scale_factor = config["postprocessing"].get("hires_scale_factor", 4.0)
        if hires_enabled:
            # High-resolution MP4 export using streaming approach
            warnings.warn("High-resolution export is not recommended for vector primitives. Use it only when you have raster primitives.")
            hires_mp4_path = os.path.join(output_dir, f'output_{timestamp}_hires.mp4')
            print(f"Generating high-resolution MP4 ({scale_factor}x scale)...")
            renderer.render_export_mp4_hires(
                x, y, r, theta, v, c,
                video_path=hires_mp4_path,
                scale_factor=scale_factor,
                fps=60
            )
        else:
            video_path = os.path.join(output_dir, f'output_{timestamp}.mp4')
            renderer.render_export_mp4(cached_masks, v, c, video_path=video_path)

# Compute metrics if requested
if config['postprocessing'].get('compute_psnr', False):
    try:
        import piq
        
        if sequential_config.get("enabled", False):
            # Compute metrics for each frame in sequential processing
            print("\nComputing metrics for sequential frames...")
            total_psnr = 0
            total_ssim = 0
            total_vif = 0
            total_lpips = 0
            
            for frame_idx, (frame_result, I_target_frame) in enumerate(zip(frame_results, I_targets)):
                # Render frame for metrics
                with torch.no_grad():
                    cached_masks = renderer._batched_soft_rasterize(
                        frame_result['x'], frame_result['y'], frame_result['r'], frame_result['theta'],
                        sigma=0
                    )
                    rendered_frame = renderer.render(cached_masks, frame_result['v'], frame_result['c'])
                
                # Convert to tensor format for metrics
                rendered_t = rendered_frame.permute(2, 0, 1).unsqueeze(0)
                target_t = I_target_frame.permute(2, 0, 1).unsqueeze(0)
                
                # Compute metrics for this frame
                psnr_val = piq.psnr(rendered_t, target_t, data_range=1.0)
                ssim_val = piq.ssim(rendered_t, target_t, data_range=1.0)
                vif_val = piq.vif_p(rendered_t, target_t, data_range=1.0)
                lpips_val = piq.LPIPS()(rendered_t, target_t)
                
                print(f"Frame {frame_idx + 1}: PSNR: {psnr_val.item():.2f} dB, SSIM: {ssim_val.item():.4f}, VIF: {vif_val.item():.4f}, LPIPS: {lpips_val.item():.4f}")
                
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
            
            # Save trained parameters for testing
            from test_cuda_forward import save_trained_parameters
            save_trained_parameters(
                frame_results[0]['x'], frame_results[0]['y'], frame_results[0]['r'], 
                frame_results[0]['theta'], frame_results[0]['v'], frame_results[0]['c'],
                renderer.S, (H, W)
            )
            
        else:
            # Single image metrics (original behavior)
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
            psnr_val = piq.psnr(rendered_t, target_t, data_range=1.0)
            ssim_val = piq.ssim(rendered_t, target_t, data_range=1.0)
            vif_val = piq.vif_p(rendered_t, target_t, data_range=1.0)
            lpips_val = piq.LPIPS()(rendered_t, target_t)

            # If no background, compute alpha loss
            if not exist_bg:
                print("Alpha loss (X 10^3): {:.4f}".format(alpha_loss.item()*1000.0))
            
            print(f"PSNR: {psnr_val.item():.2f} dB")
            print(f"SSIM: {ssim_val.item():.4f}")
            print(f"VIF: {vif_val.item():.4f}")
            print(f"LPIPS: {lpips_val.item():.4f}")
            print(f"Number of splats: {len(x)}")
            
            # Save trained parameters for testing
            from test_cuda_forward import save_trained_parameters
            save_trained_parameters(x, y, r, theta, v, c, renderer.S, (H, W))
            
    except ImportError as e:
        print(f"Required library missing: {e}. Cannot compute metrics.")

end_time = time.time()
formatted_time = str(timedelta(seconds=int(end_time - start_time)))
print(f"total_cost_time: {formatted_time}")

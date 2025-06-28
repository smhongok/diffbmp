import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import piq
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime, timedelta
import os
from itertools import product
import gc
import pandas as pd

from core.preprocessing import Preprocessor
from util.svg_loader import SVGLoader
from util.svg_converter import FontParser, ImageToSVG
from util.utils import set_global_seed, gaussian_blur
from util.pdf_exporter import PDFExporter

# Enable gradient checkpointing for memory efficiency
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute image quality metrics."""
    # Convert to correct format (NCHW) and ensure float32 dtype
    if isinstance(pred, torch.Tensor):
        if pred.dtype == torch.float16:
            pred = pred.float()
        pred_t = pred.permute(2, 0, 1).unsqueeze(0)
    else:
        # If numpy array, ensure float32
        if pred.dtype == np.float16:
            pred = pred.astype(np.float32)
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
    
    if isinstance(target, torch.Tensor):
        if target.dtype == torch.float16:
            target = target.float()
        target_t = target.permute(2, 0, 1).unsqueeze(0)
    else:
        # If numpy array, ensure float32
        if target.dtype == np.float16:
            target = target.astype(np.float32)
        target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
    
    # For LPIPS, convert to [-1, 1] range
    pred_lpips = pred_t * 2 - 1
    target_lpips = target_t * 2 - 1
    
    # Compute metrics
    # PSNR, SSIM, VIF expect [0, 1] range
    psnr = piq.psnr(pred_t, target_t, data_range=1.0).item()
    ssim = piq.ssim(pred_t, target_t, data_range=1.0).item()
    vif = piq.vif_p(pred_t, target_t, data_range=1.0).item()
    # LPIPS expects [-1, 1] range
    lpips = piq.LPIPS()(pred_lpips, target_lpips).item()
    
    # Clear memory
    del pred_t, target_t, pred_lpips, target_lpips
    torch.cuda.empty_cache()
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'VIF': vif,
        'LPIPS': lpips
    }

def process_combination(args):
    """Process a single initializer-renderer combination."""
    initial_params, init, renderer_name, config, I_target, svg_loader = args
    print(f"\nUsing {init.__class__.__name__} with {renderer_name}")
    
    # Move tensors to device
    I_target = I_target.to(device)
    
    # Check if we should use FP16 (half precision) for memory efficiency
    use_fp16 = config["optimization"].get("use_fp16", False)  # Default to False for CPU compatibility
    
    # Track VRAM before processing
    peak_memory_before = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    # Get bitmap tensor for renderer with appropriate precision
    bmp_tensor = svg_loader.load_alpha_bitmap()
    if use_fp16:
        bmp_tensor = bmp_tensor.to(dtype=torch.float16)
    else:
        bmp_tensor = bmp_tensor.to(dtype=torch.float32)
        
    H, W = config['canvas_size']
    
    # Create renderer only when needed - defer instantiation
    if renderer_name == "MseRenderer":
        from core.renderer.mse_renderer import MseRenderer
        renderer = MseRenderer((H, W), S=bmp_tensor, 
                            alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                            device=device,
                            use_fp16=use_fp16,
                            output_path=config["postprocessing"].get("output_folder", "./outputs/"))
    else:
        raise ValueError(f"Unknown renderer: {renderer_name}")
    
    # Copy input bitmap to renderer then free from memory
    del bmp_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    # Enable checkpointing for memory-intensive renderers
    if hasattr(renderer, 'enable_checkpointing'):
        renderer.enable_checkpointing()
        if hasattr(renderer, 'memory_report'):
            renderer.memory_report(f"Initial memory for {renderer_name}")
    0
    x, y, r, v, theta, c = [t.clone().detach().to(device).requires_grad_(True) for t in initial_params]
    
    # Start timing the initialization and optimization
    start_time = time.time()
    
    # Use autocast only when use_fp16 is True
    if use_fp16:
        if "LevelInitializer" in init.__class__.__name__:
            print(f"{init.__class__.__name__} init and optim skip since it's already done in main()")    
        else:
            # Enable gradient checkpointing for memory efficiency for all renderers
            if hasattr(renderer, 'enable_checkpointing'):
                renderer.enable_checkpointing()
                print(f"Enabled checkpointing for {renderer_name}")
            
            # Add memory report at start if available
            if hasattr(renderer, 'memory_report'):
                renderer.memory_report(f"Before optimization with {renderer_name}")
            
            # Enable debug_memory in optimization config if the renderer supports memory reporting
            if hasattr(renderer, 'memory_report'):
                config['optimization']['debug_memory'] = False
            
            # Optimize parameters - this is the memory-intensive part
            x, y, r, v, theta, c = renderer.optimize_parameters(
                x, y, r, v, theta, c,
                I_target, 
                opt_conf=config['optimization']
            )
            
            # Memory report after optimization
            if hasattr(renderer, 'memory_report'):
                renderer.memory_report(f"After optimization with {renderer_name}")
    else:
        # Standard FP32 processing without autocast
        if "LevelInitializer" in init.__class__.__name__:
            print(f"{init.__class__.__name__} init and optim skip since it's already done in main()")    
        else:
            # Optimize parameters 
            x, y, r, v, theta, c = renderer.optimize_parameters(
                x, y, r, v, theta, c,
                I_target, 
                opt_conf=config['optimization']
            )
    
    # End timing after optimization completes
    runtime = time.time() - start_time
    
    # Generate final render (not included in timing)
    if use_fp16:
        with torch.no_grad():  # Disable gradient computation for final render
            # Process in smaller chunks to save memory
            opt_conf = config['optimization']
            stream_render = opt_conf.get("streaming_render", False)
            
            if stream_render:
                rendered = renderer._stream_render(
                    x, y, r, theta, v, c,
                    sigma=0.0
                )
            else:
                cached_masks = renderer._batched_soft_rasterize(
                    x, y, r, theta,
                    sigma=0.0
                )
                rendered = renderer.render(cached_masks, v, c)
                del cached_masks
    else:
        with torch.no_grad():  # Disable gradient computation for final render
            cached_masks = renderer._batched_soft_rasterize(
                x, y, r, theta,
                sigma=0.0
            )
            rendered = renderer.render(cached_masks, v, c)
            del cached_masks
    
    # Calculate peak memory usage
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    peak_memory_used = peak_memory - peak_memory_before
    
    # Move tensors to CPU for metrics computation
    rendered_cpu = rendered.cpu()
    # Convert to float32 for metrics computation if it's a half precision tensor
    if rendered_cpu.dtype == torch.float16:
        rendered_cpu = rendered_cpu.float()
    I_target_cpu = I_target.cpu()
    if I_target_cpu.dtype == torch.float16:
        I_target_cpu = I_target_cpu.float()
    
    # Clear GPU memory as soon as possible
    del rendered
    torch.cuda.empty_cache()
    
    # Compute metrics
    metrics = compute_metrics(rendered_cpu, I_target_cpu)
    
    # Add runtime and VRAM metrics
    metrics['Runtime'] = runtime  # Now this is just initialization+optimization time
    metrics['VRAM_MB'] = peak_memory_used
    metrics['ImgName'] = os.path.basename(config["preprocessing"].get("img_path", "unknown"))
    metrics['Width'] = W
    metrics['NumPrimitives'] = len(x)
    
    # Clear CPU tensors
    del I_target_cpu
    
    # Export PDF
    output_dir = config["postprocessing"].get("output_folder", "./outputs/")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(config["preprocessing"]["img_path"]))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}__{timestamp}.pdf")

    H, W = config['canvas_size']
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
    
    '''
    exporter.export_with_pngs(x,y,r,theta,v,c,
                output_folder=folder_path,
                svg_hollow=config["svg"].get("svg_hollow", False))
    '''
    
    # Copy parameters to CPU before returning
    params_cpu = (
        x.detach().cpu(), 
        y.detach().cpu(), 
        r.detach().cpu(), 
        v.detach().cpu(), 
        theta.detach().cpu(), 
        c.detach().cpu()
    )
    
    with open(pdf_path, "rb") as f:
        pdf_file = f.read()
    
    res = {
        'initializer': init.__class__.__name__,
        'renderer': renderer_name,
        'rendered': rendered_cpu.numpy(),
        'metrics': metrics,
        'params': params_cpu,
        'pdf_file': pdf_file,
    }
    print("[RUN COMPLETE] {}".format(res))
    
def run(initializers_config, renderer_name, config, I_target, svg_loader, device):
    H, W = config['canvas_size']
    
    # Process each initializer-renderer combination one at a time
    init_name, init_config = initializers_config
    # Create initializer
    if init_name == "StructureAwareInitializer":
        from core.initializer.svgsplat_initializater import StructureAwareInitializer
        initializer = StructureAwareInitializer(init_config)
    elif init_name == "RandomInitializer":
        from core.initializer.random_initializater import RandomInitializer
        initializer = RandomInitializer(init_config)
    elif init_name == "MultiLevelInitializer":
        from core.initializer.multilevel_initializer import MultiLevelInitializer
        initializer = MultiLevelInitializer(init_config)
    else:
        raise ValueError(f"Unsupported initializer: {init_name}")
        
    # Create VectorRenderer only when needed
    bmp_tensor = svg_loader.load_alpha_bitmap()
    if config["optimization"].get("use_fp16", False):
        bmp_tensor = bmp_tensor.to(dtype=torch.float16)
    else:
        bmp_tensor = bmp_tensor.to(dtype=torch.float32)
    
    # Generate initialization parameters
    if "LevelInitializer" in init_name:
        from core.renderer.vector_renderer import VectorRenderer
        vec_renderer = VectorRenderer((H, W), S=bmp_tensor, 
                                alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                                device=device,
                                use_fp16=config["optimization"].get("use_fp16", False))
        params = initializer.initialize(I_tar=I_target, renderer=vec_renderer, opt_conf=config["optimization"])
        del vec_renderer
    else:
        params = initializer.initialize(I_target)
        
    # Clean up temporary renderer
    del bmp_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    # Report memory usage
    current = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    print(f"After initialization: Current={current:.2f}MB, Reserved={reserved:.2f}MB")
        
    # Process each combination and store results
    process_combination((params, initializer, renderer_name, config, I_target, svg_loader))    
    
def main():
    parser = argparse.ArgumentParser(description="Compare different initializers and renderers")
    parser.add_argument('--n_iter', type=int, default=100, help='100, ...')
    parser.add_argument('--n_primitives', type=int, default=3000, help='3000, ...')
    parser.add_argument('--svg_path', type=str, default='', help='LOVE.svg, ...')
    parser.add_argument('--svg_text', type=str, default='', help='G, B, M, ...')
    parser.add_argument('--primitive_type', type=str, default='circle', help='circle, line, ...')
    parser.add_argument('--img_path', type=str, default='', help='images/HighFreq/0831.png, images/LowFreq/0831.png, ...')
    parser.add_argument('--output_path', type=str, default='', help='results/0831.png, results/0831.gif, ...')
    args = parser.parse_args()
       
    # Load configuration
    with open("configs/default_task.json", "r") as f:
        config = json.load(f)
    
    # Process initializers one at a time to save memory
    initializers_config = ("StructureAwareInitializer", config["initialization"])
    renderer_name = "MseRenderer"
    
    config["optimization"]["num_iterations"] = args.n_iter    
    config["initialization"]["N"] = args.n_primitives
    
    if args.svg_path != '':
        config["svg"]["svg_file"] = args.svg_path        
    elif args.svg_text != '':
        config["svg"]["text"] = args.svg_text
    else:
        config["svg"]["text"] = args.primitive_type #[TODO]
        
    config["preprocessing"]["img_path"] = args.img_path
    print(f"img_path: {args.img_path}")
    config["postprocessing"]["output_folder"] = args.output_path
    
    print(f"Using FP16 (half precision): {config['optimization'].get('use_fp16', False)}")
    
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
        
    # Load and preprocess target image
    I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
    # Convert to appropriate precision based on config
    if config["optimization"].get("use_fp16", False):
        I_target = torch.tensor(I_target, dtype=torch.float16).to(device)
    else:
        I_target = torch.tensor(I_target, dtype=torch.float32).to(device)
        
    H = preprocessor.final_height
    W = preprocessor.final_width
    config['canvas_size'] = (H, W)
    
    run(initializers_config, renderer_name, config, I_target, svg_loader, device)

if __name__ == '__main__':
    main() 
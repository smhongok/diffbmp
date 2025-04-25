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

from core.preprocessing import Preprocessor
from util.svg_loader import SVGLoader
from util.font_to_svg import FontParser
from util.utils import set_global_seed, gaussian_blur
from util.pdf_exporter import PDFExporter

from core.initializer.random_initializater import RandomInitializer
from core.initializer.svgsplat_initializater import StructureAwareInitializer

from core.renderer.vector_renderer import VectorRenderer
from core.renderer.mse_renderer import MseRenderer
from core.renderer.lpips_renderer import LpipsRenderer
from core.renderer.mix_renderer import MixRenderer

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute image quality metrics."""
    # Convert to correct format (NCHW)
    pred_t = pred.permute(2, 0, 1).unsqueeze(0)
    target_t = target.permute(2, 0, 1).unsqueeze(0)
    
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
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'VIF': vif,
        'LPIPS': lpips
    }

def plot_results(results: List[Dict[str, Any]], save_path: str, target_image: np.ndarray):
    """Plot results in a 2x3 grid with metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3)
    
    # Add target image as a small inset in the top-right corner
    ax_inset = fig.add_axes([0.85, 0.85, 0.1, 0.1])
    ax_inset.imshow(target_image)
    ax_inset.axis('off')
    ax_inset.set_title('Target', fontsize=8)
    
    for idx, result in enumerate(results):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Plot image
        ax.imshow(result['rendered'])
        ax.axis('off')
        
        # Create title with metrics
        title = f"{result['initializer']}/{result['renderer']}\n"
        metrics = result['metrics']
        title += f"PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.2f}\n"
        title += f"VIF: {metrics['VIF']:.2f}, LPIPS: {metrics['LPIPS']:.2f}"
        ax.set_title(title, fontsize=8)
    
    # Save plots
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()

def process_combination(args):
    """Process a single initializer-renderer combination."""
    init, renderer, config, I_target, bmp_tensor, svg_path, svg_loader = args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move tensors to device
    I_target = I_target.to(device)
    bmp_tensor = bmp_tensor.to(device)
    
    print(f"\nUsing {init.__class__.__name__} with {renderer.__class__.__name__}")
    
    # Get initial parameters
    initial_params = init.initialize(I_target)
    
    # Reset parameters to initial values
    x, y, r, v, theta, c = [t.clone().detach().requires_grad_(True) for t in initial_params]
    
    # Optimize
    x, y, r, v, theta, c = renderer.optimize_parameters(
        x, y, r, v, theta, c,
        I_target, bmp_tensor,
        config['optimization']
    )
    
    # Generate final render
    cached_masks = renderer._batched_soft_rasterize(
        bmp_tensor, x, y, r, theta,
        sigma=config["optimization"].get("blur_sigma_end", 1.0)
    )
    rendered = renderer.render(cached_masks, v, c)
    
    # Compute metrics
    metrics = compute_metrics(rendered.cpu(), I_target.cpu())
    
    # Export PDF
    output_dir = config["postprocessing"].get("output_folder", "./outputs/")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = os.path.join(output_dir, 
                           f"{init.__class__.__name__}_{renderer.__class__.__name__}_{timestamp}.pdf")
    
    H, W = config['canvas_size']
    exporter = PDFExporter(
        svg_path, 
        canvas_size=(W, H),
        viewbox_size=svg_loader.get_svg_size(),
        alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5),
        stroke_width=config["postprocessing"].get("linewidth", 3.0)
    )
    
    exporter.export(x, y, r, theta, v, c,
                   output_path=pdf_path,
                   svg_hollow=config["svg"].get("svg_hollow", False))
    
    return {
        'initializer': init.__class__.__name__,
        'renderer': renderer.__class__.__name__,
        'rendered': rendered.detach().cpu().numpy(),
        'metrics': metrics,
        'params': (x, y, r, v, theta, c)
    }

def main():
    parser = argparse.ArgumentParser(description="Compare different initializers and renderers")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    
    # Set random seed
    set_global_seed(config.get("seed", 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize preprocessor
    pp_conf = config["preprocessing"]
    preprocessor = Preprocessor(
        final_width=pp_conf.get("final_width", 128),
        trim=pp_conf.get("trim", False),
        FM_halftone=pp_conf.get("FM_halftone", False),
        transform_mode=pp_conf.get("transform", "none"),
    )
    
    # Load and preprocess target image
    I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
    I_target = torch.tensor(I_target).to(device)
    H = preprocessor.final_height
    W = preprocessor.final_width
    config['canvas_size'] = (H, W)
    
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
    bmp_tensor = svg_loader.load_alpha_bitmap()
    
    # Create instances of initializers
    initializers = [
        RandomInitializer(
            num_init=config["initialization"].get("N", 10000),
            alpha=config["initialization"].get("alpha", 0.3),
            min_distance=config["initialization"].get("min_distance", 5),
            peak_threshold=config["initialization"].get("peak_threshold", 0.5),
            radii_min=config["initialization"].get("radii_min", 2),
            radii_max=config["initialization"].get("radii_max", None),
            v_init_bias=config["initialization"].get("v_init_bias", -5.0),
            v_init_slope=config["initialization"].get("v_init_slope", 10.0),
            keypoint_extracting=config["initialization"].get("keypoint_extracting", False),
            debug_mode=config["initialization"].get("debug_mode", False)
        ),
        StructureAwareInitializer(
            num_init=config["initialization"].get("N", 10000),
            alpha=config["initialization"].get("alpha", 0.3),
            min_distance=config["initialization"].get("min_distance", 5),
            peak_threshold=config["initialization"].get("peak_threshold", 0.5),
            radii_min=config["initialization"].get("radii_min", 2),
            radii_max=config["initialization"].get("radii_max", None),
            v_init_bias=config["initialization"].get("v_init_bias", -5.0),
            v_init_slope=config["initialization"].get("v_init_slope", 10.0),
            keypoint_extracting=config["initialization"].get("keypoint_extracting", False),
            debug_mode=config["initialization"].get("debug_mode", False)
        )
    ]
    
    # Create instances of renderers
    renderers = [
        MseRenderer((H, W), alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), device=device),
        LpipsRenderer((H, W), alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), device=device),
        MixRenderer((H, W), alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), device=device)
    ]
    
    results = [process_combination((init, renderer, config, I_target, bmp_tensor, svg_path, svg_loader))
                for init, renderer in product(initializers, renderers)]
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(config["postprocessing"].get("output_folder", "./outputs/"),
                            f'comparison_{timestamp}')
    
    # Plot and save results
    plot_results(results, save_path, I_target.numpy() if args.multi_process else I_target.cpu().numpy())

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    formatted_time = str(timedelta(seconds=int(end_time - start_time)))
    # 수행 시간 출력
    print(f"total_cost_time: {formatted_time}")
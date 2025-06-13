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

def plot_results(results: List[Dict[str, Any]], save_path: str, target_image: np.ndarray, config: Dict[str, Any]):
    """Plot results in a grid with metrics."""
    print("===== VISUALIZATION START =====")
    
    # Report memory usage
    if torch.cuda.is_available():
        print(f"CUDA memory at visualization start: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Convert target_image to numpy array if it's a tensor
    if isinstance(target_image, torch.Tensor):
        target_image = target_image.float().cpu().numpy()
    
    print(f"Number of results: {len(results)}")
    
    # Determine debug mode
    debug_mode = config.get("initialization", {}).get("debug_mode", False)
    
    # Determine grid layout
    num_results = len(results)
    
    if debug_mode:
        # In debug mode, we need space for point visualizations
        # Each initializer gets a row with points in col 0 and renders in remaining columns
        initializers = set(r['initializer'] for r in results)
        num_initializers = len(initializers)
        renderers_per_initializer = len(results) // num_initializers
        
        rows = num_initializers
        cols = renderers_per_initializer + 1  # +1 for point visualization
    else:
        # Regular layout: just fit all results in a grid
        if num_results <= 3:
            rows, cols = 1, num_results
        elif num_results == 4:
            rows, cols = 2, 2
        elif num_results <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
    
    print(f"Layout: {rows} rows x {cols} columns, debug_mode={debug_mode}")
    
    # Create figure
    fig_width = 5 * cols
    fig_height = 5 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=72)
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.85)
    
    # Normalize axes to 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    if config["optimization"].get("use_fp16", False):
        target_image = target_image.astype(np.float32)

    # Add configuration info
    try:
        preproc_conf = config["preprocessing"]
        init_conf = config["initialization"]
        opt_conf = config["optimization"]
        
        config_line1 = f'final_width={preproc_conf.get("final_width", "N/A")}, N={init_conf.get("N", "N/A")}, iterations={opt_conf.get("num_iterations", "N/A")}, ' \
                    f'gaussian_blur={opt_conf.get("do_gaussian_blur", False)}, ' \
                    f'sparsify={opt_conf.get("sparsifying", {}).get("do_sparsify", False)}, ' \
                    f'sparsified_N={opt_conf.get("sparsifying", {}).get("sparsified_N", "N/A")}'
        
        lr_conf = opt_conf.get("learning_rate", {})
        config_line2 = f'learning_rate: default={lr_conf.get("default", "N/A")}, ' \
                     f'x={lr_conf.get("gain_x", "N/A")}, r={lr_conf.get("gain_r", "N/A")}, ' \
                     f'v={lr_conf.get("gain_v", "N/A")}, c={lr_conf.get("gain_c", "N/A")}'
        
        plt.figtext(0.5, 0.96, config_line1, ha='center', va='center', fontsize=8)
        plt.figtext(0.5, 0.94, config_line2, ha='center', va='center', fontsize=8)
    except Exception as e:
        print(f"Error adding configuration info: {e}")
    
    # Add target image as small inset in top-right corner
    try:
        ax_inset = fig.add_axes([0.85, 0.85, 0.1, 0.1])
        ax_inset.imshow(target_image)
        ax_inset.axis('off')
        ax_inset.set_title('Target', fontsize=8)
    except Exception as e:
        print(f"Error adding target image inset: {e}")
    
    # Create a mapping of initializers to their row indices
    if debug_mode:
        initializer_rows = {}
        for i, initializer in enumerate(sorted(initializers)):
            initializer_rows[initializer] = i
    
    # Visualize results
    for idx, result in enumerate(results):
        initializer = result['initializer']
        renderer = result['renderer']
        print(f"Processing: Result {idx} ({initializer}/{renderer})")
        
        # Convert rendered image data to numpy
        rendered_img = result['rendered']
        if isinstance(rendered_img, torch.Tensor):
            rendered_img = rendered_img.float().detach().cpu().numpy()
        
        # Ensure image is float64 for display
        if rendered_img.dtype != np.float32:
            rendered_img = rendered_img.astype(np.float32)
        
        # Replace any NaN/Inf values
        if np.isnan(rendered_img).any() or np.isinf(rendered_img).any():
            rendered_img = np.nan_to_num(rendered_img)
        
        # Determine subplot position
        if debug_mode:
            # In debug mode, each initializer gets a row
            row = initializer_rows[initializer]
            
            # First find how many renderers we've seen for this initializer so far
            renderer_idx = sum(1 for r in results[:idx] if r['initializer'] == initializer)
            
            # Place in column 1 + renderer_idx (column 0 reserved for points)
            col = 1 + renderer_idx
        else:
            # In regular mode, just place in grid order
            row = idx // cols
            col = idx % cols
        
        print(f"Placing {initializer}/{renderer} at grid position [{row},{col}]")
        
        try:
            # Display image
            ax = axes[row, col]
            ax.imshow(rendered_img)
            ax.axis('off')
            
            # Create title with metrics
            metrics = result['metrics']
            title = f"{initializer}/{renderer}\n"
            title += f"PSNR: {metrics['PSNR']:.3f}, SSIM: {metrics['SSIM']:.3f}\n"
            title += f"VIF: {metrics['VIF']:.3f}, LPIPS: {metrics['LPIPS']:.3f}"
            ax.set_title(title, fontsize=8)
        except Exception as e:
            print(f"Error displaying image at [{row},{col}]: {e}")
        
        # Add point visualization in debug_mode
        if debug_mode and initializer:
            # We only need to do this once per initializer
            if all(r['initializer'] != initializer for r in results[:idx]):
                print(f"Adding point visualization for {initializer}")
                try:
                    # Extract parameters
                    x, y, r, v, theta, c = result['params']

                    # Convert to numpy 
                    x_np = x.float().detach().cpu().numpy()
                    y_np = y.float().detach().cpu().numpy()
                    
                    # Get the point visualization subplot
                    point_ax = axes[row, 0]
                    
                    # Display target image with transparency
                    point_ax.imshow(target_image, alpha=0.5)
                    
                    # Add points on top
                    point_ax.scatter(x_np, y_np, c='red', s=2, alpha=0.8)
                    point_ax.set_title(f"{initializer} Points", fontsize=8)
                    point_ax.axis('off')
                    
                    # Save individual point visualization
                    result_path = f"{save_path}_{initializer}_{renderer}"
                    
                    # Create separate figure for points
                    fig_sep = plt.figure(figsize=(10, 10))
                    plt.imshow(target_image, alpha=0.5)
                    plt.scatter(x_np, y_np, c='red', s=2, alpha=0.8)
                    plt.title(f"{initializer} Points")
                    plt.axis('off')
                    plt.savefig(f"{result_path}_points.png", dpi=150, bbox_inches='tight')
                    plt.close(fig_sep)
                    print(f"Points visualization saved: {result_path}_points.png")
                except Exception as e:
                    print(f"Error in point visualization: {e}")
    
    # Hide empty subplots
    for row in range(rows):
        for col in range(cols):
            try:
                # Skip if this axes is out of bounds
                if row >= axes.shape[0] or col >= axes.shape[1]:
                    continue
                
                # Check if this subplot was not used
                if not axes[row, col].has_data():
                    axes[row, col].axis('off')
            except Exception as e:
                print(f"Error hiding subplot at [{row},{col}]: {e}")
    
    # Save outputs
    try:
        # Save PNG (high resolution)
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        print(f"PNG saved: {save_path}.png")
        
        # Save PDF
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
        print(f"PDF saved: {save_path}.pdf")
    except Exception as e:
        print(f"Error saving outputs: {e}")
        try:
            # Try alternative saving methods
            plt.savefig(f"{save_path}_low_dpi.pdf", format='pdf', dpi=72, bbox_inches='tight')
            print(f"Low DPI PDF saved: {save_path}_low_dpi.pdf")
        except Exception:
            print("Failed to save PDF with alternative method")
    
    plt.close(fig)
    print("===== VISUALIZATION COMPLETE =====")

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
    
    # Export PDF
    output_dir = config["postprocessing"].get("output_folder", "./outputs/")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = os.path.join(output_dir, 
                        f"{init.__class__.__name__}_{renderer_name}_{timestamp}.pdf")
    folder_path = os.path.join(output_dir,
                        f"{init.__class__.__name__}_{renderer_name}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)

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
                rendered = renderer.render_export_mp4(cached_masks, v, c, video_path=folder_path+".mp4")
                del cached_masks
    else:
        with torch.no_grad():  # Disable gradient computation for final render
            cached_masks = renderer._batched_soft_rasterize(
                x, y, r, theta,
                sigma=0.
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
    
    # exporter.export_with_pngs(x,y,r,theta,v,c,
    #             output_folder=folder_path,
    #             svg_hollow=config["svg"].get("svg_hollow", False))
    
    # Copy parameters to CPU before returning
    params_cpu = (
        x.detach().cpu(), 
        y.detach().cpu(), 
        r.detach().cpu(), 
        v.detach().cpu(), 
        theta.detach().cpu(), 
        c.detach().cpu()
    )
    
    # Clear GPU tensors and explicitly delete renderer
    del x, y, r, v, theta, c
    del renderer
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'initializer': init.__class__.__name__,
        'renderer': renderer_name,
        'rendered': rendered_cpu.numpy(),
        'metrics': metrics,
        'params': params_cpu
    }
    
def run_comparison(initializers_configs, renderer_names, config, I_target, svg_loader, device):
    H, W = config['canvas_size']
    results = []
    
    # Process each initializer-renderer combination one at a time
    for init_name, init_config in initializers_configs:
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
        elif init_name == "SequentialInitializer":
            from core.initializer.sequnetial_initializater import SequentialInitializer
            initializer = SequentialInitializer(init_config)
        else:
            continue  # Skip unsupported initializer
        
        # Create VectorRenderer only when needed
        bmp_tensor = svg_loader.load_alpha_bitmap()
        if config["optimization"].get("use_fp16", False):
            bmp_tensor = bmp_tensor.to(dtype=torch.float16)
        else:
            bmp_tensor = bmp_tensor.to(dtype=torch.float32)
            
        from core.renderer.vector_renderer import VectorRenderer
        vec_renderer = VectorRenderer((H, W), S=bmp_tensor, 
                                    alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                                    device=device,
                                    use_fp16=config["optimization"].get("use_fp16", False))
        
        # Generate initialization parameters
        if "LevelInitializer" in init_name:
            params = initializer.initialize(I_tar=I_target, renderer=vec_renderer, opt_conf=config["optimization"])
        else:
            params = initializer.initialize(I_target)
            
        # Clean up temporary renderer
        del vec_renderer, bmp_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        # Report memory usage
        current = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"After initialization: Current={current:.2f}MB, Reserved={reserved:.2f}MB")
        
        # Process each renderer
        for renderer_idx, renderer_name in enumerate(renderer_names):
            # Thorough memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Process each combination and store results
            try:
                result = process_combination((params, initializer, renderer_name, config, I_target, svg_loader))
                results.append(result)
                
                # Explicit memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
                
                # Report memory usage
                current = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                print(f"After {init_name} + {renderer_name}: Current={current:.2f}MB, Reserved={reserved:.2f}MB")
            
            except Exception as e:
                print(f"Error processing {init_name} with {renderer_name}: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
            
            # Periodic memory cleanup
            if renderer_idx % 2 == 1:
                # Force memory cache clearing
                torch.cuda.empty_cache()
                gc.collect()
                
                # Synchronize CUDA events to ensure GPU tasks are completed
                torch.cuda.synchronize()
        
    # Clean up all initializers
    del initializer, params
    torch.cuda.empty_cache()
    gc.collect()
    
    # Exit if no valid results
    if not results:
        print("No valid results were generated. Exiting.")
        return
    
    # Move I_target to CPU to minimize memory usage
    I_target_cpu = I_target.cpu()
    del I_target
    torch.cuda.empty_cache()
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(config["postprocessing"].get("output_folder", "./outputs/"),
                            f'comparison_{timestamp}')
    
    # Plot and save results
    try:
        plot_results(results, save_path, I_target_cpu.numpy(), config)
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()
    
    # Save metrics to Excel
    try:
        excel_path = save_metrics_to_excel(results, config)
        print(f"Metrics saved to Excel: {excel_path}")
    except Exception as e:
        print(f"Error saving metrics to Excel: {e}")
        import traceback
        traceback.print_exc()
    
    print("Comparison complete!")
    
def save_metrics_to_excel(results, config):
    """Save metrics from all runs to an Excel file"""
    if not results:
        print("No results to save to Excel")
        return
        
    # Create a DataFrame from the results
    data = []
    for result in results:
        metrics = result.get('metrics', {})
        row = {
            'Image': metrics.get('ImgName', os.path.basename(config["preprocessing"].get("img_path", "unknown"))),
            'Primitives': metrics.get('NumPrimitives', config["initialization"].get("N", 0)),
            'Initializer': result.get('initializer', 'unknown'),
            'Renderer': result.get('renderer', 'unknown'),
            'blur_sigma': config["optimization"].get("blur_sigma", 1.0),
            'PSNR': metrics.get('PSNR', 0),
            'SSIM': metrics.get('SSIM', 0),
            'LPIPS': metrics.get('LPIPS', 0),
            'VIF': metrics.get('VIF', 0),
            'Runtime (s)': metrics.get('Runtime', 0),
            'VRAM (MB)': metrics.get('VRAM_MB', 0)
        }
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config["postprocessing"].get("output_folder", "./outputs/")
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, f'metrics_{timestamp}.xlsx')
    
    # Save to Excel
    df.to_excel(excel_path, index=False)
    print(f"Metrics saved to {excel_path}")
    
    # Return path to Excel file
    return excel_path

def main():
    parser = argparse.ArgumentParser(description="Compare different initializers and renderers")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--initializer', type=str, default='', help='StructureAwareInitializer, RandomInitializer, MultiLevelInitializer, ...')
    parser.add_argument('--renderer', type=str, default='', help='MseRenderer, ...')
    parser.add_argument('--svg_text', type=str, default='', help='G, B, M, ...')
    parser.add_argument('--svg_path', type=str, default='', help='LOVE.svg, ...')
    parser.add_argument('--img_path', type=str, default='', help='images/HighFreq/0831.png, images/LowFreq/0831.png, ...')
    args = parser.parse_args()
       
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    if args.initializer != '':
        initializers_configs = [(init, config["initialization"]) for init in args.initializer.split(",")]
    else:
        # Process initializers one at a time to save memory
        initializers_configs = [
            ("StructureAwareInitializer", config["initialization"]),
            # Add more initializers as needed
        ]
    if args.renderer != '':
        renderer_names = args.renderer.split(",")
    else:
        renderer_names = ["MseRenderer"] # Add more renderers as needed
    
    if args.svg_path != '':
        config["svg"]["svg_file"] = args.svg_path
        config["postprocessing"]["output_folder"] = config["postprocessing"]["output_folder"].replace("outputs/", "outputs/" + args.svg_path.split("/")[-1].split(".")[0] + "-")
    elif args.svg_text != '':
        config["svg"]["text"] = args.svg_text
        config["postprocessing"]["output_folder"] = config["postprocessing"]["output_folder"].replace("outputs/", "outputs/" + args.svg_text + "-")
    
    if args.img_path != '':
        config["preprocessing"]["img_path"] = args.img_path
    
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
            svg_path = [str(font_parser.text_to_svg(t, mode="opt-path")) for t in texts]
        else:
            svg_path = str(font_parser.text_to_svg(texts, mode="opt-path"))
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
        
    orig_final_width = config["preprocessing"].get("final_width", 128)
    if type(orig_final_width) is list:
        final_width_list = orig_final_width
    else:
        final_width_list = [orig_final_width]
    
    orig_N = config["initialization"].get("N", 1000)
    if type(orig_N) is list:
        N_list = orig_N
    else:
        N_list = [orig_N]
        
    orig_img_path = config["preprocessing"].get("img_path", "images/HighFreq/0831.png")
    if type(orig_img_path) is list:
        img_path_list = orig_img_path
    else:
        img_path_list = [orig_img_path]
        
    orig_output_folder = config["postprocessing"].get("output_folder", "./outputs/")
    
    for final_width in final_width_list:
        config["preprocessing"]["final_width"] = final_width
        print(f"final_width: {final_width}")
        print(f"type(final_width): {type(final_width)}")
        
        preprocessor.final_width = final_width
        for img_path in img_path_list:
            print(f"img_path: {img_path}")
            print(f"type(img_path): {type(img_path)}")
            config["preprocessing"]["img_path"] = img_path
            
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
            
            for N in N_list:
                config["initialization"]["N"] = N
                config["postprocessing"]["output_folder"] = os.path.join(orig_output_folder, img_path.split("/")[-2], "width" + str(final_width), "N" + str(N))
                run_comparison(initializers_configs, renderer_names, config, I_target, svg_loader, device)
                
                torch.cuda.empty_cache()
                gc.collect()
        
        del I_target
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main() 

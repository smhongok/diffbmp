import argparse
import time
import torch
from torch.cuda.amp import GradScaler, autocast
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

from core.preprocessing import Preprocessor
from util.svg_loader import SVGLoader
from util.font_to_svg import FontParser
from util.utils import set_global_seed, gaussian_blur
from util.pdf_exporter import PDFExporter

#from core.initializer.random_initializater import RandomInitializer
from core.initializer.svgsplat_initializater import StructureAwareInitializer
#from core.initializer.multilevel_initializer import MultiLevelInitializer
from core.initializer.base_initializer import BaseInitializer

from core.renderer.vector_renderer import VectorRenderer
from core.renderer.mse_renderer import MseRenderer
from core.renderer.lpips_renderer import LpipsRenderer
from core.renderer.mix_renderer import MixRenderer

# Enable gradient checkpointing for memory efficiency
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

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
    """Plot results in a 2x3 grid with metrics."""
    print("===== VISUALIZATION START =====")
    
    # Report memory usage
    if torch.cuda.is_available():
        print(f"CUDA memory at visualization start: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Convert target_image data (always to numpy float64)
    print("Processing target image...")
    try:
        if isinstance(target_image, torch.Tensor):
            print(f"  Target image type: torch.Tensor, dtype={target_image.dtype}")
            target_image = target_image.float().cpu().numpy()
        
        if target_image.dtype != np.float64:
            print(f"  Converting target image type: {target_image.dtype} -> float64")
            target_image = target_image.astype(np.float64)
        print(f"  Final target image shape: {target_image.shape}, dtype: {target_image.dtype}")
    except Exception as e:
        print(f"Error processing target image: {e}")
        import traceback
        traceback.print_exc()
    
    print("Number of results: ", len(results))
    
    # Check data types in results (for debugging)
    for i, result in enumerate(results[:1]):  # Only check first result
        print(f"Result {i} data types:")
        print(f"  initializer: {result['initializer']}")
        print(f"  renderer: {result['renderer']}")
        rendered = result['rendered']
        if isinstance(rendered, torch.Tensor):
            print(f"  rendered: torch.Tensor, shape={rendered.shape}, dtype={rendered.dtype}")
        else:
            print(f"  rendered: {type(rendered)}, shape={rendered.shape}, dtype={rendered.dtype}")
        
        # Check params
        for j, param_name in enumerate(['x', 'y', 'r', 'v', 'theta', 'c']):
            param = result['params'][j]
            if isinstance(param, torch.Tensor):
                print(f"  {param_name}: torch.Tensor, shape={param.shape}, dtype={param.dtype}")
    
    # Determine rows and columns
    num_results = len(results)
    debug_mode = config.get("initialization", {}).get("debug_mode", False)
    
    if num_results <= 3:
        rows, cols = 1, num_results + (1 if debug_mode else 0)
    elif num_results == 4:
        rows, cols = 2, 2 + (1 if debug_mode else 0)
    elif num_results <= 6:
        rows, cols = 2, 3 + (1 if debug_mode else 0)
    else:
        rows, cols = 3, 3 + (1 if debug_mode else 0)
    
    print(f"Layout: {rows} rows x {cols} columns, debug_mode={debug_mode}")
    
    # Create figure (start with small size for memory efficiency)
    fig_width = 5 * cols
    fig_height = 5 * rows
    print(f"Figure size: {fig_width}x{fig_height}")
    
    # Start with low dpi for memory efficiency
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=72)
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.85)
    
    # Normalize grid structure
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Add configuration info
    print("Adding configuration information...")
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
        print("  Configuration text added")
    except Exception as e:
        print(f"Error adding configuration info: {e}")
    
    # Add target image as small inset in top-right corner
    print("Adding target image inset...")
    try:
        ax_inset = fig.add_axes([0.85, 0.85, 0.1, 0.1])
        ax_inset.imshow(target_image)
        ax_inset.axis('off')
        ax_inset.set_title('Target', fontsize=8)
        print("  Target image inset added")
    except Exception as e:
        print(f"Error adding target image inset: {e}")
    
    # Track which initializer we're processing
    current_initializer = None
    current_row = 0
    
    # Visualize results
    print("Starting result visualization...")
    
    try:
        for idx, result in enumerate(results):
            print(f"\nProcessing: Result {idx} ({result['initializer']}/{result['renderer']})")
            
            # Determine row and column
            row = idx // cols if not debug_mode else idx // (cols - 1)
            col = idx % cols if not debug_mode else (idx % (cols - 1)) + 1
            
            # Check for new initializer in debug_mode
            if result['initializer'] != current_initializer:
                current_initializer = result['initializer']
                current_row = row
                print(f"  New initializer: {current_initializer}, row={current_row}")
            
            # Convert rendered image data
            print(f"  Converting image data...")
            rendered_img = result['rendered']
            if isinstance(rendered_img, torch.Tensor):
                print(f"    Converting tensor to numpy, dtype={rendered_img.dtype}")
                rendered_img = rendered_img.float().detach().cpu().numpy()
            
            if rendered_img.dtype != np.float64:
                print(f"    Converting dtype: {rendered_img.dtype} -> float64")
                rendered_img = rendered_img.astype(np.float64)
            
            print(f"    Final image: shape={rendered_img.shape}, dtype={rendered_img.dtype}")
            
            # Check for NaN or infinity values that could break PDF
            if np.isnan(rendered_img).any() or np.isinf(rendered_img).any():
                print("    !!WARNING!! Image contains NaN or infinity values. Replacing with zeros.")
                rendered_img = np.nan_to_num(rendered_img)
            
            # Select subplot
            print(f"  Selecting subplot: axes[{row},{col}]")
            try:
                ax = axes[row, col]
            except IndexError:
                print(f"    Error: axes[{row},{col}] index out of range. Array shape: {axes.shape}")
                continue
            
            # Display image
            print("  Displaying image...")
            try:
                ax.imshow(rendered_img)
                ax.axis('off')
                
                # Create title with metrics
                metrics = result['metrics']
                title = f"{result['initializer']}/{result['renderer']}\n"
                title += f"PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.2f}\n"
                title += f"VIF: {metrics['VIF']:.2f}, LPIPS: {metrics['LPIPS']:.2f}"
                ax.set_title(title, fontsize=8)
                print("    Image and title displayed")
            except Exception as e:
                print(f"    Error displaying image: {e}")
                import traceback
                traceback.print_exc()
            
            # Add point visualization in debug_mode
            if debug_mode and row == current_row and col == 1:
                print("  Adding point visualization...")
                try:
                    # Extract parameters
                    x, y, r, v, theta, c = result['params']
                    
                    # Convert to numpy (float64)
                    print("    Converting coordinate data...")
                    x_np = x.float().detach().cpu().numpy().astype(np.float64)
                    y_np = y.float().detach().cpu().numpy().astype(np.float64)
                    
                    # Create initial points array
                    init_pts = np.column_stack((x_np, y_np))
                    
                    # Point visualization with semi-transparent background
                    point_ax = axes[row, 0]
                    
                    # First display the target image with transparency
                    point_ax.imshow(target_image, alpha=0.5)  # Semi-transparent background
                    
                    # Then add points on top
                    point_ax.scatter(x_np, y_np, c='red', s=2, alpha=0.8)
                    point_ax.set_title(f"{result['initializer']} Points", fontsize=8)
                    point_ax.axis('off')
                    print("    Point visualization completed")
                    
                    # Save individual point visualization files
                    print("    Saving individual point visualization files...")
                    result_path = f"{save_path}_{result['initializer']}_{result['renderer']}"
                    
                    densified_pts = np.array([])
                    adjusted_pts = np.array([])
                    
                    # Create a custom visualization for points with transparent background
                    fig_sep = plt.figure(figsize=(10, 10))
                    plt.imshow(target_image, alpha=0.5)  # Semi-transparent background
                    plt.scatter(x_np, y_np, c='red', s=2, alpha=0.8)
                    plt.title(f"{result['initializer']} Points")
                    plt.axis('off')
                    plt.savefig(f"{result_path}_points.png", dpi=150, bbox_inches='tight')
                    plt.close(fig_sep)
                    print(f"    Points visualization saved: {result_path}_points.png")
                    
                    # Also save as PDF
                    print("    Saving point visualization PDF...")
                    try:
                        fig_points = plt.figure(figsize=(10, 10))
                        plt.imshow(target_image, alpha=0.5)  # Semi-transparent background
                        plt.scatter(x_np, y_np, c='red', s=2, alpha=0.8)
                        plt.title(f"{result['initializer']}/{result['renderer']} Points")
                        plt.axis('off')
                        plt.savefig(f"{result_path}_points.pdf", format='pdf', bbox_inches='tight')
                        plt.close(fig_points)
                        print(f"    PDF saved: {result_path}_points.pdf")
                    except Exception as e:
                        print(f"    Error saving point PDF: {e}")
                except Exception as e:
                    print(f"  Error in point visualization: {e}")
        
        # Hide empty subplots
        print("\nHiding empty subplots...")
        for row in range(rows):
            for col in range(cols):
                try:
                    if row >= len(axes) or col >= len(axes[row]):
                        print(f"  Warning: axes[{row},{col}] index out of range. Skipping.")
                        continue
                    
                    # Check if this subplot was not used
                    if not axes[row, col].has_data():
                        print(f"  Hiding: axes[{row},{col}]")
                        axes[row, col].axis('off')
                except Exception as e:
                    print(f"  Error hiding subplot (axes[{row},{col}]): {e}")
        
        # Save PNG (high resolution)
        print("\nSaving PNG...")
        try:
            plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
            print(f"PNG saved: {save_path}.png")
        except Exception as e:
            print(f"Error saving PNG: {e}")
        
        # Prepare for PDF saving
        print("\nPreparing for PDF saving...")
        
        # Additional safety checks for PDF compatibility
        for row in range(rows):
            for col in range(cols):
                if row < len(axes) and col < len(axes[row]):
                    ax = axes[row, col]
                    
                    # Check image data is float64
                    for img in ax.get_images():
                        if img.get_array() is not None:
                            arr = img.get_array()
                            if arr.dtype != np.float64:
                                print(f"  Converting axes[{row},{col}] image to float64")
                                img.set_array(arr.astype(np.float64))
                            
                            # Check for NaN or infinity values
                            if np.isnan(arr).any() or np.isinf(arr).any():
                                print(f"  Removing NaN/Inf values from axes[{row},{col}] image")
                                img.set_array(np.nan_to_num(arr))
                    
                    # Check scatter plot data is float64
                    for collection in ax.collections:
                        if hasattr(collection, '_offsets') and collection._offsets is not None:
                            offsets = collection._offsets.data
                            if offsets.dtype != np.float64:
                                print(f"  Converting axes[{row},{col}] scatter data to float64")
                                collection._offsets = np.array(offsets, dtype=np.float64)
        
        # Try safer PDF saving approach
        print("Attempting to save PDF...")
        try:
            # Direct PDF save attempt
            plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
            print(f"PDF saved: {save_path}.pdf")
        except Exception as primary_err:
            print(f"Error in primary PDF saving: {primary_err}")
            print("Trying alternative PDF saving methods...")
            
            try:
                # Alternative 1: Lower DPI
                plt.savefig(f"{save_path}_low_dpi.pdf", format='pdf', dpi=72, bbox_inches='tight')
                print(f"Low DPI PDF saved: {save_path}_low_dpi.pdf")
            except Exception as e:
                print(f"Error saving low DPI PDF: {e}")
                
                try:
                    # Alternative 2: Convert from PNG
                    from PIL import Image
                    print("Attempting PNG to PDF conversion...")
                    
                    # Save as PNG then create PDF with PIL
                    png_path = f"{save_path}_for_pdf.png"
                    plt.savefig(png_path, dpi=150, bbox_inches='tight')
                    
                    # Convert PNG to PDF
                    img = Image.open(png_path)
                    img.save(f"{save_path}_from_png.pdf", "PDF", resolution=100.0)
                    print(f"PDF converted from PNG saved: {save_path}_from_png.pdf")
                    
                    # Remove temporary file
                    import os
                    os.remove(png_path)
                except Exception as e2:
                    print(f"Error converting PNG to PDF: {e2}")
        
        print("Closing figure...")
        plt.close(fig)
        print("===== VISUALIZATION COMPLETE =====")
        
    except Exception as e:
        print(f"Unexpected error during visualization: {e}")
        import traceback
        traceback.print_exc()

def process_combination(args):
    """Process a single initializer-renderer combination."""
    initial_params, init, renderer_name, config, I_target, svg_path, svg_loader = args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing {init.__class__.__name__} with {renderer_name}")
    
    # Move tensors to device
    I_target = I_target.to(device)
    
    # Get bitmap tensor for renderer
    bmp_tensor = svg_loader.load_alpha_bitmap().to(dtype=torch.float16)
    H, W = config['canvas_size']
    
    # Create renderer only when needed - defer instantiation
    if renderer_name == "MseRenderer":
        from core.renderer.mse_renderer import MseRenderer
        renderer = MseRenderer((H, W), S=bmp_tensor, 
                            alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                            device=device)
    elif renderer_name == "LpipsRenderer":
        from core.renderer.lpips_renderer import LpipsRenderer
        renderer = LpipsRenderer((H, W), S=bmp_tensor, 
                                alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                                device=device)
    elif renderer_name == "MixRenderer":
        from core.renderer.mix_renderer import MixRenderer
        renderer = MixRenderer((H, W), S=bmp_tensor, 
                            alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                            device=device, 
                            classify_svg=svg_loader.classify_svg())
    else:
        raise ValueError(f"Unknown renderer: {renderer_name}")
    
    # 입력 비트맵을 렌더러에 복사한 후 메모리에서 해제
    del bmp_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    # Enable checkpointing for memory-intensive renderers
    if hasattr(renderer, 'enable_checkpointing'):
        renderer.enable_checkpointing()
        if hasattr(renderer, 'memory_report'):
            renderer.memory_report(f"Initial memory for {renderer_name}")
    
    # Force streaming render for memory-intensive renderers
    if renderer_name in ["LpipsRenderer", "MixRenderer"]:
        if 'optimization' not in config:
            config['optimization'] = {}
        config['optimization']['streaming_render'] = True
        print(f"Enabled streaming render for {renderer_name}")
        
        # LPIPS 및 MixRenderer에 대한 추가 메모리 최적화 설정
        # 그래디언트 누적 (더 작은 배치 처리)
        config['optimization']['gradient_accumulation_steps'] = 4
        
        # 작은 청크 크기로 설정
        config['optimization']['raster_chunk_size'] = 10
        
        # 배치 최적화 비활성화 (메모리 사용량 때문)
        config['optimization']['batch_optimization'] = False
    else:
        # MseRenderer는 상대적으로 가벼우므로 다른 설정 사용
        if 'optimization' not in config:
            config['optimization'] = {}
        config['optimization']['gradient_accumulation_steps'] = 1
        config['optimization']['raster_chunk_size'] = 50
    
    # 체크포인팅과 배치 최적화가 충돌하지 않도록 설정 확인
    if 'optimization' in config and config['optimization'].get('batch_optimization', False):
        if hasattr(renderer, 'enable_checkpointing') and renderer.use_checkpointing:
            # LpipsRenderer와 MixRenderer는 체크포인팅이 중요하므로 배치 최적화를 비활성화
            if renderer_name in ["LpipsRenderer", "MixRenderer"]:
                config['optimization']['batch_optimization'] = False
                print(f"배치 최적화가 {renderer_name}의 체크포인팅과 충돌하므로 배치 최적화를 비활성화합니다.")
            # MseRenderer는 체크포인팅보다 배치 최적화가 더 중요할 수 있음
            else:
                renderer.disable_checkpointing()
                print(f"배치 최적화를 위해 {renderer_name}의 체크포인팅을 비활성화합니다.")
    
    with autocast():
        # Reset parameters to initial values
        x, y, r, v, theta, c = [t.clone().detach().requires_grad_(True) for t in initial_params]
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
                if 'optimization' not in config:
                    config['optimization'] = {}
                config['optimization']['debug_memory'] = True
            
            # Optimize parameters - this is the memory-intensive part
            x, y, r, v, theta, c = renderer.optimize_parameters(
                x, y, r, v, theta, c,
                I_target, 
                opt_conf=config['optimization']
            )
            
            # Memory report after optimization
            if hasattr(renderer, 'memory_report'):
                renderer.memory_report(f"After optimization with {renderer_name}")
            
        # Generate final render
        with torch.no_grad():  # Disable gradient computation for final render
            # Process in smaller chunks to save memory
            opt_conf = config['optimization']
            stream_render = opt_conf.get("streaming_render", False)
            
            if stream_render:
                rendered = renderer._stream_render(
                    x, y, r, theta, v, c,
                    sigma=opt_conf.get("blur_sigma_end", 1.0)
                )
            else:
                cached_masks = renderer._batched_soft_rasterize(
                    x, y, r, theta,
                    sigma=opt_conf.get("blur_sigma_end", 1.0)
                )
                rendered = renderer.render(cached_masks, v, c)
                del cached_masks
        
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
    
    # Clear CPU tensors
    del I_target_cpu
    
    # Export PDF
    output_dir = config["postprocessing"].get("output_folder", "./outputs/")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = os.path.join(output_dir, 
                        f"{init.__class__.__name__}_{renderer_name}_{timestamp}.pdf")
    
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

def main():
    parser = argparse.ArgumentParser(description="Compare different initializers and renderers")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    
    # 메모리 사용량 초기화
    torch.cuda.empty_cache()
    gc.collect()
    
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
    # Convert to half precision here if using FP16 throughout
    I_target = torch.tensor(I_target, dtype=torch.float16).to(device)
    H = preprocessor.final_height
    W = preprocessor.final_width
    config['canvas_size'] = (H, W)
    
    # 전처리기 정리
    del preprocessor
    
    # Handle SVG file loading
    svg_ext = os.path.splitext(config["svg"].get("svg_file"))[1].lower()
    if (svg_ext in (".otf", ".ttf")) and ("text" in config["svg"]):
        font_parser = FontParser(config["svg"].get("svg_file"))
        svg_path = str(font_parser.text_to_svg(config["svg"].get("text"), mode="opt-path"))
        # Font parser 정리
        del font_parser
    else:
        svg_path = config["svg"].get("svg_file", "assets/svg/MaruBuri-Bold_HELLO.svg")
    
    # Load SVG file
    svg_loader = SVGLoader(
        svg_path=svg_path,
        output_width=config["svg"].get("output_width", 128),
        device=device
    )
    classify_svg = svg_loader.classify_svg()
    print(f"SVG is classified as: {classify_svg}")
    
    # Define renderer names - 메모리 사용량이 적은 순서로 정렬
    #renderer_names = ["MseRenderer", "LpipsRenderer", "MixRenderer"]
    renderer_names = ["MseRenderer"]
    
    # Process initializers one at a time to save memory
    initializers_configs = [
        ("StructureAwareInitializer", config["initialization"]),
        # ("RandomInitializer", config["initialization"]),
        # ("MultiLevelInitializer", config["initialization"]),
    ]
    
    results = []
    
    # 각 초기화기와 렌더러 조합에 대해 한 번에 하나씩 처리
    for init_name, init_config in initializers_configs:
        # 초기화기 생성
        if init_name == "StructureAwareInitializer":
            from core.initializer.svgsplat_initializater import StructureAwareInitializer
            initializer = StructureAwareInitializer(init_config)
        # 다른 initializer가 필요한 경우 여기에 추가
        else:
            continue  # 지원하지 않는 initializer 건너뛰기
        
        # 필요한 경우에만 VectorRenderer 생성
        bmp_tensor = svg_loader.load_alpha_bitmap()
        from core.renderer.vector_renderer import VectorRenderer
        vec_renderer = VectorRenderer((H, W), S=bmp_tensor, 
                                    alpha_upper_bound=config["optimization"].get("alpha_upper_bound", 0.5), 
                                    device=device)
        
        # 초기화 매개변수 생성
        if "LevelInitializer" in init_name:
            params = initializer.initialize(I_tar=I_target, renderer=vec_renderer, opt_conf=config["optimization"])
        else:
            params = initializer.initialize(I_target)
            
        # 초기화에 사용된 임시 렌더러 해제
        del vec_renderer, bmp_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        # 메모리 사용량 확인 및 출력
        current = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"After initialization: Current={current:.2f}MB, Reserved={reserved:.2f}MB")
        
        # 각 렌더러에 대해 처리 - 메모리 사용량에 따라 순서 최적화
        for renderer_idx, renderer_name in enumerate(renderer_names):
            # 메모리 정리를 더 철저히 수행
            torch.cuda.empty_cache()
            gc.collect()
            
            # 각 렌더러 조합 처리 및 결과 저장
            try:
                result = process_combination((params, initializer, renderer_name, config, I_target, svg_path, svg_loader))
                results.append(result)
                
                # 메모리 명시적 정리
                torch.cuda.empty_cache()
                gc.collect()
                
                # 메모리 사용량 출력
                current = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                print(f"After {init_name} + {renderer_name}: Current={current:.2f}MB, Reserved={reserved:.2f}MB")
            
            except Exception as e:
                print(f"Error processing {init_name} with {renderer_name}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
            
            # 메모리를 주기적으로 정리하기 위한 강제 수집
            if renderer_idx % 2 == 1:
                # 강제로 메모리 캐시 비우기
                torch.cuda.empty_cache()
                gc.collect()
                
                # CUDA 이벤트 동기화로 GPU 작업 완료 보장
                torch.cuda.synchronize()
        
    # 모든 초기화기 해제
    del initializer, params
    torch.cuda.empty_cache()
    gc.collect()
    
    # 결과가 없으면 종료
    if not results:
        print("No valid results were generated. Exiting.")
        return
    
    # 메모리 사용량 최소화를 위해 I_target을 CPU로 이동
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
    
    # 최종 정리
    del results, I_target_cpu
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    formatted_time = str(timedelta(seconds=int(end_time - start_time)))
    # 수행 시간 출력
    print(f"total_cost_time: {formatted_time}")

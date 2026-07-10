"""
Reusable pipeline for DiffBMP image processing.
Extracted from main.py for use in Gradio demo and other applications.
"""
import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime
from pydiffbmp.core.renderer.simple_tile_renderer import SimpleTileRenderer
from pydiffbmp.util.svg_loader import SVGLoader
from pydiffbmp.util.primitive_loader import PrimitiveLoader
from pydiffbmp.util.svg_converter import FontParser, ImageToSVG
from pydiffbmp.core.initializer.svgsplat_initializater import StructureAwareInitializer
from pydiffbmp.core.initializer.random_initializater import RandomInitializer
from pydiffbmp.core.initializer.designated_initializer import DesignatedInitializer
from pydiffbmp.util.primitive_utils import expand_primitive_wildcards
from pydiffbmp.core.preprocessing import Preprocessor
from pydiffbmp.util.utils import set_global_seed
from pydiffbmp.util.pdf_exporter import PDFExporter
from pydiffbmp.util.constants import apply_constants_to_config


def process_single_image(
    img_path,
    config,
    output_dir="outputs/",
    force_cpu=False,
    disable_cuda_kernel=True
):
    """
    Process a single image with DiffBMP.
    
    Args:
        img_path: Path to input image (or None for text-to-drawing mode)
        config: Configuration dictionary
        output_dir: Output directory for results
        force_cpu: Force CPU usage even if GPU available
        disable_cuda_kernel: Disable custom CUDA kernel, use PyTorch fallback
        
    Returns:
        dict: Results containing output paths and metrics
            {
                'output_path': str,
                'pdf_path': str (if applicable),
                'metrics': dict (if computed)
            }
    """
    # Force PyTorch fallback if requested
    if disable_cuda_kernel:
        os.environ['DIFFBMP_FORCE_PYTORCH'] = '1'
    
    # Set device
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if disable_cuda_kernel:
        print("CUDA custom kernel disabled - using PyTorch fallback")
    
    # Apply configuration defaults
    config = apply_constants_to_config(config)
    set_global_seed(config.get("seed", 42))
    
    # Extract configurations
    pp_conf = config["preprocessing"]
    opt_conf = config["optimization"]
    init_conf = config["initialization"]
    
    # Force FP32 for CPU/PyTorch compatibility
    use_fp16 = False if (force_cpu or disable_cuda_kernel) else opt_conf.get("use_fp16", False)
    if use_fp16 != opt_conf.get("use_fp16", False):
        print(f"Overriding use_fp16: {opt_conf.get('use_fp16', False)} -> {use_fp16}")
        opt_conf["use_fp16"] = use_fp16
    
    exist_bg = pp_conf.get("exist_bg", True)
    
    # Load primitive file
    primitive_file_config = config["primitive"].get("primitive_file")
    primitive_file_config = expand_primitive_wildcards(primitive_file_config)
    
    # Handle primitive loading (simplified version - supports SVG and PNG)
    svg_path = None
    if isinstance(primitive_file_config, list):
        svg_path = []
        for file_item in primitive_file_config:
            file_ext = os.path.splitext(file_item)[1].lower()
            if file_ext == ".svg":
                svg_path.append(os.path.join("assets/svg", file_item))
            elif file_ext in (".png", ".jpg", ".jpeg"):
                svg_path.append(os.path.join("assets/primitives", file_item))
    else:
        primitive_ext = os.path.splitext(primitive_file_config)[1].lower()
        if primitive_ext == ".svg":
            svg_path = os.path.join("assets/svg", primitive_file_config)
        elif primitive_ext in (".png", ".jpg", ".jpeg"):
            svg_path = os.path.join("assets/primitives", primitive_file_config)
        else:
            # Default fallback
            svg_path = "assets/svg/circle.svg"
    
    # Load primitives
    try:
        primitive_loader = PrimitiveLoader(
            primitive_paths=svg_path,
            output_width=config["primitive"]["output_width"],
            device=device,
            bg_threshold=config["primitive"]["bg_threshold"],
            radial_transparency=config["primitive"]["radial_transparency"],
            resampling=config["primitive"]["resampling"]
        )
        svg_loader = primitive_loader
        print(f"Loaded primitives: {len(primitive_loader.primitive_paths)} files")
    except Exception as e:
        print(f"PrimitiveLoader failed, falling back to SVGLoader: {e}")
        svg_loader = SVGLoader(
            svg_path=svg_path,
            output_width=config["primitive"]["output_width"],
            device=device
        )
        primitive_loader = None
    
    # Load target image or create canvas
    is_text_only_mode = (img_path is None)
    
    if is_text_only_mode:
        # Text-to-drawing mode: create white canvas
        H = pp_conf["final_width"]
        W = pp_conf["final_width"]
        I_target = torch.ones((H, W, 3), device=device, dtype=torch.float32)
        target_binary_mask_np = None
        print(f"Created white canvas: {W}x{H}")
    else:
        # Load image
        preprocessor = Preprocessor(
            final_width=pp_conf["final_width"],
            trim=pp_conf.get("trim", False),
            FM_halftone=pp_conf.get("FM_halftone", False),
            transform_mode=pp_conf.get("transform", "none"),
        )
        
        # Update config with current image path
        config["preprocessing"]["img_path"] = img_path
        
        if exist_bg:
            print("Target image has background")
            I_target = preprocessor.load_image_8bit_color(config["preprocessing"]).astype(np.float32) / 255.0
            target_binary_mask_np = None
        else:
            print("Target image has no background, using color and opacity")
            I_target, target_binary_mask_np = preprocessor.load_image_8bit_color_opacity(config["preprocessing"])
            I_target = I_target.astype(np.float32) / 255.0
        
        I_target = torch.tensor(I_target, device=device)
        H = preprocessor.final_height
        W = preprocessor.final_width
    
    # Initialize renderer
    bmp_tensor = svg_loader.load_alpha_bitmap()
    if use_fp16:
        bmp_tensor = bmp_tensor.to(dtype=torch.float16)
    else:
        bmp_tensor = bmp_tensor.to(dtype=torch.float32)
    
    # Extract primitive colors
    if primitive_loader is not None:
        primitive_colors = primitive_loader.get_primitive_color_maps()
        print(f"Extracted primitive colors: {primitive_colors.shape}")
    else:
        num_primitives = bmp_tensor.shape[0] if bmp_tensor.ndim == 3 else 1
        primitive_colors = torch.zeros(num_primitives, 128, 128, 3, device=device)
        print("Using default colors for primitives")
    
    # Create renderer
    renderer = SimpleTileRenderer(
        canvas_size=(H, W),
        S=bmp_tensor,
        alpha_upper_bound=config["optimization"]["alpha_upper_bound"],
        device=device,
        use_fp16=use_fp16,
        output_path=output_dir,
        tile_size=opt_conf["tile_size"],
        sigma=opt_conf["blur_sigma"] if opt_conf.get("do_gaussian_blur", False) else 0.0,
        c_blend=config["optimization"].get("c_blend", 0.0),
        primitive_colors=primitive_colors,
        max_prims_per_pixel=config["initialization"].get("max_prims_per_pixel"),
    )
    print(f"Using SimpleTileRenderer for optimization")
    
    # Initialize parameters
    print("---Initializing vector graphics---")
    if init_conf.get("initializer", "none") == "structure_aware":
        initializer = StructureAwareInitializer(init_conf)
    elif init_conf.get("initializer", "none") == "random":
        initializer = RandomInitializer(init_conf)
    elif init_conf.get("initializer", "none") == "designated":
        initializer = DesignatedInitializer(init_conf)
    else:
        raise ValueError(f"Invalid initializer: {init_conf.get('initializer', 'none')}")
    
    # Generate target mask if no background
    target_binary_mask = None
    if not exist_bg and target_binary_mask_np is not None:
        target_binary_mask = torch.from_numpy(target_binary_mask_np[:,:] > 0).to(device)
    
    # Initialize parameters
    x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target, target_binary_mask)
    
    # Optimize parameters
    print("---Starting optimization---")
    x, y, r, v, theta, c = renderer.optimize_parameters(
        x, y, r, v, theta, c,
        I_target, 
        opt_conf=opt_conf,
        target_binary_mask=target_binary_mask,
        initializer=initializer
    )
    
    if not exist_bg:
        I_target = I_target[..., :3]  # Remove alpha channel if exists
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'output_{timestamp}.png')
    
    # Render final image
    with torch.no_grad():
        bg_color = opt_conf.get("bg_color", "white")
        final_bg = renderer._get_background_for_render(bg_color, export=True)
        
        if renderer.use_fp16:
            try:
                from torch.amp import autocast
                autocast_ctx = autocast(device_type='cuda')
            except ImportError:
                from torch.cuda.amp import autocast
                autocast_ctx = autocast()
            with autocast_ctx:
                rendered, rendered_alpha = renderer.render_from_params(
                    x, y, r, theta, v, c, 
                    return_alpha=True, 
                    I_bg=final_bg, 
                    sigma=0.0, 
                    is_final=True
                )
        else:
            rendered, rendered_alpha = renderer.render_from_params(
                x, y, r, theta, v, c, 
                return_alpha=True, 
                I_bg=final_bg, 
                sigma=0.0, 
                is_final=True
            )
        
        # Save PNG
        rendered_np = rendered.detach().cpu().numpy()
        rendered_np = (rendered_np * 255).astype(np.uint8)
        Image.fromarray(rendered_np).save(output_path)
        print(f"Saved output to: {output_path}")
    
    # Export PDF if applicable (SVG-only primitives)
    pdf_path = None
    if not (primitive_loader and primitive_loader.has_raster_primitives()):
        pdf_path = os.path.join(output_dir, f'output_{timestamp}.pdf')
        try:
            exporter = PDFExporter(
                svg_loader.svg_path, 
                canvas_size=(W, H),
                viewbox_size=svg_loader.get_svg_size(),
                alpha_upper_bound=config["optimization"]["alpha_upper_bound"],
                stroke_width=config["postprocessing"].get("linewidth", 1.0)
            )
            exporter.export(
                x, y, r, theta, v, c,
                output_path=pdf_path,
                svg_hollow=config["primitive"]["primitive_hollow"],
                export_pdf=True
            )
            print(f"Saved PDF to: {pdf_path}")
        except Exception as e:
            print(f"PDF export failed: {e}")
            pdf_path = None
    
    # Compute metrics if requested
    metrics = {}
    if config.get('postprocessing', {}).get('compute_psnr', False):
        try:
            import piq
            
            rendered_t = rendered.permute(2, 0, 1).unsqueeze(0)
            target_t = I_target.permute(2, 0, 1).unsqueeze(0)
            
            # Apply mask if no background
            if not exist_bg and target_binary_mask is not None:
                foreground_mask = (target_binary_mask == 0)
                mask_tensor = foreground_mask.unsqueeze(0).unsqueeze(0).float()
                mask_tensor = mask_tensor.expand(-1, 3, -1, -1)
                rendered_t = rendered_t * mask_tensor
                target_t = target_t * mask_tensor
            
            rendered_t_f32 = torch.clamp(rendered_t.float(), 0.0, 1.0)
            target_t_f32 = torch.clamp(target_t.float(), 0.0, 1.0)
            
            metrics['psnr'] = piq.psnr(rendered_t_f32, target_t_f32, data_range=1.0).item()
            metrics['ssim'] = piq.ssim(rendered_t_f32, target_t_f32, data_range=1.0).item()
            metrics['vif'] = piq.vif_p(rendered_t_f32, target_t_f32, data_range=1.0).item()
            metrics['lpips'] = piq.LPIPS()(rendered_t_f32, target_t_f32).item()
            metrics['num_primitives'] = len(x)
            
            print(f"PSNR: {metrics['psnr']:.2f} dB")
            print(f"SSIM: {metrics['ssim']:.4f}")
            print(f"VIF: {metrics['vif']:.4f}")
            print(f"LPIPS: {metrics['lpips']:.4f}")
            print(f"Number of primitives: {metrics['num_primitives']}")
        except ImportError as e:
            print(f"Metrics computation failed: {e}")
    
    results = {
        'output_path': output_path,
        'pdf_path': pdf_path,
        'metrics': metrics,
        'num_primitives': len(x)
    }
    
    return results

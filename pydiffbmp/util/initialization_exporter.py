#!/usr/bin/env python3
"""
Initialization Exporter

This script extracts the preprocessing and initialization functionality from main.py
to test and visualize different initializer methods without running the full optimization.
It renders the initial state of primitives using different initializers and exports them as PNG images.
"""

import os
import sys
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import json
import argparse

# Import GradientVisualizer
from gradient_visualizer import GradientVisualizer

# Add the project root to the path to import pydiffbmp modules
# Script is at: circle_art/pydiffbmp/util/initialization_exporter.py
# Project root is: circle_art/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydiffbmp.core.preprocessing import Preprocessor
from pydiffbmp.core.initializer.svgsplat_initializater import StructureAwareInitializer
from pydiffbmp.core.initializer.random_initializater import RandomInitializer
from pydiffbmp.core.renderer.simple_tile_renderer import SimpleTileRenderer
from pydiffbmp.util.svg_loader import SVGLoader
from pydiffbmp.util.primitive_loader import PrimitiveLoader
from pydiffbmp.util.primitive_utils import expand_primitive_wildcards
from pydiffbmp.util.utils import set_global_seed, extract_chars_from_file
from pydiffbmp.util.constants import apply_constants_to_config

NUMBER_OF_VISUALIZATION_PRIMITIVES = 10

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def setup_environment():
    """Setup device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_and_preprocess_image(config, device):
    """Load and preprocess the target image"""
    print("Loading and preprocessing target image...")
    
    pp_conf = config["preprocessing"]
    
    # Handle multiple images - use first one for initialization export
    img_path = pp_conf["img_path"]
    if isinstance(img_path, list):
        img_path = img_path[0]
        pp_conf["img_path"] = img_path
        print(f"Multiple images detected, using first image: {img_path}")
    
    # Handle list parameters - extract first element (same as main.py)
    final_width = pp_conf.get("final_width", 128)
    if isinstance(final_width, list):
        final_width = final_width[0]
        print(f"Using first final_width value: {final_width}")
    
    # Initialize preprocessor
    preprocessor = Preprocessor(
        final_width=final_width,
        trim=pp_conf.get("trim", False),
        FM_halftone=pp_conf.get("FM_halftone", False),
        transform_mode=pp_conf.get("transform", "none"),
    )
    
    # Load target image
    target_binary_mask_np = None
    exist_bg = pp_conf.get("exist_bg", True)
    if exist_bg:
        print("Target image has background")
        I_target = preprocessor.load_image_8bit_color(pp_conf).astype(np.float32) / 255.0
    else:
        print("Target image has no background, using color and opacity")
        I_target, target_binary_mask_np = preprocessor.load_image_8bit_color_opacity(pp_conf)
        I_target = I_target.astype(np.float32) / 255.0
    
    I_target = torch.tensor(I_target, device=device)  # (H, W, 3) or (H, W, 4) if no background
    H = preprocessor.final_height
    W = preprocessor.final_width
    
    print(f"Target image shape: {I_target.shape}")
    print(f"Canvas size: {H} x {W}")
    
    return I_target, H, W, preprocessor, target_binary_mask_np

def load_primitive_templates(config, device):
    """Load primitive templates (SVG, PNG, JPG, TTF, etc.) using the same logic as main.py"""
    print("Loading primitive templates...")
    
    primitive_conf = config["primitive"]
    
    # Handle primitive file loading (same logic as main.py)
    primitive_file_config = primitive_conf.get("primitive_file")
    primitive_file_config = expand_primitive_wildcards(primitive_file_config)
    
    # Handle list of files (multiple primitives)
    if isinstance(primitive_file_config, list):
        # Multiple primitives - process each one
        svg_path = []
        for file_item in primitive_file_config:
            file_ext = os.path.splitext(file_item)[1].lower()
            if file_ext == ".svg":
                svg_path.append(os.path.join("assets/svg", file_item))
            elif file_ext in (".png", ".jpg", ".jpeg"):
                if primitive_conf.get("convert_to_svg"):
                    from pydiffbmp.util.svg_converter import ImageToSVG
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
            if primitive_conf.get("convert_to_svg", True):
                from pydiffbmp.util.svg_converter import ImageToSVG
                img_converter = ImageToSVG()
                svg_path = img_converter.extract_filled_outlines(primitive_file_config, threshold=100, min_area_ratio=0.000001)
                del img_converter
            else:
                svg_path = os.path.join("assets/primitives", primitive_file_config)
        elif primitive_ext in (".otf", ".ttf"):
            # Handle font files - text rendering
            texts = None
            if "text" in primitive_conf and primitive_conf["text"] is not None:
                texts = primitive_conf["text"]
            elif "text_file" in primitive_conf and primitive_conf["text_file"] is not None:
                # File-based text extraction
                text_ext = os.path.splitext(primitive_conf.get("text_file"))[1].lower()
                text_path = os.path.join("./assets/texts", primitive_conf.get("text_file"))
                
                if text_ext == ".txt" or text_ext == ".lrc":
                    texts, char_counts, word_lengths_per_line = extract_chars_from_file(
                        text_path, text_ext, 
                        remove_punct=primitive_conf.get("remove_punctuation", False), 
                        punct_to_remove=".,;:(){}[]\"'"
                    )
                else:
                    raise ValueError(f"Unsupported text_file type: {text_ext}")
            
            if texts is not None:
                from pydiffbmp.util.svg_converter import FontParser
                font_parser = FontParser(primitive_file_config)
                if isinstance(texts, list):
                    svg_paths = [str(font_parser.text_to_svg(t, mode="opt-path")) for t in texts]
                else:
                    svg_paths = str(font_parser.text_to_svg(texts, mode="opt-path"))
                svg_path = svg_paths
                del font_parser
            else:
                raise ValueError("No text source ('text' or 'text_file') provided in svg config.")
        else:
            svg_path = primitive_file_config if primitive_file_config else "assets/svg/circle.svg"
    
    # Load primitives using PrimitiveLoader with fallback to SVGLoader (same as main.py)
    try:
        primitive_loader = PrimitiveLoader(
            primitive_paths=svg_path,
            output_width=primitive_conf.get("output_width", 128),
            device=device,
            bg_threshold=primitive_conf.get("bg_threshold", 250),
            radial_transparency=primitive_conf.get("radial_transparency", False),
            resampling=primitive_conf.get("resampling", "LANCZOS")
        )
        # Keep reference for backward compatibility
        svg_loader = primitive_loader
        print(f"Loaded primitives: {len(primitive_loader.primitive_paths)} files")
        print(f"Primitive types: {primitive_loader.primitive_types}")
        bmp_tensor = svg_loader.load_alpha_bitmap()
    except Exception as e:
        print(f"PrimitiveLoader failed, falling back to SVGLoader: {e}")
        svg_loader = SVGLoader(
            svg_path=svg_path,
            output_width=primitive_conf.get("output_width", 128),
            device=device
        )
        primitive_loader = None
        bmp_tensor = svg_loader.load_alpha_bitmap()
    
    print(f"Primitive template shape: {bmp_tensor.shape}")
    
    # Extract primitive colors for c_o initialization (same logic as main.py)
    if primitive_loader is not None:
        primitive_colors = primitive_loader.get_primitive_color_maps()  # (num_primitives, 3)
        print(f"Extracted primitive colors: {primitive_colors.shape}")
    else:
        # Fallback: use default colors if primitive_loader is not available
        num_primitives = bmp_tensor.shape[0] if bmp_tensor.ndim == 3 else 1
        primitive_colors = torch.zeros(num_primitives, 128, 128, 3, device=device)
        print("Using default colors for primitives")
    
    return bmp_tensor, primitive_colors

def create_renderer(config, H, W, bmp_tensor, primitive_colors, device):
    """Create the renderer for visualization"""
    print("Creating renderer...")
    
    opt_conf = config["optimization"]
    output_conf = config["postprocessing"]
    
    renderer_kwargs = {
        "canvas_size": (H, W),
        "S": bmp_tensor,
        "alpha_upper_bound": opt_conf.get("alpha_upper_bound", 0.5),
        "device": device,
        "use_fp16": opt_conf.get("use_fp16", False),
        "output_path": output_conf.get("output_folder", "./outputs/"),
        "tile_size": opt_conf.get("tile_size", 32),
        "sigma": opt_conf.get("blur_sigma", 0.0) if opt_conf.get("do_gaussian_blur", False) else 0.0,
        "c_blend": opt_conf.get("c_blend", 0.0),
        "primitive_colors": primitive_colors,  # Pass primitive colors for c_o initialization
        "max_prims_per_pixel": config["initialization"].get("max_prims_per_pixel"),  # Pass max_prims_per_pixel from config
    }
    
    renderer = SimpleTileRenderer(**renderer_kwargs)
    print(f"Using SimpleTileRenderer for visualization")
    
    return renderer

def initialize_with_method(config, renderer, I_target, target_binary_mask=None):
    """Initialize primitives using the specified method from config"""
    init_conf = config["initialization"]
    initializer_name = init_conf.get("initializer", "structure_aware")
    
    print(f"Initializing primitives with {initializer_name} method...")
    
    if initializer_name == "structure_aware":
        initializer = StructureAwareInitializer(init_conf)
    elif initializer_name == "random":
        initializer = RandomInitializer(init_conf)
    else:
        raise ValueError(f"Unknown initializer: {initializer_name}")
    
    # Initialize parameters with optional target_binary_mask
    x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target, target_binary_mask)
    
    print(f"Initialized {len(x)} primitives")
    print(f"Position range: x=[{x.min():.3f}, {x.max():.3f}], y=[{y.min():.3f}, {y.max():.3f}]")
    print(f"Scale range: r=[{r.min():.3f}, {r.max():.3f}]")
    print(f"Visibility range: v=[{v.min():.3f}, {v.max():.3f}]")
    print(f"Rotation range: theta=[{theta.min():.3f}, {theta.max():.3f}]")
    print(f"Color range: c=[{c.min():.3f}, {c.max():.3f}]")
    
    return x, y, r, v, theta, c

def render_and_save(renderer, x, y, r, v, theta, c, output_path, method_name):
    """Render the initialized primitives and save as PNG"""
    print(f"Rendering {method_name} initialization...")
    
    with torch.no_grad():
        # Create white background
        white_bg = torch.ones((renderer.H, renderer.W, 3), device=renderer.device)
        
        # Render using tile-based rendering
        rendered = renderer.render_from_params(
            x, y, r, theta, v, c, 
            I_bg=white_bg, 
            sigma=0.0
        )
        
        # Convert to numpy and save
        rendered_np = rendered.detach().cpu().numpy()
        rendered_np = np.clip(rendered_np, 0, 1)
        rendered_np = (rendered_np * 255).astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(rendered_np)
        img.save(output_path)
        print(f"Saved {method_name} initialization to: {output_path}")

def save_target_image(I_target, output_path):
    """Save the target image for comparison"""
    if isinstance(I_target, torch.Tensor):
        # Convert tensor to numpy array
        I_np = I_target.detach().cpu().numpy()
        
        # Handle different tensor formats
        if I_np.ndim == 3 and I_np.shape[2] == 3:
            # RGB format (H, W, 3)
            I_np = (I_np * 255).astype(np.uint8)
        else:
            # Grayscale or other format
            I_np = (I_np * 255).astype(np.uint8)
            if I_np.ndim == 2:
                I_np = np.stack([I_np] * 3, axis=-1)
    
    # Save as PNG
    Image.fromarray(I_np).save(output_path)
    print(f"Target image saved to: {output_path}")

def analyze_edge_proximity(x, y, I_target, output_dir, timestamp, method_name, top_k=50):
    """
    Analyze edge proximity of primitives and visualize closest primitives on edge map.
    Uses the same edge processing logic as base_initializer.py and svgsplat_initializater.py.
    
    Args:
        x, y: Primitive coordinates (torch tensors)
        I_target: Target image tensor
        output_dir: Directory to save results
        timestamp: Timestamp for unique filenames
        method_name: Name of initialization method
        top_k: Number of closest-to-edge primitives to extract
    
    Returns:
        torch.Tensor: Indices of primitives closest to edges [top_k]
    """
    print(f"\nAnalyzing edge proximity for {method_name} initialization...")
    
    # Convert target image to numpy format for edge processing
    if isinstance(I_target, torch.Tensor):
        I_np = I_target.detach().cpu().numpy()
        # Convert to grayscale if needed
        if I_np.ndim == 3:
            I_np = np.mean(I_np, axis=2)
    else:
        I_np = I_target
        if I_np.ndim == 3:
            I_np = cv2.cvtColor(I_np, cv2.COLOR_RGB2GRAY)
    
    # Ensure image is in correct format for edge detection
    if I_np.dtype != np.uint8:
        I_np = (I_np * 255).astype(np.uint8)
    
    H, W = I_np.shape
    
    # Edge processing - EXACTLY matching the initializer logic
    edges = cv2.Canny(I_np, 100, 200)
    inverted_edges = cv2.bitwise_not(edges)
    distance_map = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
    
    # Convert primitive coordinates to numpy
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    # Sample distance at each primitive location
    idx_x = np.clip(np.round(x_np).astype(int), 0, W - 1)
    idx_y = np.clip(np.round(y_np).astype(int), 0, H - 1)
    primitive_distances = distance_map[idx_y, idx_x]
    
    # Find indices of primitives closest to edges (smallest distances)
    closest_indices = np.argsort(primitive_distances)[:top_k]
    closest_distances = primitive_distances[closest_indices]
    
    # Get coordinates of closest primitives
    closest_x = x_np[closest_indices]
    closest_y = y_np[closest_indices]
    
    # Calculate statistics
    distance_stats = {
        'min': np.min(primitive_distances),
        'max': np.max(primitive_distances),
        'mean': np.mean(primitive_distances),
        'std': np.std(primitive_distances)
    }
    
    # Create visualization: overlay closest primitives on edge map
    plt.figure(figsize=(12, 8))
    
    # Display edge map as background
    plt.imshow(edges, cmap='gray', alpha=0.7)
    
    # Overlay closest primitives as red circles
    plt.scatter(closest_x, closest_y, c='red', s=30, alpha=0.8, edgecolors='white', linewidth=1)
    
    # Add title with statistics
    title = f"{method_name.upper()} Initialization - Top {top_k} Closest to Edges\n"
    title += f"Total: {len(x_np)} primitives | "
    title += f"Avg distance: {distance_stats['mean']:.2f} | "
    title += f"Min distance: {distance_stats['min']:.2f} | "
    title += f"Closest range: {np.min(closest_distances):.2f}-{np.max(closest_distances):.2f}"
    
    plt.title(title, fontsize=12, pad=20)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # Add legend
    plt.scatter([], [], c='red', s=30, alpha=0.8, edgecolors='white', linewidth=1, label=f'Top {top_k} closest to edges')
    plt.legend(loc='upper right')
    
    # Set axis limits and invert y-axis to match image coordinates
    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save the visualization
    viz_path = os.path.join(output_dir, f"edge_proximity_viz_{method_name}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Edge proximity analysis completed for {method_name}:")
    print(f"  Visualization saved to: {viz_path}")
    print(f"  Total primitives: {len(x_np)}")
    print(f"  Closest {top_k} primitives - distance range: {np.min(closest_distances):.4f} to {np.max(closest_distances):.4f}")
    print(f"  Overall distance stats - mean: {distance_stats['mean']:.4f}, std: {distance_stats['std']:.4f}")
    
    # Return closest indices as torch tensor for compatibility with GradientVisualizer
    return torch.tensor(closest_indices, dtype=torch.long)

def main(config):
    """Main function to run initialization export"""
    print("=" * 60)
    print("INITIALIZATION EXPORTER")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup
    device = setup_environment()
    
    # Handle list parameters - extract first element (same as main.py)
    if isinstance(config["preprocessing"].get("final_width"), list):
        config["preprocessing"]["final_width"] = config["preprocessing"]["final_width"][0]
        print("Using only one final_width value for initialization export")
    if isinstance(config["initialization"].get("N"), list):
        config["initialization"]["N"] = config["initialization"]["N"][0]
        print("Using only one N value for initialization export")
    if isinstance(config["primitive"].get("primitive_file"), list) and len(config["primitive"]["primitive_file"]) > 1:
        # Note: expand_primitive_wildcards already handles this, but log it
        print(f"Using {len(config['primitive']['primitive_file'])} primitive files")
    
    # Create output directory
    output_conf = config["postprocessing"]
    output_dir = output_conf.get("output_folder", "./outputs/initialization_test/")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    I_target, H, W, preprocessor, target_binary_mask_np = load_and_preprocess_image(config, device)
    
    # Convert target_binary_mask_np to tensor if it exists
    target_binary_mask = None
    if target_binary_mask_np is not None:
        target_binary_mask = torch.from_numpy(target_binary_mask_np[:, :] > 0).to(device)
        print(f"Target binary mask created: {target_binary_mask.shape}")
    
    # Load primitive templates
    bmp_tensor, primitive_colors = load_primitive_templates(config, device)
    
    # Create renderer
    renderer = create_renderer(config, H, W, bmp_tensor, primitive_colors, device)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save target image for comparison
    target_path = os.path.join(output_dir, f"target_{timestamp}.png")
    save_target_image(I_target, target_path)
    
    # Test the configured initializer
    init_conf = config["initialization"]
    method_name = init_conf.get("initializer", "structure_aware")
    
    print(f"\n{'-' * 40}")
    print(f"Testing {method_name.upper()} initializer")
    print(f"{'-' * 40}")
    
    try:
        # Initialize primitives
        x, y, r, v, theta, c = initialize_with_method(
            config, renderer, I_target, target_binary_mask
        )
        
        # Render and save
        output_path = os.path.join(output_dir, f"init_{method_name}_{timestamp}.png")
        render_and_save(renderer, x, y, r, v, theta, c, output_path, method_name)
        
        # Analyze edge proximity of initialized primitives
        closest_indices = analyze_edge_proximity(
            x, y, I_target, output_dir, timestamp, method_name, top_k=NUMBER_OF_VISUALIZATION_PRIMITIVES
        )
        
        # Create GradientVisualizer and visualize gradients for closest primitives
        gradient_save_path = os.path.join(output_dir, f"gradient_{method_name}_{timestamp}")
        gradient_visualizer = GradientVisualizer(
            target_image=I_target,
            save_path=gradient_save_path,
            color_spectrum="full",
            background_color=(1.0, 1.0, 1.0),  # White background
            enable_logging=True,
            center_dot_radius=1,
        )
        
        # Visualize per-pixel gradients for closest-to-edge primitives
        print(f"\nVisualizing gradients for {len(closest_indices)} closest-to-edge primitives...")
        vis_tensor, saved_path = gradient_visualizer.visualize_gradients(
            renderer, x, y, r, v, theta, c, closest_indices,
            suffix="closest_to_edges",
            title_prefix=f"Gradient Visualization - {method_name.upper()} Closest to Edges",
            center_dot_color=(0.0, 0.0, 0.0)  # Black dots
        )
        print(f"Gradient visualization saved to: {saved_path}")
        
    except Exception as e:
        print(f"Error with {method_name} initializer: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'=' * 60}")
    print("INITIALIZATION EXPORT COMPLETED")
    print(f"{'=' * 60}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - target_{timestamp}.png (original target image)")
    print(f"  - init_{method_name}_{timestamp}.png (initialized with {method_name})")
    print(f"  - edge_proximity_viz_{method_name}_{timestamp}.png (edge proximity analysis)")
    print(f"  - gradient_{method_name}_{timestamp}_closest_to_edges.png (gradient visualization)")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test and visualize different initializer methods")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    
    # Load configuration from JSON file
    print(f"Loading config from: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Apply constants as default values (same as main.py)
    config = apply_constants_to_config(config)
    
    # Set random seed from config
    set_global_seed(config.get("seed", 42))
    
    # Run main with loaded config
    main(config)

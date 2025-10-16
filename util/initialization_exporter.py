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

# Import GradientVisualizer
from gradient_visualizer import GradientVisualizer

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessing import Preprocessor
from core.initializer.svgsplat_initializater import StructureAwareInitializer
from core.initializer.random_initializater import RandomInitializer
from core.renderer.simple_tile_renderer import SimpleTileRenderer
from util.svg_loader import SVGLoader
from util.primitive_loader import PrimitiveLoader
from util.utils import set_global_seed

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "img_path": "./images/personal_trial/box.png",  # Change this to your target image
    "final_width": 256,
    "trim": False,
    "FM_halftone": False,
    "transform": "none",
    "exist_bg": True,
    # Required parameters for preprocessing functions
    "do_equalize": False,
    "do_local_contrast": False,
    "do_tone_curve": False,
    "bg_threshold": 250,
    "local_contrast": {
        "radius": 2.0,
        "amount": 3.0
    },
    "tone_params": {
        "in_low": 50,
        "in_high": 200,
        "out_low": 30,
        "out_high": 230
    },
    "vertical_paddings": [0, 0]  # [top, bottom] padding
}

# SVG/Primitive configuration
SVG_CONFIG = {
    "primitive_file": "arial.ttf",  # Change this to your primitive file
    "convert_to_svg": True,
    "output_width": 128,
    "bg_threshold": 250,
    "svg_hollow": False,
    # For font files (TTF/OTF)
    "text": ["A", "B", "C"],  # Text to render if using font files
    "text_file": None,  # Path to text file if using external text
    "remove_punctuation": False
}

# Initialization configuration
INITIALIZATION_CONFIG = {
    "N": 1000,  # Number of primitives
    "structure_aware": {
        "initializer": "structure_aware",
        "N": 1000,
        # Base initializer parameters
        "alpha": 0.3,
        "min_distance": 20,
        "peak_threshold": 0.5,
        "radii_min": 2,
        "radii_max": 8,
        "v_init_bias": -5.0,
        "v_init_slope": 10.0,
        "keypoint_extracting": False,
        "debug_mode": False,
        "distance_factor": 0.0,
    },
    "random": {
        "initializer": "random",
        "N": 1000,
        # Base initializer parameters
        "alpha": 0.3,
        "min_distance": 20,
        "peak_threshold": 0.5,
        "radii_min": 2,
        "radii_max": 8,
        "v_init_bias": -5.0,
        "v_init_slope": 10.0,
        "keypoint_extracting": False,
        "debug_mode": False,
        "distance_factor": 0.0,
    }
}

# Optimization configuration (minimal, just for renderer setup)
OPTIMIZATION_CONFIG = {
    "alpha_upper_bound": 0.5,
    "use_fp16": False,
    "tile_size": 32,
    "blur_sigma": 0.0,
    "do_gaussian_blur": False
}

# Output configuration
OUTPUT_CONFIG = {
    "output_folder": "./outputs/box/initialization_test001/",
    "save_format": "png"
}

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def setup_environment():
    """Setup device and random seed"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seed(RANDOM_SEED)
    print(f"Using device: {device}")
    return device

def load_and_preprocess_image(device):
    """Load and preprocess the target image"""
    print("Loading and preprocessing target image...")
    
    # Initialize preprocessor
    preprocessor = Preprocessor(
        final_width=PREPROCESSING_CONFIG.get("final_width", 128),
        trim=PREPROCESSING_CONFIG.get("trim", False),
        FM_halftone=PREPROCESSING_CONFIG.get("FM_halftone", False),
        transform_mode=PREPROCESSING_CONFIG.get("transform", "none"),
    )
    
    # Load target image
    if PREPROCESSING_CONFIG.get("exist_bg", True):
        print("Target image has background")
        I_target = preprocessor.load_image_8bit_color(PREPROCESSING_CONFIG).astype(np.float32) / 255.0
    else:
        print("Target image has no background, using color and opacity")
        I_target, target_binary_mask_np = preprocessor.load_image_8bit_color_opacity(PREPROCESSING_CONFIG)
        I_target = I_target.astype(np.float32) / 255.0
    
    I_target = torch.tensor(I_target, device=device)  # (H, W, 3) or (H, W, 4) if no background
    H = preprocessor.final_height
    W = preprocessor.final_width
    
    print(f"Target image shape: {I_target.shape}")
    print(f"Canvas size: {H} x {W}")
    
    return I_target, H, W, preprocessor

def load_primitive_templates(device):
    """Load primitive templates (SVG, PNG, JPG, TTF, etc.) using the same logic as main.py"""
    print("Loading primitive templates...")
    
    # Handle SVG file loading (same logic as main.py lines 134-197)
    primitive_file = SVG_CONFIG.get("primitive_file")
    svg_ext = os.path.splitext(primitive_file)[1].lower()
    
    if svg_ext == ".svg":
        svg_path = os.path.join("./assets/svg", primitive_file)
    elif svg_ext in (".png", ".jpg", ".jpeg"):
        if SVG_CONFIG.get("convert_to_svg", True):
            from util.svg_converter import ImageToSVG
            img_converter = ImageToSVG()
            svg_path = img_converter.extract_filled_outlines(primitive_file, threshold=100, min_area_ratio=0.000001)
            del img_converter
        else:
            svg_path = os.path.join("./assets/primitives", primitive_file)
    elif svg_ext in (".otf", ".ttf"):
        # Handle font files - text rendering
        texts = None
        if "text" in SVG_CONFIG and SVG_CONFIG["text"] is not None:
            texts = SVG_CONFIG["text"]
        elif "text_file" in SVG_CONFIG and SVG_CONFIG["text_file"] is not None:
            # File-based text extraction
            text_ext = os.path.splitext(SVG_CONFIG.get("text_file"))[1].lower()
            text_path = os.path.join("./assets/texts", SVG_CONFIG.get("text_file"))
            
            if text_ext == ".txt" or text_ext == ".lrc":
                texts, char_counts, word_lengths_per_line = extract_chars_from_file(
                    text_path, text_ext, 
                    remove_punct=SVG_CONFIG.get("remove_punctuation", False), 
                    punct_to_remove=".,;:(){}[]\"'"
                )
            else:
                raise ValueError(f"Unsupported text_file type: {text_ext}")
        
        if texts is not None:
            from util.svg_converter import FontParser
            font_parser = FontParser(primitive_file)
            if isinstance(texts, list):
                svg_paths = [str(font_parser.text_to_svg(t, mode="opt-path")) for t in texts]
            else:
                svg_paths = str(font_parser.text_to_svg(texts, mode="opt-path"))
            svg_path = svg_paths
            del font_parser
        else:
            raise ValueError("No text source ('text' or 'text_file') provided in svg config.")
    else:
        svg_path = SVG_CONFIG.get("primitive_file", "assets/svg/circle.svg")
    
    # Load primitives using PrimitiveLoader with fallback to SVGLoader (same as main.py)
    try:
        primitive_loader = PrimitiveLoader(
            primitive_paths=svg_path,
            output_width=SVG_CONFIG.get("output_width", 128),
            device=device,
            bg_threshold=SVG_CONFIG.get("bg_threshold", 250)
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
            output_width=SVG_CONFIG.get("output_width", 128),
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

def create_renderer(H, W, bmp_tensor, primitive_colors, device):
    """Create the renderer for visualization"""
    print("Creating renderer...")
    
    renderer_kwargs = {
        "canvas_size": (H, W),
        "S": bmp_tensor,
        "alpha_upper_bound": OPTIMIZATION_CONFIG.get("alpha_upper_bound", 0.5),
        "device": device,
        "use_fp16": OPTIMIZATION_CONFIG.get("use_fp16", False),
        "output_path": OUTPUT_CONFIG.get("output_folder", "./outputs/"),
        "tile_size": OPTIMIZATION_CONFIG.get("tile_size", 32),
        "sigma": OPTIMIZATION_CONFIG.get("blur_sigma", 0.0) if OPTIMIZATION_CONFIG.get("do_gaussian_blur", False) else 0.0,
        "c_blend": OPTIMIZATION_CONFIG.get("c_blend", 0.0),
        "primitive_colors": primitive_colors,  # Pass primitive colors for c_o initialization
    }
    
    renderer = SimpleTileRenderer(**renderer_kwargs)
    print(f"Using SimpleTileRenderer for visualization")
    
    return renderer

def initialize_with_method(initializer_name, initializer_config, renderer, I_target):
    """Initialize primitives using the specified method"""
    print(f"Initializing primitives with {initializer_name} method...")
    
    if initializer_name == "structure_aware":
        initializer = StructureAwareInitializer(initializer_config)
    elif initializer_name == "random":
        initializer = RandomInitializer(initializer_config)
    else:
        raise ValueError(f"Unknown initializer: {initializer_name}")
    
    # Initialize parameters
    x, y, r, v, theta, c = renderer.initialize_parameters(initializer, I_target)
    
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

def main():
    """Main function to run initialization export"""
    print("=" * 60)
    print("INITIALIZATION EXPORTER")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup
    device = setup_environment()
    
    # Create output directory
    output_dir = OUTPUT_CONFIG.get("output_folder", "./outputs/initialization_test/")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    I_target, H, W, preprocessor = load_and_preprocess_image(device)
    
    # Load primitive templates
    bmp_tensor, primitive_colors = load_primitive_templates(device)
    
    # Create renderer
    renderer = create_renderer(H, W, bmp_tensor, primitive_colors, device)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save target image for comparison
    # target_path = os.path.join(output_dir, f"target_{timestamp}.png")
    # save_target_image(I_target, target_path)
    
    # Test different initializers
    initializers_to_test = [
        ("structure_aware", INITIALIZATION_CONFIG["structure_aware"]),
        ("random", INITIALIZATION_CONFIG["random"])
    ]
    
    for method_name, method_config in initializers_to_test:
        print(f"\n{'-' * 40}")
        print(f"Testing {method_name.upper()} initializer")
        print(f"{'-' * 40}")
        
        try:
            # Initialize primitives
            x, y, r, v, theta, c = initialize_with_method(
                method_name, method_config, renderer, I_target
            )
            
            # Render and save
            output_path = os.path.join(output_dir, f"init_{method_name}_{timestamp}.png")
            render_and_save(renderer, x, y, r, v, theta, c, output_path, method_name)
            
            # Analyze edge proximity of initialized primitives
            closest_indices = analyze_edge_proximity(
                x, y, I_target, output_dir, timestamp, method_name, top_k=10
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
            continue
    
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
    for method_name, _ in initializers_to_test:
        print(f"  - init_{method_name}_{timestamp}.png (initialized with {method_name})")

if __name__ == "__main__":
    main()

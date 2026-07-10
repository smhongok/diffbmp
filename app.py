"""
Gradio demo for DiffBMP: Differentiable Painting with Primitives
Hugging Face Spaces deployment version using PyTorch fallback (no custom CUDA kernel)
"""
import os
import gradio as gr
import torch
import json
from PIL import Image
import tempfile
import shutil

# Force PyTorch fallback - disable custom CUDA kernel
os.environ['DIFFBMP_FORCE_PYTORCH'] = '1'

from diffbmp_pipeline import process_single_image


# Load available config templates
CONFIG_DIR = "configs"
AVAILABLE_CONFIGS = {}

try:
    import glob
    config_files = glob.glob(os.path.join(CONFIG_DIR, "default*.json"))
    for config_file in config_files:
        config_name = os.path.basename(config_file).replace(".json", "")
        with open(config_file, "r") as f:
            AVAILABLE_CONFIGS[config_name] = json.load(f)
except Exception as e:
    print(f"Warning: Could not load config templates: {e}")

# Default configuration template for demo
DEFAULT_CONFIG = {
    "seed": 42,
    "preprocessing": {
        "final_width": 256,
        "img_path": None,
        "trim": False,
        "exist_bg": True
    },
    "primitive": {
        "primitive_file": "flowers/Gerbera1.png",
        "primitive_hollow": False,
        "output_width": 128,
        "radial_transparency": False,
        "resampling": "bilinear"
    },
    "initialization": {
        "initializer": "structure_aware",
        "N": 200,
        "v_init_bias": -4.0,
        "radii_min": 4,
        "radii_max": 20,
        "debug_mode": False
    },
    "optimization": {
        "use_fp16": False,
        "num_iterations": 50,
        "learning_rate": {
            "default": 0.1
        },
        "c_blend": 1.0,
        "alpha_upper_bound": 1.0,
        "loss_config": {
            "type": "combined",
            "components": [
                {"name": "grayscale_l1", "weight": 1.0},
                {"name": "mse", "weight": 0.2}
            ]
        },
        "do_decay": True,
        "do_gaussian_blur": True,
        "blur_sigma": 1.0,
        "tile_size": 32,
        "bg_color": "white"
    },
    "postprocessing": {
        "output_folder": "outputs/",
        "compute_psnr": False,
        "linewidth": 1.0
    }
}

# Important config parameters with descriptions
IMPORTANT_PARAMS = """
---
### 🔑 Key Configuration Parameters

**⭐ Most Important (affects quality/speed):**
- **`initialization.N`**: Number of primitives (50-2000+)
  - More = better detail, slower processing
- **`optimization.num_iterations`**: Training steps (10-300+)
  - More = better quality, slower processing
- **`preprocessing.final_width`**: Output resolution (128-720)
  - Higher = more detail, slower processing

**🎨 Important for style:**
- **`optimization.c_blend`**: Color optimization (0.0-1.0)
  - 0 = use primitive's original colors
  - 1 = fully optimize colors (recommended)
- **`initialization.initializer`**: Placement strategy
  - `"structure_aware"` (default, smart placement)
  - `"random"` or `"designated"`

**⚙️ Advanced settings:**
- **`preprocessing.exist_bg`**: Background handling
  - `true` = image has solid background
  - `false` = transparent/alpha channel support
- **`primitive.output_width`**: Primitive resolution (64-256)
- **`initialization.radii_min/max`**: Primitive size range
- **`optimization.alpha_upper_bound`**: Max opacity (0.0-1.0)
- **`optimization.loss_config`**: Loss function weights
  - Try `mse`, `grayscale_l1`, `alpha`, `lpips`, etc.
- **`postprocessing.compute_psnr`**: Calculate quality metrics
- **`postprocessing.export_psd`**: Export layered PSD file

**⚠️ Required for demo:**
- **`optimization.use_fp16`**: Must be `false` (PyTorch fallback)
"""


def load_config_template(config_name):
    """Load a config template and return as JSON string"""
    if config_name in AVAILABLE_CONFIGS:
        config = AVAILABLE_CONFIGS[config_name].copy()
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Normalize list values to single values (for multi-image configs)
    if "preprocessing" in config:
        config["preprocessing"]["img_path"] = None
        # Handle final_width as list (e.g., [720] -> 720)
        if isinstance(config["preprocessing"].get("final_width"), list):
            config["preprocessing"]["final_width"] = config["preprocessing"]["final_width"][0]
    
    if "initialization" in config:
        # Handle N as list (e.g., [50] -> 50)
        if isinstance(config["initialization"].get("N"), list):
            config["initialization"]["N"] = config["initialization"]["N"][0]
    
    if "primitive" in config:
        config["primitive"]["primitive_file"] = "<will be set from uploaded file>"
    
    if "optimization" in config:
        config["optimization"]["use_fp16"] = False  # Force PyTorch fallback
    
    return json.dumps(config, indent=2)


def generate_art(
    input_image,
    primitive_files,
    config_json,
    progress=gr.Progress()
):
    """
    Main processing function for Gradio interface.
    
    Args:
        input_image: PIL Image from Gradio (target image)
        primitive_files: Uploaded primitive file(s) - single file path or list of paths
        config_json: Configuration as JSON string
        progress: Gradio progress tracker
    
    Returns:
        tuple: (output_image, info_text)
    """
    print("=" * 60)
    print("🎨 GENERATE_ART FUNCTION CALLED")
    print(f"Input image: {'Provided' if input_image else 'None'}")
    print(f"Primitive files type: {type(primitive_files)}")
    print(f"Primitive files: {primitive_files}")
    print("=" * 60)
    
    if input_image is None:
        return None, "❌ Please upload a target image first!"
    
    if primitive_files is None or (isinstance(primitive_files, list) and len(primitive_files) == 0):
        return None, "❌ Please upload at least one primitive file (SVG, PNG, or JPG)!"
    
    # Parse config JSON
    try:
        config = json.loads(config_json)
        print(f"📋 Config parsed, primitive_file before update: {config.get('primitive', {}).get('primitive_file')}")
    except json.JSONDecodeError as e:
        return None, f"❌ Invalid JSON configuration:\n{str(e)}"
    
    # Normalize list values (config files may have lists for multi-image processing)
    if "preprocessing" in config:
        if isinstance(config["preprocessing"].get("final_width"), list):
            config["preprocessing"]["final_width"] = config["preprocessing"]["final_width"][0]
        if isinstance(config["preprocessing"].get("img_path"), list):
            config["preprocessing"]["img_path"] = config["preprocessing"]["img_path"][0]
    
    if "initialization" in config:
        if isinstance(config["initialization"].get("N"), list):
            config["initialization"]["N"] = config["initialization"]["N"][0]
    
    try:
        progress(0, desc="Initializing...")
        
        # Save input image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            input_image.save(tmp.name)
            img_path = tmp.name
        
        # Handle primitive file(s) upload
        # Gradio returns a single path for single file, or list of paths for multiple files
        if not isinstance(primitive_files, list):
            primitive_files = [primitive_files]  # Convert single file to list
        
        # Create temporary directory for primitives
        temp_primitive_dir = tempfile.mkdtemp(prefix='diffbmp_primitives_')
        
        # Copy all uploaded files to temp directory
        primitive_file_names = []
        print(f"📁 Processing {len(primitive_files)} uploaded file(s)...")
        for prim_file_path in primitive_files:
            primitive_ext = os.path.splitext(prim_file_path)[1].lower()
            
            # Validate file type
            if primitive_ext not in ['.svg', '.png', '.jpg', '.jpeg']:
                return None, f"❌ Unsupported primitive file type: {primitive_ext}. Use SVG, PNG, or JPG."
            
            # Copy to temp directory
            primitive_filename = os.path.basename(prim_file_path)
            dest_path = os.path.join(temp_primitive_dir, primitive_filename)
            shutil.copy(prim_file_path, dest_path)
            primitive_file_names.append(primitive_filename)
        
        print(f"✅ Uploaded files: {', '.join(primitive_file_names[:5])}{'...' if len(primitive_file_names) > 5 else ''}")
        
        # Determine primitive_file_config
        if len(primitive_file_names) == 1:
            # Single file - use filename directly
            primitive_file_config = primitive_file_names[0]
        else:
            # Multiple files - use wildcard pattern or list
            # Determine common extension
            extensions = set(os.path.splitext(name)[1] for name in primitive_file_names)
            if len(extensions) == 1:
                # All same extension - use wildcard
                ext = list(extensions)[0]
                primitive_file_config = f"*{ext}"
            else:
                # Mixed extensions - prefer PNG, otherwise use most common
                from collections import Counter
                ext_counts = Counter([os.path.splitext(name)[1] for name in primitive_file_names])
                
                # Prefer .png if available
                if '.png' in ext_counts and ext_counts['.png'] > 0:
                    primitive_file_config = "*.png"
                    print(f"⚠️  Mixed extensions detected. Using PNG files only ({ext_counts['.png']} files)")
                    # Remove non-PNG files from the list to copy
                    primitive_file_names = [name for name in primitive_file_names if name.endswith('.png')]
                else:
                    most_common_ext = ext_counts.most_common(1)[0][0]
                    primitive_file_config = f"*{most_common_ext}"
                    print(f"⚠️  Mixed extensions detected. Using most common: {most_common_ext}")
                    # Keep only files with the most common extension
                    primitive_file_names = [name for name in primitive_file_names if name.endswith(most_common_ext)]
        
        # Copy to assets folder for the pipeline to find
        assets_primitive_dir = os.path.join('assets', 'primitives', 'demo_upload')
        os.makedirs(assets_primitive_dir, exist_ok=True)
        
        # Clean up previous uploads in this directory
        for old_file in os.listdir(assets_primitive_dir):
            os.remove(os.path.join(assets_primitive_dir, old_file))
        
        # Copy files
        for name in primitive_file_names:
            src = os.path.join(temp_primitive_dir, name)
            dst = os.path.join(assets_primitive_dir, name)
            shutil.copy(src, dst)
        
        # Update config to use the demo_upload folder
        primitive_file_config = os.path.join('demo_upload', primitive_file_config)
        print(f"🎨 Primitive config set to: {primitive_file_config}")
        
        # Create temporary output directory
        output_dir = tempfile.mkdtemp()
        
        # Update configuration with uploaded files
        config["preprocessing"]["img_path"] = img_path
        config["primitive"]["primitive_file"] = primitive_file_config
        config["postprocessing"]["output_folder"] = output_dir
        
        print(f"📋 Final config primitive_file: {config['primitive']['primitive_file']}")
        
        # Force PyTorch fallback
        if "optimization" not in config:
            config["optimization"] = {}
        config["optimization"]["use_fp16"] = False
        
        progress(0.1, desc="Processing image...")
        
        # Process image
        results = process_single_image(
            img_path=img_path,
            config=config,
            output_dir=output_dir,
            force_cpu=False,  # Use GPU if available
            disable_cuda_kernel=True  # Force PyTorch fallback
        )
        
        progress(0.9, desc="Loading results...")
        
        # Load output image
        output_image = Image.open(results['output_path'])
        
        # Create info text
        num_primitive_files = len(primitive_file_names)
        primitive_info = f"{num_primitive_files} file(s)" if num_primitive_files > 1 else primitive_file_names[0]
        
        info_text = f"""
✅ **Processing Complete!**

📊 **Results:**
- Number of primitives: {results['num_primitives']}
- Primitive files: {primitive_info}
- Output size: {config['preprocessing'].get('final_width', 'N/A')}
- Iterations: {config['optimization'].get('num_iterations', 'N/A')}
- Device: {'GPU (PyTorch)' if torch.cuda.is_available() else 'CPU'}
- Initializer: {config['initialization'].get('initializer', 'N/A')}
- C_blend: {config['optimization'].get('c_blend', 'N/A')}

💡 **Tips:**
- Upload multiple primitive files for variety (like logos)
- Edit the config JSON to fine-tune parameters
- Try different config templates from the dropdown
- Adjust `N` (primitives) and `num_iterations` for quality/speed trade-off
"""
        
        # Cleanup
        os.unlink(img_path)
        
        progress(1.0, desc="Done!")
        
        return output_image, info_text
        
    except Exception as e:
        import traceback
        error_msg = f"❌ **Error occurred:**\n\n```\n{str(e)}\n```\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
        return None, error_msg


# Create Gradio interface
with gr.Blocks(title="DiffBMP: Differentiable Painting") as demo:
    gr.Markdown("""
    # 🎨 DiffBMP: Differentiable Painting with Primitives
    
    Transform your images into beautiful vector art using optimization-based rendering.
    
    **How it works:**
    1. 📸 **Upload a target image** - The image you want to vectorize
    2. 🎨 **Upload primitive file(s)** - Single or multiple SVG/PNG/JPG files
       - Single file: One shape repeated
       - Multiple files: Variety of shapes (like logos)
    3. ⚙️ **Select a config template** - Choose from preset configurations
    4. ✏️ **Edit JSON config** (optional) - Fine-tune parameters directly
    5. 🚀 **Click "Generate Art"** - Wait 1-5 minutes
    6. 💾 **Download result** - Your vectorized artwork!
    
    ⚠️ **Note:** This demo uses GPU + PyTorch (no custom CUDA kernels) for compatibility.
    Processing time depends on config settings (especially `N` and `num_iterations`).
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input")
            input_image = gr.Image(
                type="pil",
                label="Upload Target Image",
                image_mode="RGB"
            )
            
            primitive_files = gr.File(
                label="Upload Primitive File(s) - Single or multiple SVG/PNG/JPG (e.g., logos)",
                file_types=[".svg", ".png", ".jpg", ".jpeg"],
                file_count="multiple",
                type="filepath"
            )
            
            gr.Markdown("### Configuration")
            
            # Config template selector
            config_choices = ["default (demo)"] + sorted(list(AVAILABLE_CONFIGS.keys()))
            config_template = gr.Dropdown(
                choices=config_choices,
                value="default (demo)",
                label="Select Config Template",
                info="Choose a preset configuration or edit the JSON below"
            )
            
            # JSON editor
            config_editor = gr.Code(
                value=json.dumps(DEFAULT_CONFIG, indent=2),
                language="json",
                label="Configuration JSON (Editable)",
                lines=20
            )
            
            generate_btn = gr.Button("🎨 Generate Art", variant="primary", size="lg")
            
            # Important parameters info
            gr.Markdown(IMPORTANT_PARAMS)
        
        with gr.Column():
            gr.Markdown("### Output")
            output_image = gr.Image(
                type="pil",
                label="Generated Art"
            )
            
            info_output = gr.Markdown("")
    
    # Connect config template selector to JSON editor
    def update_config_editor(template_name):
        if template_name == "default (demo)":
            return json.dumps(DEFAULT_CONFIG, indent=2)
        else:
            return load_config_template(template_name)
    
    config_template.change(
        fn=update_config_editor,
        inputs=[config_template],
        outputs=[config_editor]
    )
    
    # Example images
    gr.Markdown("### 📸 Config Templates:")
    gr.Markdown("""
    **Available templates:**
    - `default_logo`: Logo/no-background images
    - `default_no_bg`: Transparent background support
    - `default_logos_marilyn_color`: **Multiple logo primitives** (upload many files!)
    - `default_2k`: High resolution (2K)
    - And more in the dropdown above!
    
    **Tips:**
    - Select a template from the dropdown to load its configuration
    - For `default_logos_marilyn_color`: Upload multiple logo/primitive files
    - Edit the JSON directly to customize parameters
    - Important parameters are highlighted in the info box below
    """)
    
    # Connect button to function
    generate_btn.click(
        fn=generate_art,
        inputs=[
            input_image,
            primitive_files,
            config_editor
        ],
        outputs=[output_image, info_output]
    )
    
    gr.Markdown("""
    ---
    ### 🔬 About DiffBMP
    
    DiffBMP is a differentiable rendering framework that converts raster images into vector art
    using optimization. Unlike traditional vectorization, it uses gradient-based optimization
    to position and color primitives for the best reconstruction.
    
    **Key Features:**
    - ✨ **Any custom primitive files** (SVG, PNG, JPG) - single or multiple!
    - 🎭 **Multiple primitives** - Upload dozens of logos/shapes for variety
    - 🎨 Automatic color optimization
    - 📐 Structure-aware initialization
    - 🚀 GPU-accelerated rendering
    - ⚙️ **Full config control** - Edit all parameters via JSON
    
    **Research Paper:** [Coming soon]
    
    **Code:** [GitHub Repository](#)
    """)


# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # HF Spaces default port
        share=False,  # Don't create public link (HF handles this)
        theme=gr.themes.Soft()  # Moved from Blocks() for Gradio 6.0 compatibility
    )

# Installation Guide for image-as-brush

## Quick Start

### 1. Install as Editable Package (Recommended for Development)

This allows you to modify the code and have changes immediately reflected without reinstalling:

```bash
cd /home/sonic/ICL_SMH/Research_compass_aftersubmit/image-as-brush

# Install in editable mode
pip install -e .
```

### 2. Build CUDA Extensions

After installing the package, build the CUDA extensions:

```bash
cd cuda_tile_rasterizer

# Build standard CUDA rasterizer
python setup.py clean
rm -rf build/ *.egg-info *.so
python setup.py build_ext --inplace

# Build FP16 version (optional, for memory optimization)
python setup_fp16.py clean
rm -rf build/ *.egg-info *.so
python setup_fp16.py build_ext --inplace

# Build PSD export extension (optional, for PSD layer export)
python setup_psd_export.py clean
rm -rf build/ *.egg-info *.so
python setup_psd_export.py build_ext --inplace

cd ..
```

### 3. Verify Installation

```python
import image_as_brush
print(f"image-as-brush version: {image_as_brush.__version__}")
print(f"CUDA available: {image_as_brush.check_cuda_available()}")
```

## Using in Another Project (e.g., SVGDreamer)

### Method 1: Editable Install (Best for Development)

From your SVGDreamer environment:

```bash
# Activate your conda environment
conda activate svgsplat

# Install image-as-brush as editable package
pip install -e /path/to/image-as-brush
```

Now you can use it in your code:

```python
from image_as_brush import SimpleTileRenderer, PrimitiveLoader

# Use the renderer
loader = PrimitiveLoader("circle.svg", output_width=256, device='cuda')
renderer = SimpleTileRenderer(canvas_size=(512, 512), S=loader.load_alpha_bitmap())
```

### Method 2: Add to requirements.txt

Add this line to your project's `requirements.txt`:

```txt
-e ../image-as-brush
```

Then install:

```bash
pip install -r requirements.txt
```

## Specifying GPU Architecture

If you have a specific GPU or CUDA version:

```bash
cd cuda_tile_rasterizer

# Example for RTX 3090 (compute capability 8.6) with CUDA 12.1
TORCH_CUDA_ARCH_LIST="8.6" CUDA_HOME=/usr/local/cuda-12.1 \
python setup.py clean && rm -rf build/ *.egg-info *.so && \
TORCH_CUDA_ARCH_LIST="8.6" CUDA_HOME=/usr/local/cuda-12.1 \
python setup.py build_ext --inplace
```

Common compute capabilities:
- RTX 30 series (3090, 3080): `8.6`
- RTX 40 series (4090, 4080): `8.9`
- A100: `8.0`
- V100: `7.0`

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the right environment:

```bash
which python
# Should point to your conda environment
```

### CUDA Compilation Errors

If CUDA compilation fails:

1. Check CUDA version: `nvcc --version`
2. Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
3. Make sure they match or are compatible

### Module Not Found

If Python can't find the module:

```bash
# Check if installed
pip list | grep image-as-brush

# Reinstall
pip uninstall image-as-brush
pip install -e .
```

## Uninstallation

```bash
pip uninstall image-as-brush
```

Note: This won't delete the source code, just removes it from pip's registry.

## Development Workflow

When developing with editable install:

1. Make changes to code in `/image-as-brush/`
2. Changes are immediately available (no reinstall needed)
3. If you add new files, they're automatically included
4. If you modify CUDA code, rebuild with `python setup.py build_ext --inplace`

## Updating from Git

If your team updates the code:

```bash
cd /path/to/image-as-brush
git pull

# Rebuild CUDA extensions if they changed
cd cuda_tile_rasterizer
python setup.py build_ext --inplace
cd ..
```

Your code using `import image_as_brush` will automatically use the updated version.

# DiffBMP

## Overview

DiffBMP provides tools and methodologies for fast differentiable painting with any image primitives. It includes initialization techniques, rendering methods, and evaluation scripts aimed at comparing the effectiveness and quality of various vector art generation algorithms.


## Installation

### From PyPI (Recommended)

```bash
pip install pydiffbmp
```

**System Requirements:**
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Platform**: Linux x86_64 (pre-built CUDA binaries included)
- **GPU**: NVIDIA GPU with CUDA capability 8.6+ (RTX 3090, A100, RTX 4090, L40S, etc.)
- **CUDA Driver**: 11.8+ (no CUDA toolkit installation required)
- **PyTorch**: 1.13.0+

Additional system dependencies:
```bash
sudo apt-get install poppler-utils
```

### From Source (Development)

For development or if you need to customize CUDA kernels:

```bash
git clone https://github.com/smhongok/diffbmp.git
cd diffbmp
pip install -r requirements.txt

# Build CUDA extensions (requires CUDA toolkit 12.3+)
./build_wheels.sh

# Install in editable mode
pip install -e .
```

**Build Requirements** (only for source installation):
- CUDA Toolkit 12.3+ (recommended)
- C++ compiler (GCC 9.4.0+ recommended)
- PyTorch 1.13.0+

## Usage

### Building
Tips :
- Recommended CUDA version is 12.3.
- Recommended C++ version is 9.4.0.

To build:

```bash
cd cuda_tile_rasterizer && python setup.py clean && rm -rf build/ *.egg-info *.so && python setup.py build_ext --inplace && cd ..
```

To build for fp16:
```bash
cd cuda_tile_rasterizer && python setup_fp16.py clean && rm -rf build/ *.egg-info *.so && python setup_fp16.py build_ext --inplace && cd ..
```

To build for psd_exporter:
```bash
cd cuda_tile_rasterizer && python setup_psd_export.py clean && rm -rf build/ *.egg-info *.so && python setup_psd_export.py build_ext --inplace && cd ..
```

If you want to specify the spec of your GPU or CUDA version:
```bash
cd cuda_tile_rasterizer && TORCH_CUDA_ARCH_LIST="8.6" CUDA_HOME=/usr/local/cuda-12.1 python setup_fp16.py clean && rm -rf build/ *.egg-info *.so && TORCH_CUDA_ARCH_LIST="8.6" CUDA_HOME=/usr/local/cuda-12.1 python setup_fp16.py build_ext --inplace && cd ..
```

If you have some errors when you build, remove followings and do above commands again:
`cuda_tile_rasterizer/cuda_tile_rasterizer`, `cuda_tile_rasterizer/cuda_tile_rasterizer_fp16`, `cuda_tile_rasterizer/build` 


### Running the Main Script

Execute the main script with a configuration file:

```
python main.py --config configs/default.json
```

### Evaluating Methods

To compare different circle-art methods:

```
python compare_methods.py --config configs/default.json
```

### Running Evaluations

To execute specific evaluations on generated results:

```
python run_evaluation.py
```

### Dynamic DiffBMP Visualization Methods

To visualize frame-by-frame primitive (x,y) movement:

paste the code below in configs/sequential.json
```
sequential_debug = {
    "route_visualization": {
        "enabled": True,
        "export_path": "./outputs/seq_test"
    }
}
```

To visualize per-pixel gradient of opacity-reduced primitives (by our heuristic):

paste the code below in configs/sequential.json
```
sequential_debug = {
    "gradient_visualization": {
        "enabled": True,
        "enable_non_problematic_primitive": False,
        "gradient_threshold": 1e-15,
        "save_dir": "./outputs/vis_class/debug_gradients_sequential"
    }
}
```

To visualize difference mask between frames (which becomes critical input for our heuristic):

paste the code below in configs/sequential.json
```
sequential_debug = {
    "diff_mask": {
        "enabled": True,
        "export_path": "./outputs/vis_class/diff_mask_sequential"
    }
}
```

Complete debugging config:
```
sequential_debug = {
    "gradient_visualization": {
        "enabled": True,
        "enable_non_problematic_primitive": False,
        "gradient_threshold": 1e-15,
        "save_dir": "./outputs/vis_class/debug_gradients_sequential"
    },
    "diff_mask": {
        "enabled": True,
        "export_path": "./outputs/vis_class/diff_mask_sequential"
    },
    "route_visualization": {
        "enabled": True,
        "export_path": "./outputs/seq_test"
    }
}
```


## Assets

* Put any predefined SVG templates based on 'path' tag in the `assets/svg` directory.
* Put fonts for rendering in `assets/font`.

## Examples and Datasets

The `images` directory contains various datasets and sample images categorized for quick testing:

* Artwork
* Nature
* Movie Posters
* Benchmark images (BSDS500, CelebA)

## Contributing

Feel free to submit pull requests or report issues to enhance the functionality or resolve problems.

## License

Please do not distribute. This is for the purpose of anonymous review.

---

Enjoy creating beautiful SVG drawing artwork!

## Testing

To test all default*.json configs:

```bash
python test_configs.py --gpu 6 --no-wandb
```
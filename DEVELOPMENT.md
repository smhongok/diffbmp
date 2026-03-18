# DiffBMP Development Guide

This guide provides detailed information for developers working with the DiffBMP codebase.

---

## Table of Contents

- [Building from Source](#building-from-source)
- [Running the Research Code](#running-the-research-code)
- [Configuration](#configuration)
- [Evaluation and Testing](#evaluation-and-testing)
- [Debugging and Visualization](#debugging-and-visualization)
- [Project Structure](#project-structure)

---

## Building from Source

### Detailed Build Instructions

#### Prerequisites

- CUDA Toolkit 12.3+ (recommended)
- C++ compiler (GCC 9.4.0+ recommended)
- PyTorch 1.13.0+
- Python 3.8-3.11

#### Building CUDA Extensions

**FP32 CUDA Extension:**
```bash
cd cuda_tile_rasterizer
python setup.py clean
rm -rf build/ *.egg-info *.so
python setup.py build_ext --inplace
cd ..
```

**FP16 CUDA Extension:**
```bash
cd cuda_tile_rasterizer
python setup_fp16.py clean
rm -rf build/ *.egg-info *.so
python setup_fp16.py build_ext --inplace
cd ..
```

**PSD Exporter:**
```bash
cd cuda_tile_rasterizer
python setup_psd_export.py clean
rm -rf build/ *.egg-info *.so
python setup_psd_export.py build_ext --inplace
cd ..
```

#### Specifying GPU Architecture and CUDA Version

```bash
cd cuda_tile_rasterizer
TORCH_CUDA_ARCH_LIST="8.6" CUDA_HOME=/usr/local/cuda-12.1 python setup_fp16.py clean
rm -rf build/ *.egg-info *.so
TORCH_CUDA_ARCH_LIST="8.6" CUDA_HOME=/usr/local/cuda-12.1 python setup_fp16.py build_ext --inplace
cd ..
```

#### Troubleshooting Build Errors

If you encounter build errors, try removing the following directories and rebuilding:
- `cuda_tile_rasterizer/cuda_tile_rasterizer`
- `cuda_tile_rasterizer/cuda_tile_rasterizer_fp16`
- `cuda_tile_rasterizer/build`

---

## Running the Research Code

### Basic Usage

Execute the main script with a configuration file:

```bash
python main.py --config configs/default.json
```

### Available Configuration Files

The `configs/` directory contains various configuration files for different experiments:
- `default.json` - Basic configuration
- `sequential.json` - Sequential rendering configuration
- And more...

---

## Configuration

Configuration files are in JSON format and control various aspects of the rendering process:

### Example Configuration Structure

```json
{
  "preprocessing": {
    "final_width": [256],
    "img_path": ["images/person/vangogh1.jpg"]
  },
  "primitive": {
    "primitive_file": ["fingerprint.jpg"],
    "output_width": 64
  },
  "initialization": {
    "initializer": "structure_aware",
    "N": [100],
    "radii_min": 5,
    "radii_max": 30
  },
  "optimization": {
    "use_fp16": true,
    "num_iterations": 500,
    "learning_rate": {
      "default": 0.1
    }
  }
}
```

---

## Evaluation and Testing

### Comparing Methods

To compare different rendering methods:

```bash
python compare_methods.py --config configs/default.json
```

### Running Evaluations

To execute specific evaluations on generated results:

```bash
python run_evaluation.py
```

### Testing All Configurations

To test all `default*.json` configs:

```bash
python test_configs.py --gpu 6 --no-wandb
```

---

## Debugging and Visualization

### Dynamic DiffBMP Visualization

#### Route Visualization

Visualize frame-by-frame primitive (x,y) movement:

Add to `configs/sequential.json`:
```json
{
  "sequential_debug": {
    "route_visualization": {
      "enabled": true,
      "export_path": "./outputs/seq_test"
    }
  }
}
```

#### Gradient Visualization

Visualize per-pixel gradient of opacity-reduced primitives:

```json
{
  "sequential_debug": {
    "gradient_visualization": {
      "enabled": true,
      "enable_non_problematic_primitive": false,
      "gradient_threshold": 1e-15,
      "save_dir": "./outputs/vis_class/debug_gradients_sequential"
    }
  }
}
```

#### Difference Mask Visualization

Visualize difference mask between frames:

```json
{
  "sequential_debug": {
    "diff_mask": {
      "enabled": true,
      "export_path": "./outputs/vis_class/diff_mask_sequential"
    }
  }
}
```

#### Complete Debugging Configuration

```json
{
  "sequential_debug": {
    "gradient_visualization": {
      "enabled": true,
      "enable_non_problematic_primitive": false,
      "gradient_threshold": 1e-15,
      "save_dir": "./outputs/vis_class/debug_gradients_sequential"
    },
    "diff_mask": {
      "enabled": true,
      "export_path": "./outputs/vis_class/diff_mask_sequential"
    },
    "route_visualization": {
      "enabled": true,
      "export_path": "./outputs/seq_test"
    }
  }
}
```

---

## Project Structure

```
diffbmp/
├── pydiffbmp/                    # Main package
│   ├── core/                     # Core rendering logic
│   │   ├── initializer/          # Initialization methods
│   │   └── renderer/             # Rendering engines
│   └── util/                     # Utility functions
├── cuda_tile_rasterizer/         # CUDA extensions
│   ├── cuda_kernels/             # CUDA kernel implementations
│   ├── setup.py                  # FP32 build script
│   ├── setup_fp16.py             # FP16 build script
│   └── setup_psd_export.py       # PSD exporter build script
├── configs/                      # Configuration files
├── images/                       # Test images and datasets
├── assets/                       # SVG templates and fonts
├── main.py                       # Main entry point
└── requirements.txt              # Python dependencies
```

### Key Directories

- **`pydiffbmp/core/initializer/`**: Different initialization strategies
  - `structure_aware_initializer.py` - Structure-aware initialization
  - `random_initializer.py` - Random initialization
  - `sequential_initializer.py` - Sequential initialization

- **`pydiffbmp/core/renderer/`**: Rendering implementations
  - `simple_tile_renderer.py` - Tile-based renderer
  - `vector_renderer.py` - Base vector renderer
  - `sequential_renderer.py` - Sequential rendering

- **`pydiffbmp/util/`**: Utility modules
  - `primitive_loader.py` - Load bitmap primitives
  - `svg_converter.py` - SVG conversion utilities
  - `psd_exporter.py` - PSD export functionality

---

## Assets and Datasets

### Assets

- **SVG Templates**: Put predefined SVG templates (based on 'path' tag) in `assets/svg/`
- **Fonts**: Put fonts for rendering in `assets/font/`

### Example Datasets

The `images/` directory contains various datasets and sample images:
- Artwork
- Nature scenes
- Movie posters
- Benchmark images (BSDS500, CelebA)

---

## Advanced Topics

### Custom Primitives

To use custom bitmap primitives:

1. Prepare your primitive image (PNG, JPG, etc.)
2. Place it in a accessible directory
3. Reference it in your configuration:

```json
{
  "primitive": {
    "primitive_file": ["path/to/your/primitive.png"],
    "output_width": 64
  }
}
```

### Custom Loss Functions

Loss functions are defined in `pydiffbmp/util/loss_functions.py`. You can add custom loss functions and reference them in your configuration:

```json
{
  "optimization": {
    "loss_config": {
      "type": "custom_loss_name"
    }
  }
}
```

---

## Contributing

Contributions are welcome\! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

---

## Support

For questions or issues:
- Open an issue on [GitHub](https://github.com/smhongok/diffbmp/issues)
- Contact the authors (see main README.md)

---

**Last Updated**: March 2026

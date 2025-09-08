# Circle Art

## Overview

This project provides tools and methodologies for creating and evaluating svg-based vector art from images. It includes initialization techniques, rendering methods, and evaluation scripts aimed at comparing the effectiveness and quality of various svgsplat art generation algorithms.


## Requirements

Install dependencies using pip:

```
pip install -r requirements.txt
```

Install Poppler

```
sudo apt-get install poppler-utils
```

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
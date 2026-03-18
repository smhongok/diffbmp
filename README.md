<h1 align="center">DiffBMP: Differentiable Rendering with Bitmap Primitives</h1>

<p align="center">
  <a href="https://diffbmp.com"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2602.22625"><img src="https://img.shields.io/badge/arXiv-2602.22625-b31b1b.svg" alt="arXiv"></a>
  <a href="https://pypi.org/project/pydiffbmp/"><img src="https://img.shields.io/pypi/v/pydiffbmp" alt="PyPI"></a>
  <a href="https://github.com/smhongok/diffbmp"><img src="https://img.shields.io/github/stars/smhongok/diffbmp?style=social" alt="GitHub stars"></a>
</p>

<p align="center"><strong>Accepted to CVPR 2026</strong></p>

## Authors

[**Seongmin Hong**](https://smhongok.github.io)<sup>1,*</sup>, 
[**Junghun James Kim**](https://www.linkedin.com/in/james-hun-kim-a4682b106/)<sup>2,*</sup>, 
[**Daehyeop Kim**](https://www.linkedin.com/in/daehyeop-kim-41536530a/)<sup>3</sup>, 
[**Insoo Chung**](https://www.linkedin.com/in/insoo-chung-07a242358/)<sup>3</sup>, 
[**Se Young Chun**](https://icl.snu.ac.kr/)<sup>1,2,3,†</sup>

<sup>1</sup> INMC, <sup>2</sup> IPAI, <sup>3</sup> Dept. of ECE, Seoul National University, Republic of Korea

<sup>*</sup> Co-first authors &nbsp;&nbsp; <sup>†</sup> Corresponding author

**Contact**: {smhongok, jonghean12, 2012abcd, insoo_chung, sychun}@snu.ac.kr

---

## Overview

DiffBMP is a fast differentiable rendering framework for creating vector art with arbitrary bitmap primitives. Unlike traditional vector graphics that rely on geometric primitives, DiffBMP enables the use of any image as a primitive, opening new possibilities for artistic expression and image synthesis.


## 🚀 Installation

You have **two options** for installation:

### Option 1: PyPI Installation (Linux/WSL Only - Simple & Fast)

Best for users who want to quickly use DiffBMP without building from source.

```bash
# Install PyTorch first (if not already installed)
# Note: You can use any CUDA version compatible with your system (cu118, cu121, cu124, etc.)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install pydiffbmp
pip install pydiffbmp

# Install system dependencies
sudo apt-get install poppler-utils

# Install additional Python dependencies (if needed)
pip install tqdm nvidia-ml-py fonttools svgwrite svgpathtools scour cairosvg psd-tools piq moviepy pyvista pandas scikit-image scipy
```

### System Requirements

- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Platform**: Linux x86_64 (Windows/macOS not supported for PyPI installation)
- **GPU**: NVIDIA GPU with CUDA capability 8.6+ (RTX 3090, A100, RTX 4090, L40S, etc.)
- **CUDA Driver**: 11.8+ (**CUDA toolkit NOT required** for PyPI installation)
- **PyTorch**: 1.13.0+

> ⚠️ **Note**: Pre-built binaries are currently available for Linux only. Windows users should use WSL2 or build from source.

### Option 2: Build from Source (All Platforms - Flexible)

Best for developers, customization, or if you need to build on Windows/macOS.

```bash
# Clone the repository
git clone https://github.com/smhongok/diffbmp.git
cd diffbmp

# Install dependencies
pip install -r requirements.txt

# Build CUDA extensions
cd cuda_tile_rasterizer
python setup.py build_ext --inplace
python setup_fp16.py build_ext --inplace
python setup_psd_export.py build_ext --inplace
cd ..

# Install in editable mode
pip install -e .
```

**Build Requirements**:
- CUDA Toolkit 12.3+ (recommended)
- C++ compiler (GCC 9.4.0+ recommended)
- PyTorch 1.13.0+

For detailed build instructions and troubleshooting, see [DEVELOPMENT.md](DEVELOPMENT.md).

### Running the Original Research Code

The original research implementation is available in the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/smhongok/diffbmp.git
cd diffbmp

# Run with configuration
python main.py --config configs/default.json
```

For more details on research code, evaluation scripts, and advanced features, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## 📚 Citation

If you use DiffBMP in your research, please cite our paper:

```bibtex
@inproceedings{hong2026diffbmp,
  title={DiffBMP: Differentiable Rendering with Bitmap Primitives},
  author={Hong, Seongmin and Kim, Junghun James and Kim, Daehyeop and Chung, Insoo and Chun, Se Young},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For questions or issues, please open an issue on [GitHub](https://github.com/smhongok/diffbmp/issues).
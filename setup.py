"""
setup.py for DiffBMP
Make this package pip installable with: pip install -e .

This setup.py includes CUDA extension compilation.
"""

from setuptools import setup, find_packages
import os
import sys

# Check if CUDA is available
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    CUDA_AVAILABLE = torch.cuda.is_available()
    
    # For PyTorch 1.x with CUDA 12.x mismatch, skip version check
    if CUDA_AVAILABLE:
        torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
        if torch_version < (2, 0):
            # PyTorch 1.x - may need to skip CUDA version check
            import torch.utils.cpp_extension
            try:
                # Try to get CUDA version
                cuda_version = torch.version.cuda
                if cuda_version:
                    cuda_major = int(cuda_version.split('.')[0])
                    # If PyTorch was built with CUDA 11.x but system has 12.x
                    if cuda_major < 12:
                        def _skip_cuda_version_check(*args, **kwargs):
                            print("Warning: Skipping CUDA version check (PyTorch 1.x with potential CUDA mismatch)")
                        torch.utils.cpp_extension._check_cuda_version = _skip_cuda_version_check
            except:
                pass
        
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: PyTorch not found. Install PyTorch first: pip install torch")
    sys.exit(1)

# Read requirements
def get_requirements():
    requirements = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and git+ URLs
            if line and not line.startswith('#') and not line.startswith('git+'):
                # Handle commented out packages
                if line.startswith('# '):
                    continue
                requirements.append(line)
    return requirements

# Setup CUDA extensions
def get_cuda_extensions():
    """Build CUDA extensions if available.
    
    Note: CUDA extensions should be built separately using:
      cd cuda_tile_rasterizer && python setup.py build_ext --inplace
      cd cuda_tile_rasterizer && python setup_fp16.py build_ext --inplace
    
    This function returns an empty list to skip building during pip install.
    """
    # Skip CUDA extension build - they should be built separately
    return []
    
    # Original code below (disabled)
    if not CUDA_AVAILABLE:
        print("Warning: CUDA not available. Skipping CUDA extension build.")
        print("  DiffBMP will still work but without GPU acceleration.")
        return []
    
    extensions = []
    cuda_dir = 'cuda_tile_rasterizer'
    
    # Main CUDA extension
    if os.path.exists(cuda_dir):
        # Ensure output directory exists
        os.makedirs(os.path.join(cuda_dir, 'cuda_tile_rasterizer'), exist_ok=True)
        
        extensions.append(
            CUDAExtension(
                name='cuda_tile_rasterizer._C',
                sources=[
                    os.path.join(cuda_dir, 'ext.cpp'),
                    os.path.join(cuda_dir, 'tile_rasterize.cu'),
                    os.path.join(cuda_dir, 'cuda_kernels/tile_common.cu'),
                    os.path.join(cuda_dir, 'cuda_kernels/tile_forward.cu'),
                    os.path.join(cuda_dir, 'cuda_kernels/tile_backward.cu'),
                ],
                include_dirs=[
                    cuda_dir,
                    os.path.join(cuda_dir, 'cuda_kernels'),
                ],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': [
                        '-O3',
                        '--use_fast_math',
                        '-Xptxas=-v',
                        '-gencode=arch=compute_86,code=sm_86',  # RTX 3090, A100
                        '-gencode=arch=compute_89,code=sm_89',  # L40S, RTX 4090
                    ]
                }
            )
        )
    
    # FP16 extension (optional) - sources are in cuda_tile_rasterizer directory
    if os.path.exists(cuda_dir):
        fp16_sources = [
            os.path.join(cuda_dir, 'ext_fp16.cpp'),
            os.path.join(cuda_dir, 'tile_rasterize_fp16.cpp'),
            os.path.join(cuda_dir, 'cuda_kernels/tile_common_fp16.cu'),
            os.path.join(cuda_dir, 'cuda_kernels/tile_forward_fp16.cu'),
            os.path.join(cuda_dir, 'cuda_kernels/tile_backward_fp16.cu'),
        ]
        
        # Check if FP16 source files exist
        if all(os.path.exists(src) for src in fp16_sources):
            try:
                # Ensure output directory exists
                os.makedirs(os.path.join(cuda_dir, 'cuda_tile_rasterizer_fp16'), exist_ok=True)
                
                extensions.append(
                    CUDAExtension(
                        name='cuda_tile_rasterizer_fp16._C_fp16',
                        sources=fp16_sources,
                        include_dirs=[
                            cuda_dir,
                            os.path.join(cuda_dir, 'cuda_kernels'),
                        ],
                        extra_compile_args={
                            'cxx': ['-O3'],
                            'nvcc': [
                                '-O3',
                                '--use_fast_math',
                                '-Xptxas=-v',
                                '-gencode=arch=compute_86,code=sm_86',
                                '-gencode=arch=compute_89,code=sm_89',
                            ]
                        }
                    )
                )
                print("FP16 CUDA extension will be built")
            except Exception as e:
                print(f"Warning: FP16 extension build failed (optional): {e}")
        else:
            print("Warning: FP16 source files not found, skipping FP16 extension")
    
    return extensions

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pydiffbmp',
    version='0.1.0',
    description='DiffBMP: Fast Differentiable Painting with Any Image Primitives',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DiffBMP Team',
    author_email='',
    url='https://github.com/smhongok/diffbmp',
    
    # Auto-discover all packages
    packages=find_packages(include=['pydiffbmp', 'pydiffbmp.*']),
    
    # Install dependencies from requirements.txt
    install_requires=get_requirements(),
    
    # CUDA extensions
    ext_modules=get_cuda_extensions(),
    cmdclass={'build_ext': BuildExtension} if (CUDA_AVAILABLE and BuildExtension) else {},
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Additional metadata
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    
    # Keywords
    keywords='differentiable-rendering 2d-splatting image-synthesis pytorch cuda',
    
    # Include package data (images, configs, etc.)
    include_package_data=True,
    package_data={
        'pydiffbmp': ['*.png', '*.jpg', '*.svg'],
    },
    
    # Extras for optional features
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
        ],
        'video': [
            'moviepy',
            'pyvista',
            'pylrc',
        ],
        'clip': [
            'ftfy',
            'regex',
        ],
    },
)

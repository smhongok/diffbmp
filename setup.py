"""
setup.py for DiffBMP
Make this package pip installable with: pip install -e .

This setup.py includes CUDA extension compilation.
"""

from setuptools import setup, find_packages
import os
import sys

# For pre-built binary distribution, we don't need torch at build time
# torch will be installed as a dependency when users pip install pydiffbmp

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

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pydiffbmp',
    version='0.1.3',
    description='DiffBMP: Differentiable Rendering with Bitmap Primitives',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DiffBMP Team',
    author_email='',
    url='https://github.com/smhongok/diffbmp',
    
    # Explicitly list all packages including CUDA extensions
    packages=[
        'pydiffbmp',
        'pydiffbmp.util',
        'pydiffbmp.core',
        'pydiffbmp.core.initializer',
        'pydiffbmp.core.renderer',
        'cuda_tile_rasterizer',
        'cuda_tile_rasterizer.cuda_tile_rasterizer',
        'cuda_tile_rasterizer.cuda_tile_rasterizer_fp16',
    ],
    
    # Install dependencies from requirements.txt
    install_requires=get_requirements(),
    
    # CUDA extensions are pre-built and included as .so files
    # No compilation needed during pip install
    
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
        'cuda_tile_rasterizer': ['*.so'],  # psd_export_cuda.so
        'cuda_tile_rasterizer.cuda_tile_rasterizer': ['*.so'],  # _C.so (FP32)
        'cuda_tile_rasterizer.cuda_tile_rasterizer_fp16': ['*.so'],  # _C_fp16.so (FP16)
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

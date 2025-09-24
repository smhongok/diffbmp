
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

# Ensure cuda_tile_rasterizer directory exists
os.makedirs('cuda_tile_rasterizer', exist_ok=True)

setup(
    name='cuda_tile_rasterizer',
    ext_modules=[
        CUDAExtension(
            name='cuda_tile_rasterizer._C',
            sources=[
                'ext.cpp',
                'tile_rasterize.cu',
                'cuda_kernels/tile_common.cu',
                'cuda_kernels/tile_forward.cu',
                'cuda_kernels/tile_backward.cu',
            ],
            include_dirs=[
                '.',
                'cuda_kernels',
            ],
            extra_compile_args={
                'cxx': ['-O3'],  # Debug flags for C++
                'nvcc': [
                    '-O3',  # Debug flags for CUDA
                    '--use_fast_math',
                    '-Xptxas=-v',
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090, A100
                    '-gencode=arch=compute_89,code=sm_89',  # L40S, RTX 4090
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
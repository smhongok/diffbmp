import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Ensure cuda_tile_rasterizer directory exists
os.makedirs('cuda_tile_rasterizer_fp16', exist_ok=True)

setup(
    name='cuda_tile_rasterizer_fp16',
    ext_modules=[
        CUDAExtension(
            name='cuda_tile_rasterizer_fp16._C_fp16',
            sources=[
                'ext_fp16.cpp',
                'tile_rasterize_fp16.cpp',
                'cuda_kernels/tile_common_fp16.cu',
                'cuda_kernels/tile_forward_fp16.cu',
                'cuda_kernels/tile_backward_fp16.cu',
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
                    '-gencode=arch=compute_80,code=sm_80',  # A100 (Ampere)
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
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

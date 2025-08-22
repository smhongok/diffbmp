from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

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
                'cxx': ['-g', '-O0'],  # Debug flags for C++
                'nvcc': [
                    '-g',  # Debug flags for CUDA
                    '--generate-line-info',
                    '-Xptxas=-v'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
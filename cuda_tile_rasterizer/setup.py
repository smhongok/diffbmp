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
                'cuda_kernels/tile_forward.cu',
                'cuda_kernels/tile_backward.cu',
            ],
            include_dirs=[
                '.',
                'cuda_kernels',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xptxas=-v',
                    '--ptxas-options=-v',
                    '-lineinfo'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='cuda_tile_rasterizer_fp16',
    ext_modules=[
        CUDAExtension(
            name='cuda_tile_rasterizer_fp16._C',
            sources=[
                'ext_fp16.cpp',
                'tile_rasterize_fp16.cpp',
                'cuda_kernels/tile_forward_fp16.cu',
                'cuda_kernels/tile_backward_fp16.cu',
            ],
            include_dirs=[
                '.',
                'cuda_kernels',
            ],
            extra_compile_args={
                'cxx': ['-g', '-O0'],
                'nvcc': [
                    '-g',
                    '--generate-line-info',
                    '-Xptxas=-v',
                    '--expt-relaxed-constexpr'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)

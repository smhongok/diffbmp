from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get CUDA toolkit path
cuda_home = os.environ.get('CUDA_HOME') or '/usr/local/cuda'

# Define the extension
ext_modules = [
    CUDAExtension(
        name='psd_export_cuda',
        sources=[
            'psd_export_bindings.cpp',
            'cuda_kernels/psd_export_kernels.cu'
        ],
        include_dirs=[
            'cuda_kernels/',
            f'{cuda_home}/include',
        ],
        libraries=['cudart'],
        library_dirs=[f'{cuda_home}/lib64'],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-arch=sm_70',  # Adjust based on your GPU
                '-arch=sm_75',
                '-arch=sm_80',
                '-arch=sm_86',
            ]
        }
    )
]

setup(
    name='psd_export_cuda',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

#!/bin/bash

# Build script for PSD Export CUDA extension
echo "🔨 Building PSD Export CUDA Extension..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
python setup_psd_export.py clean
rm -rf build/ *.egg-info *.so

# Build with specified CUDA architecture and path
echo "🚀 Building extension..."
TORCH_CUDA_ARCH_LIST="8.6" CUDA_HOME=/usr/local/cuda-12.1 python setup_psd_export.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo "🧪 Running tests..."
    cd ..
    python test_psd_cuda.py
else
    echo "❌ Build failed!"
    exit 1
fi

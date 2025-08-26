#ifndef TILE_BACKWARD_FP16_H
#define TILE_BACKWARD_FP16_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FP16 version of the backward CUDA kernel
void CudaRasterizeTilesBackwardKernelFP16(
    const __half* grad_out_color,
    const __half* grad_out_alpha,
    const __half* means2D,
    const __half* radii,
    const __half* rotations,
    const __half* opacities,
    const __half* colors,
    const __half* primitive_templates,
    __half* grad_means2D,
    __half* grad_radii,
    __half* grad_rotations,
    __half* grad_opacities,
    __half* grad_colors,
    int num_primitives,
    int template_height,
    int template_width,
    int image_height,
    int image_width,
    int tile_size,
    __half sigma,
    int total_tiles);

#endif // TILE_BACKWARD_FP16_H

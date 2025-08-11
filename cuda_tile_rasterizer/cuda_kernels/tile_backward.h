#ifndef TILE_BACKWARD_H
#define TILE_BACKWARD_H

#include <cuda_runtime.h>

void CudaRasterizeTilesBackwardKernel(
    const float* grad_out_color,
    const float* grad_out_alpha,
    const float* means2D,
    const float* radii,
    const float* rotations,
    const float* opacities,
    const float* colors,
    const float* primitive_templates,
    float* grad_means2D,
    float* grad_radii,
    float* grad_rotations,
    float* grad_opacities,
    float* grad_colors,
    int num_primitives,
    int template_height,
    int template_width,
    int image_height,
    int image_width,
    int tile_size,
    float sigma,
    int total_tiles);

#endif // TILE_BACKWARD_H
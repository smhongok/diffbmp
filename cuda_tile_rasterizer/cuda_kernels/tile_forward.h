#ifndef TILE_FORWARD_H
#define TILE_FORWARD_H

#include <cuda_runtime.h>

void CudaRasterizeTilesForwardKernel(
    const float* means2D,
    const float* radii,
    const float* rotations,
    const float* opacities,
    const float* colors,
    const float* primitive_templates,
    float* out_color,
    float* out_alpha,
    int num_primitives,
    int num_templates,
    int template_height,
    int template_width,
    int image_height,
    int image_width,
    int tile_size,
    float sigma,
    float alpha_upper_bound,
    int total_tiles);

#endif // TILE_FORWARD_H
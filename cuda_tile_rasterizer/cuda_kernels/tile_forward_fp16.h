#ifndef TILE_FORWARD_FP16_H
#define TILE_FORWARD_FP16_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FP16 version of the CUDA kernel
void CudaRasterizeTilesForwardKernelFP16(
    const __half* means2D,
    const __half* radii,
    const __half* rotations,
    const __half* opacities,
    const __half* colors,
    const __half* primitive_templates,
    __half* out_color,
    __half* out_alpha,
    // Global memory arrays for transmit_over compositing
    __half* pixel_alphas,      // [image_height * image_width * max_prims_per_pixel]
    __half* pixel_colors_r,    // [image_height * image_width * max_prims_per_pixel]
    __half* pixel_colors_g,    // [image_height * image_width * max_prims_per_pixel]
    __half* pixel_colors_b,    // [image_height * image_width * max_prims_per_pixel]
    int* pixel_prim_counts,   // [image_height * image_width]
    int max_prims_per_pixel,
    int num_primitives,
    int num_templates,
    int template_height,
    int template_width,
    int image_height,
    int image_width,
    int tile_size,
    __half sigma,
    __half alpha_upper_bound,
    int total_tiles);

#endif // TILE_FORWARD_FP16_H

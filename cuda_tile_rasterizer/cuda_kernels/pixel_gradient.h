#ifndef PIXEL_GRADIENT_H
#define PIXEL_GRADIENT_H

#include <cuda_runtime.h>
#include <torch/extension.h>
#include "tile_common.h"

// CUDA kernel for computing per-pixel gradient magnitudes in parallel
__global__ void compute_per_pixel_gradient_kernel(
    // Input parameters
    const float* means2D,           // [N, 2] - primitive positions (x, y)
    const float* radii,             // [N] - primitive radii
    const float* rotations,         // [N] - primitive rotations
    const float* opacities,         // [N] - primitive opacities (logits)
    const float* colors,            // [N, 3] - primitive colors
    const float* primitive_templates, // [T, Ht, Wt] - template masks
    const int* global_bmp_sel,      // [N] - template selection indices
    
    // Target image
    const float* target_image,      // [H, W, 3] - target image
    
    // Tile assignment data
    const int* tile_offsets,        // [num_tiles + 1] - tile primitive offsets
    const int* tile_indices,        // [total_assignments] - primitive indices per tile
    
    // Configuration
    const TileConfig tile_config,
    const PrimitiveConfig prim_config,
    
    // Sampling configuration
    int pixels_per_tile,            // Number of pixels to sample per tile
    unsigned int* random_states,    // [H*W] - random states for sampling
    
    // Output
    float* gradient_magnitudes,     // [N] - output gradient magnitudes per primitive
    float* pixel_losses            // [H, W] - intermediate pixel losses for debugging
);

// Host function to launch the CUDA kernel
torch::Tensor compute_per_pixel_gradient_cuda(
    torch::Tensor means2D,
    torch::Tensor radii, 
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    torch::Tensor target_image,
    torch::Tensor tile_offsets,
    torch::Tensor tile_indices,
    int tile_size,
    float sigma,
    float alpha_upper_bound,
    int max_prims_per_pixel,
    int pixels_per_tile
);

#endif // PIXEL_GRADIENT_H

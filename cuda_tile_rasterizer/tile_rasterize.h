#ifndef TILE_RASTERIZE_H
#define TILE_RASTERIZE_H

#include <torch/extension.h>
#include <tuple>

// CUDA kernel wrapper declarations
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

// Forward declarations for CUDA functions
std::tuple<torch::Tensor, torch::Tensor> CudaRasterizeTilesForward(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    int image_height,
    int image_width,
    int tile_size,
    float sigma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CudaRasterizeTilesBackward(
    torch::Tensor grad_out_color,
    torch::Tensor grad_out_alpha,
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    int image_height,
    int image_width,
    int tile_size,
    float sigma);

#endif // TILE_RASTERIZE_H
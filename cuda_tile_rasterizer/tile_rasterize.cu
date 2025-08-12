#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernels/tile_forward.h"
#include "cuda_kernels/tile_backward.h"

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
    float sigma) {
    
    const int num_primitives = means2D.size(0);
    const int num_tiles_x = (image_width + tile_size - 1) / tile_size;
    const int num_tiles_y = (image_height + tile_size - 1) / tile_size;
    const int total_tiles = num_tiles_x * num_tiles_y;
    
    // Create output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(means2D.device());
    torch::Tensor out_color = torch::zeros({image_height, image_width, 3}, options);
    torch::Tensor out_alpha = torch::zeros({image_height, image_width}, options);
    
    // Launch CUDA kernel with alpha_upper_bound=0.5 (match PyTorch default)
    CudaRasterizeTilesForwardKernel(
        means2D.data_ptr<float>(),
        radii.data_ptr<float>(),
        rotations.data_ptr<float>(),
        opacities.data_ptr<float>(),
        colors.data_ptr<float>(),
        primitive_templates.data_ptr<float>(),
        out_color.data_ptr<float>(),
        out_alpha.data_ptr<float>(),
        num_primitives,
        primitive_templates.size(0), // num_templates
        primitive_templates.size(1), // template_height
        primitive_templates.size(2), // template_width
        image_height,
        image_width,
        tile_size,
        sigma,
        0.5f, // alpha_upper_bound (match PyTorch default)
        total_tiles
    );
    
    return std::make_tuple(out_color, out_alpha);
}

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
    float sigma) {
    
    const int num_primitives = means2D.size(0);
    const int total_tiles = ((image_width + tile_size - 1) / tile_size) * 
                           ((image_height + tile_size - 1) / tile_size);
    
    // Create gradient tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(means2D.device());
    torch::Tensor grad_means2D = torch::zeros_like(means2D);
    torch::Tensor grad_radii = torch::zeros_like(radii);
    torch::Tensor grad_rotations = torch::zeros_like(rotations);
    torch::Tensor grad_opacities = torch::zeros_like(opacities);
    torch::Tensor grad_colors = torch::zeros_like(colors);
    
    // Launch CUDA backward kernel
    CudaRasterizeTilesBackwardKernel(
        grad_out_color.data_ptr<float>(),
        grad_out_alpha.data_ptr<float>(),
        means2D.data_ptr<float>(),
        radii.data_ptr<float>(),
        rotations.data_ptr<float>(),
        opacities.data_ptr<float>(),
        colors.data_ptr<float>(),
        primitive_templates.data_ptr<float>(),
        grad_means2D.data_ptr<float>(),
        grad_radii.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        num_primitives,
        primitive_templates.size(1), // template_height
        primitive_templates.size(2), // template_width
        image_height,
        image_width,
        tile_size,
        sigma,
        total_tiles
    );
    
    return std::make_tuple(grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors);
}

#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include "cuda_kernels/tile_forward.h"
#include "cuda_kernels/tile_backward.h"

#define DEBUG_CUDA_KERNELS 1

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
    
    // Debug: Print input tensor info
#if DEBUG_CUDA_KERNELS
    printf("C++ Wrapper: num_primitives=%d, image_size=%dx%d, tile_size=%d, total_tiles=%d\n",
           num_primitives, image_width, image_height, tile_size, total_tiles);
    printf("C++ Wrapper: means2D shape=%s, primitive_templates shape=%s\n",
           means2D.sizes().vec().data(), primitive_templates.sizes().vec().data());
#endif
    
    // Create output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(means2D.device());
    torch::Tensor out_color = torch::zeros({image_height, image_width, 3}, options);
    torch::Tensor out_alpha = torch::zeros({image_height, image_width}, options);
    
    // Launch CUDA kernel with alpha_upper_bound=0.5 (match PyTorch default)
#if DEBUG_CUDA_KERNELS
    printf("C++ Wrapper: Launching CUDA kernel...\n");
#endif
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
#if DEBUG_CUDA_KERNELS
    printf("C++ Wrapper: CUDA kernel completed\n");

    // Debug: Print output tensor info
    printf("C++ Wrapper: Output color range=[%.4f,%.4f], alpha range=[%.4f,%.4f]\n",
           out_color.min().item<float>(), out_color.max().item<float>(),
           out_alpha.min().item<float>(), out_alpha.max().item<float>());
#endif
    
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

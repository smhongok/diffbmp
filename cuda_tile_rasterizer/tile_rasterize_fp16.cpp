#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include "cuda_kernels/tile_forward_fp16.h"
#include "cuda_kernels/tile_backward_fp16.h"

#define DEBUG_CUDA_KERNELS_FP16 0

std::tuple<torch::Tensor, torch::Tensor> CudaRasterizeTilesForwardFP16(
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
#if DEBUG_CUDA_KERNELS_FP16
    printf("C++ Wrapper FP16: num_primitives=%d, image_size=%dx%d, tile_size=%d, total_tiles=%d\n",
           num_primitives, image_width, image_height, tile_size, total_tiles);
    printf("C++ Wrapper FP16: means2D shape=%s, primitive_templates shape=%s\n",
           means2D.sizes().vec().data(), primitive_templates.sizes().vec().data());
#endif
    
    // Create output tensors in FP16
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(means2D.device());
    torch::Tensor out_color = torch::zeros({image_height, image_width, 3}, options);
    torch::Tensor out_alpha = torch::zeros({image_height, image_width}, options);
    
    // Launch CUDA kernel with alpha_upper_bound=0.5 (match PyTorch default)
#if DEBUG_CUDA_KERNELS_FP16
    printf("C++ Wrapper FP16: Launching CUDA kernel...\n");
#endif
    // Allocate global memory for transmit_over compositing
    int max_prims_per_pixel = 256;  // Maximum primitives per pixel
    int total_pixels = image_height * image_width;
    int total_alpha_size = total_pixels * max_prims_per_pixel;
    int total_color_size = total_pixels * max_prims_per_pixel;
    
    __half* pixel_alphas;
    __half* pixel_colors_r;
    __half* pixel_colors_g;
    __half* pixel_colors_b;
    int* pixel_prim_counts;
    
    cudaMalloc(&pixel_alphas, total_alpha_size * sizeof(__half));
    cudaMalloc(&pixel_colors_r, total_color_size * sizeof(__half));
    cudaMalloc(&pixel_colors_g, total_color_size * sizeof(__half));
    cudaMalloc(&pixel_colors_b, total_color_size * sizeof(__half));
    cudaMalloc(&pixel_prim_counts, total_pixels * sizeof(int));
    
    // Initialize arrays to zero
    cudaMemset(pixel_alphas, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_colors_r, 0, total_color_size * sizeof(__half));
    cudaMemset(pixel_colors_g, 0, total_color_size * sizeof(__half));
    cudaMemset(pixel_colors_b, 0, total_color_size * sizeof(__half));
    cudaMemset(pixel_prim_counts, 0, total_pixels * sizeof(int));
    
    CudaRasterizeTilesForwardKernelFP16(
        reinterpret_cast<const __half*>(means2D.data_ptr()),
        reinterpret_cast<const __half*>(radii.data_ptr()),
        reinterpret_cast<const __half*>(rotations.data_ptr()),
        reinterpret_cast<const __half*>(opacities.data_ptr()),
        reinterpret_cast<const __half*>(colors.data_ptr()),
        reinterpret_cast<const __half*>(primitive_templates.data_ptr()),
        reinterpret_cast<__half*>(out_color.data_ptr()),
        reinterpret_cast<__half*>(out_alpha.data_ptr()),
        // Global memory arrays for transmit_over compositing
        pixel_alphas,      // [image_height * image_width * max_prims_per_pixel]
        pixel_colors_r,    // [image_height * image_width * max_prims_per_pixel]
        pixel_colors_g,    // [image_height * image_width * max_prims_per_pixel]
        pixel_colors_b,    // [image_height * image_width * max_prims_per_pixel]
        pixel_prim_counts, // [image_height * image_width]
        max_prims_per_pixel,
        num_primitives,
        primitive_templates.size(0), // num_templates
        primitive_templates.size(1), // template_height
        primitive_templates.size(2), // template_width
        image_height,
        image_width,
        tile_size,
        __float2half(sigma),
        __float2half(0.5f), // alpha_upper_bound (match PyTorch default)
        total_tiles
    );
    
    // Free allocated memory
    cudaFree(pixel_alphas);
    cudaFree(pixel_colors_r);
    cudaFree(pixel_colors_g);
    cudaFree(pixel_colors_b);
    cudaFree(pixel_prim_counts);
#if DEBUG_CUDA_KERNELS_FP16
    printf("C++ Wrapper FP16: CUDA kernel completed\n");

    // Debug: Print output tensor info
    printf("C++ Wrapper FP16: Output color range=[%.4f,%.4f], alpha range=[%.4f,%.4f]\n",
           out_color.min().item<float>(), out_color.max().item<float>(),
           out_alpha.min().item<float>(), out_alpha.max().item<float>());
#endif
    
    return std::make_tuple(out_color, out_alpha);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CudaRasterizeTilesBackwardFP16(
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
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(means2D.device());
    torch::Tensor grad_means2D = torch::zeros_like(means2D);
    torch::Tensor grad_radii = torch::zeros_like(radii);
    torch::Tensor grad_rotations = torch::zeros_like(rotations);
    torch::Tensor grad_opacities = torch::zeros_like(opacities);
    torch::Tensor grad_colors = torch::zeros_like(colors);
    
    // Launch CUDA backward kernel
    CudaRasterizeTilesBackwardKernelFP16(
        reinterpret_cast<const __half*>(grad_out_color.data_ptr()),
        reinterpret_cast<const __half*>(grad_out_alpha.data_ptr()),
        reinterpret_cast<const __half*>(means2D.data_ptr()),
        reinterpret_cast<const __half*>(radii.data_ptr()),
        reinterpret_cast<const __half*>(rotations.data_ptr()),
        reinterpret_cast<const __half*>(opacities.data_ptr()),
        reinterpret_cast<const __half*>(colors.data_ptr()),
        reinterpret_cast<const __half*>(primitive_templates.data_ptr()),
        reinterpret_cast<__half*>(grad_means2D.data_ptr()),
        reinterpret_cast<__half*>(grad_radii.data_ptr()),
        reinterpret_cast<__half*>(grad_rotations.data_ptr()),
        reinterpret_cast<__half*>(grad_opacities.data_ptr()),
        reinterpret_cast<__half*>(grad_colors.data_ptr()),
        num_primitives,
        primitive_templates.size(1), // template_height
        primitive_templates.size(2), // template_width
        image_height,
        image_width,
        tile_size,
        __float2half(sigma),
        total_tiles
    );
    
    return std::make_tuple(grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors);
}

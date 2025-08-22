#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include "tile_rasterize.h"
#include "cuda_kernels/tile_forward.h"
#include "cuda_kernels/tile_backward.h"
#include "cuda_kernels/tile_common.h"

#define DEBUG_CUDA_KERNELS 1

// Global instance for Python binding
std::shared_ptr<TileRasterizer> global_tile_rasterizer = nullptr;

// TileRasterizer class implementation
TileRasterizer::TileRasterizer(int image_h, int image_w, int tile_sz, float sig, float alpha_ub, int max_prims, int num_prims) 
    : image_height(image_h), image_width(image_w), tile_size(tile_sz), sigma(sig), 
      alpha_upper_bound(alpha_ub), max_prims_per_pixel(max_prims), num_primitives(num_prims),
      memory_allocated(false) {

    pixel_alphas = nullptr;
    pixel_colors_r = nullptr;
    pixel_colors_g = nullptr;
    pixel_colors_b = nullptr;
    pixel_T_values = nullptr;
    pixel_prim_counts = nullptr;
    sigma_inv = nullptr;
    grad_sigma = nullptr;
    d_tile_offsets = nullptr;
    d_tile_indices = nullptr;
    tile_offsets_size = 0;
    tile_indices_size = 0;
    
    // Initialize gradient pointers to nullptr
    out_color = nullptr;
    out_alpha = nullptr;
    grad_means2D = nullptr;
    grad_radii = nullptr;
    grad_rotations = nullptr;
    grad_opacities = nullptr;
    grad_colors = nullptr;
    
    allocateMemory();
}

TileRasterizer::~TileRasterizer() {
    freeMemory();
}

void TileRasterizer::allocateMemory() {
    if (memory_allocated) return;
    
    int total_pixels = image_height * image_width;
    int total_alpha_size = total_pixels * max_prims_per_pixel;
    
    cudaError_t err;
    
    err = cudaMalloc(&pixel_alphas, total_alpha_size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_alphas: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_colors_r, total_alpha_size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_colors_r: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_colors_g, total_alpha_size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_colors_g: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_colors_b, total_alpha_size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_colors_b: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_T_values, total_alpha_size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_T_values: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_prim_counts, total_pixels * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_prim_counts: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&sigma_inv, num_primitives * 4 * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate sigma_inv: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_sigma, num_primitives * 4 * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_sigma: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    
    // Allocate tile arrays for backward pass
    int num_tiles = ((image_width + tile_size - 1) / tile_size) * ((image_height + tile_size - 1) / tile_size);
    tile_offsets_size = num_tiles + 1;
    tile_indices_size = num_tiles * num_primitives;  // Each tile can have all primitives
    
    // Allocate device arrays
    err = cudaMalloc(&d_tile_offsets, tile_offsets_size * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate d_tile_offsets: %s\n", cudaGetErrorString(err));        
        throw std::runtime_error("CUDA memory allocation failed");
    }
    
    err = cudaMalloc(&d_tile_indices, tile_indices_size * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate d_tile_indices: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }

    err = cudaMalloc(&out_color, total_pixels * 3 * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate out_color: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&out_alpha, total_pixels * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate out_alpha: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }    
    err = cudaMalloc(&grad_means2D, num_primitives * 2 * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_means2D: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_radii, num_primitives * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_radii: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_rotations, num_primitives * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_rotations: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_opacities, num_primitives * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_opacities: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_colors, num_primitives * 3 * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_colors: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    
    // Initialize arrays to zero
    cudaMemset(pixel_alphas, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_colors_r, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_colors_g, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_colors_b, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_T_values, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_prim_counts, 0, total_pixels * sizeof(int));
    cudaMemset(sigma_inv, 0, num_primitives * 4 * sizeof(float));
    cudaMemset(grad_sigma, 0, num_primitives * 4 * sizeof(float));
    
    // Initialize tile arrays to zero
    cudaMemset(d_tile_offsets, 0, tile_offsets_size * sizeof(int));
    cudaMemset(d_tile_indices, 0, tile_indices_size * sizeof(int));
    
    printf("TileRasterizer: Tile arrays initialized - offsets_size=%d, indices_size=%d\n", tile_offsets_size, tile_indices_size);
    
    // For forward pass, we need to initialize tile offsets properly
    // Each tile should have all primitives initially
    int* h_tile_offsets_init = new int[tile_offsets_size];
    int* h_tile_indices_init = new int[tile_indices_size];
    
    // Initialize tile offsets: each tile gets all primitives
    for (int i = 0; i < tile_offsets_size; i++) {
        h_tile_offsets_init[i] = i * num_primitives;
    }
    
    // Initialize tile indices: each tile gets primitives 0 to num_primitives-1
    for (int tile = 0; tile < num_tiles; tile++) {
        for (int prim = 0; prim < num_primitives; prim++) {
            h_tile_indices_init[tile * num_primitives + prim] = prim;
        }
    }
    
    // Copy to device
    cudaMemcpy(d_tile_offsets, h_tile_offsets_init, tile_offsets_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_indices, h_tile_indices_init, tile_indices_size * sizeof(int), cudaMemcpyHostToDevice);
    
    printf("TileRasterizer: Tile arrays populated - num_tiles=%d, num_primitives=%d\n", num_tiles, num_primitives);
    
    // Clean up temporary arrays
    delete[] h_tile_offsets_init;
    delete[] h_tile_indices_init;
    
    // Initialize to zero
    cudaMemset(out_color, 0, total_pixels * 3 * sizeof(float));
    cudaMemset(out_alpha, 0, total_pixels * sizeof(float));
    cudaMemset(grad_means2D, 0, num_primitives * 2 * sizeof(float));
    cudaMemset(grad_radii, 0, num_primitives * sizeof(float));
    cudaMemset(grad_rotations, 0, num_primitives * sizeof(float));
    cudaMemset(grad_opacities, 0, num_primitives * sizeof(float));
    cudaMemset(grad_colors, 0, num_primitives * 3 * sizeof(float));
    printf("Gradient tensors allocated: means2D=%d, radii=%d, rotations=%d, opacities=%d, colors=%d\n", num_primitives * 2, num_primitives, num_primitives, num_primitives, num_primitives * 3);    
    
    memory_allocated = true;
    printf("TileRasterizer memory allocated successfully\n");
}

void TileRasterizer::freeMemory() {
    if (!memory_allocated) return;
    
    printf("TileRasterizer: Freeing memory...\n");
    
    // Free memory with null checks
    if (pixel_alphas) {
        cudaFree(pixel_alphas);
        pixel_alphas = nullptr;
    }
    if (pixel_colors_r) {
        cudaFree(pixel_colors_r);
        pixel_colors_r = nullptr;
    }
    if (pixel_colors_g) {
        cudaFree(pixel_colors_g);
        pixel_colors_g = nullptr;
    }
    if (pixel_colors_b) {
        cudaFree(pixel_colors_b);
        pixel_colors_b = nullptr;
    }
    if (pixel_T_values) {
        cudaFree(pixel_T_values);
        pixel_T_values = nullptr;
    }
    if (pixel_prim_counts) {
        cudaFree(pixel_prim_counts);
        pixel_prim_counts = nullptr;
    }
    if (sigma_inv) {
        cudaFree(sigma_inv);
        sigma_inv = nullptr;
    }
    if (grad_sigma) {
        cudaFree(grad_sigma);
        grad_sigma = nullptr;
    }
    
    // Free tile arrays
    if (d_tile_offsets) {
        cudaFree(d_tile_offsets);
        d_tile_offsets = nullptr;
    }
    if (d_tile_indices) {
        cudaFree(d_tile_indices);
        d_tile_indices = nullptr;
    }

    if (out_color) {
        cudaFree(out_color);
        out_color = nullptr;
    }
    if (out_alpha) {
        cudaFree(out_alpha);
        out_alpha = nullptr;
    }
    if (grad_means2D) {
        cudaFree(grad_means2D);
        grad_means2D = nullptr;
    }
    if (grad_radii) {
        cudaFree(grad_radii);
        grad_radii = nullptr;
    }
    if (grad_rotations) {
        cudaFree(grad_rotations);
        grad_rotations = nullptr;
    }
    if (grad_opacities) {
        cudaFree(grad_opacities);
        grad_opacities = nullptr;
    }
    if (grad_colors) {
        cudaFree(grad_colors);
        grad_colors = nullptr;
    }
    
    // Reset tile array sizes
    tile_offsets_size = 0;
    tile_indices_size = 0;
    
    // Mark memory as not allocated
    memory_allocated = false;
    
    printf("TileRasterizer: Memory freed successfully\n");
}

std::tuple<torch::Tensor, torch::Tensor> TileRasterizer::forward(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel) {
    
    // Check if memory is allocated
    if (!memory_allocated) {
        throw std::runtime_error("TileRasterizer memory not allocated");
    }
    
    // Check for null pointers
    if (!pixel_alphas || !pixel_colors_r || !pixel_colors_g || !pixel_colors_b || 
        !pixel_T_values || !pixel_prim_counts || !sigma_inv || !grad_sigma) {
        throw std::runtime_error("TileRasterizer has null pointers");
    }
    
    const int num_primitives = radii.size(0);
    
    printf("TileRasterizer::forward: %d primitives, %dx%d image, tile_size=%d\n", 
           num_primitives, image_width, image_height, tile_size);
    
    // Create configuration structures
    TileConfig tile_config(image_height, image_width, tile_size, sigma, alpha_upper_bound);
    PrimitiveConfig prim_config(num_primitives, primitive_templates.size(0), 
                               primitive_templates.size(1), primitive_templates.size(2), 
                               max_prims_per_pixel);
        
    // Clear global memory arrays for this forward pass
    int total_pixels = image_height * image_width;
    int total_alpha_size = total_pixels * max_prims_per_pixel;
    cudaMemset(pixel_alphas, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_colors_r, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_colors_g, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_colors_b, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_T_values, 0, total_alpha_size * sizeof(float));
    cudaMemset(pixel_prim_counts, 0, total_pixels * sizeof(int));
    cudaMemset(sigma_inv, 0, num_primitives * 4 * sizeof(float));
    cudaMemset(grad_sigma, 0, num_primitives * 4 * sizeof(float));
    
    // Launch CUDA forward kernel
    CudaRasterizeTilesForwardKernel(
        InputTensors(
            means2D.data_ptr<float>(),
            radii.data_ptr<float>(),
            rotations.data_ptr<float>(),
            opacities.data_ptr<float>(),
            colors.data_ptr<float>(),
            primitive_templates.data_ptr<float>(),
            global_bmp_sel.data_ptr<int>()
        ),
        OutputTensors(
            out_color,
            out_alpha,
            grad_means2D,
            grad_radii,
            grad_rotations,
            grad_opacities,
            grad_colors
        ),
        GlobalBuffers(
            pixel_alphas, pixel_colors_r, pixel_colors_g, pixel_colors_b, 
            pixel_T_values, pixel_prim_counts, sigma_inv, grad_sigma,
            d_tile_offsets, d_tile_indices, tile_offsets_size, tile_indices_size
        ),
        tile_config,
        prim_config
    );
    
    printf("TileRasterizer::forward: Kernel completed, creating CUDA tensors...\n");
    
    // Create CUDA tensors directly (no CPU copy needed)
    auto out_color_tensor = torch::zeros({image_height, image_width, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto out_alpha_tensor = torch::zeros({image_height, image_width}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Copy from GPU memory to CUDA tensors
    cudaError_t err;
    err = cudaMemcpy(out_color_tensor.data_ptr<float>(), out_color, 
                     total_pixels * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy out_color to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(out_alpha_tensor.data_ptr<float>(), out_alpha, 
                     total_pixels * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy out_alpha to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    printf("TileRasterizer::forward: CUDA tensors created successfully\n");
    
    return std::make_tuple(out_color_tensor, out_alpha_tensor);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TileRasterizer::backward(
    torch::Tensor grad_out_color,
    torch::Tensor grad_out_alpha,
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    torch::Tensor lr_config) {
    
    // Check if memory is allocated
    if (!memory_allocated) {
        throw std::runtime_error("TileRasterizer memory not allocated");
    }
    
    // Check for null pointers
    if (!pixel_alphas || !pixel_colors_r || !pixel_colors_g || !pixel_colors_b || 
        !pixel_T_values || !pixel_prim_counts || !sigma_inv || !grad_sigma) {
        throw std::runtime_error("TileRasterizer has null pointers");
    }
    
    const int num_primitives = radii.size(0);
    
    // Create configuration structures
    TileConfig tile_config(image_height, image_width, tile_size, sigma, alpha_upper_bound);
    PrimitiveConfig prim_config(num_primitives, primitive_templates.size(0), 
                               primitive_templates.size(1), primitive_templates.size(2), 
                               max_prims_per_pixel);
    
    // Launch CUDA backward kernel using the stored global memory
    CudaRasterizeTilesBackwardKernel(
        grad_out_color.data_ptr<float>(),
        grad_out_alpha.data_ptr<float>(),
        InputTensors(
            means2D.data_ptr<float>(),
            radii.data_ptr<float>(),
            rotations.data_ptr<float>(),
            opacities.data_ptr<float>(),
            colors.data_ptr<float>(),
            primitive_templates.data_ptr<float>(),
            global_bmp_sel.data_ptr<int>()
        ),
        OutputTensors(
            out_color,
            out_alpha,
            grad_means2D,
            grad_radii,
            grad_rotations,
            grad_opacities,
            grad_colors
        ),
        GlobalBuffers(
            pixel_alphas, pixel_colors_r, pixel_colors_g, pixel_colors_b, 
            pixel_T_values, pixel_prim_counts, sigma_inv, grad_sigma,
            d_tile_offsets, d_tile_indices, tile_offsets_size, tile_indices_size
        ),
        tile_config,
        prim_config,
        LearningRateConfig(
            lr_config[0].item<float>(), // default_lr
            lr_config[1].item<float>(), // gain_x
            lr_config[2].item<float>(), // gain_y
            lr_config[3].item<float>(), // gain_r
            lr_config[4].item<float>(), // gain_v
            lr_config[5].item<float>(), // gain_theta
            lr_config[6].item<float>()  // gain_c
        )
    );
    
    // Create tensors from CPU memory
    // Create CUDA tensors directly (no CPU copy needed)
    auto grad_means2D_tensor = torch::zeros({num_primitives, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto grad_radii_tensor = torch::zeros({num_primitives}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto grad_rotations_tensor = torch::zeros({num_primitives}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto grad_opacities_tensor = torch::zeros({num_primitives}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto grad_colors_tensor = torch::zeros({num_primitives, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Copy from GPU memory to CUDA tensors
    cudaError_t err;
    err = cudaMemcpy(grad_means2D_tensor.data_ptr<float>(), grad_means2D, 
                     num_primitives * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_means2D to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(grad_radii_tensor.data_ptr<float>(), grad_radii, 
                     num_primitives * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_radii to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(grad_rotations_tensor.data_ptr<float>(), grad_rotations, 
                     num_primitives * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_rotations to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(grad_opacities_tensor.data_ptr<float>(), grad_opacities, 
                     num_primitives * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_opacities to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(grad_colors_tensor.data_ptr<float>(), grad_colors, 
                     num_primitives * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_colors to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    printf("TileRasterizer::backward: Tensors created successfully\n");
    
    return std::make_tuple(grad_means2D_tensor, grad_radii_tensor, grad_rotations_tensor, grad_opacities_tensor, grad_colors_tensor);
}

// Wrapper functions for backward compatibility
std::tuple<torch::Tensor, torch::Tensor> CudaRasterizeTilesForward(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    int image_height,
    int image_width,
    int tile_size,
    float sigma) {
    
    const int num_primitives = radii.size(0);
    
    // Create configuration structures
    TileConfig tile_config(image_height, image_width, tile_size, sigma, 1.0f);
    PrimitiveConfig prim_config(num_primitives, primitive_templates.size(0), 
                               primitive_templates.size(1), primitive_templates.size(2), 256);
    
    // Debug: Print input tensor info
#if DEBUG_CUDA_KERNELS
    printf("C++ Wrapper: num_primitives=%d, image_size=%dx%d, tile_size=%d, total_tiles=%d\n",
           num_primitives, image_width, image_height, tile_size, tile_config.total_tiles);
    printf("C++ Wrapper: Launching CUDA kernel...\n");
#endif
    // Create or reuse global rasterizer instance
    if (!global_tile_rasterizer || 
        global_tile_rasterizer->image_height != image_height ||
        global_tile_rasterizer->image_width != image_width ||
        global_tile_rasterizer->tile_size != tile_size ||
        global_tile_rasterizer->num_primitives != num_primitives) {
        
        printf("Creating new TileRasterizer: %dx%d, tile_size=%d, num_prims=%d\n", 
               image_width, image_height, tile_size, num_primitives);
        
        global_tile_rasterizer = std::make_shared<TileRasterizer>(
            image_height, image_width, tile_size, sigma, 1.0f, 256, num_primitives);
        
        printf("TileRasterizer created successfully\n");
    }
    
    return global_tile_rasterizer->forward(means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel);
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
    torch::Tensor global_bmp_sel,  // [num_primitives] - template selection indices
    torch::Tensor lr_config_tensor,
    int image_height,
    int image_width,
    int tile_size,
    float sigma) {
    
    // Use the same global rasterizer instance that was used in forward pass
    if (!global_tile_rasterizer) {
        throw std::runtime_error("Backward pass called without forward pass");
    }
    
    return global_tile_rasterizer->backward(grad_out_color, grad_out_alpha, means2D, radii, rotations, 
                                           opacities, colors, primitive_templates, global_bmp_sel, lr_config_tensor);
}

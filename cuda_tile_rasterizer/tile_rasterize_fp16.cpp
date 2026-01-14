#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include "tile_rasterize_fp16.h"
#include "cuda_kernels/tile_forward_fp16.h"
#include "cuda_kernels/tile_backward_fp16.h"
#include "cuda_kernels/tile_common_fp16.h"

#define DEBUG_CUDA_KERNELS_FP16 0

// Global instance for Python binding
std::shared_ptr<TileRasterizerFP16> global_tile_rasterizer_fp16 = nullptr;

// TileRasterizerFP16 class implementation
TileRasterizerFP16::TileRasterizerFP16(int image_h, int image_w, int tile_sz, __half sig, __half alpha_ub, int max_prims, int num_prims) 
    : image_height(image_h), image_width(image_w), tile_size(tile_sz), sigma(sig), 
      alpha_upper_bound(alpha_ub), max_prims_per_pixel(max_prims), num_primitives(num_prims),
      memory_allocated(false), total_forward_time(0.0), total_backward_time(0.0), 
      forward_iteration_count(0), backward_iteration_count(0) {

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

TileRasterizerFP16::~TileRasterizerFP16() {
    freeMemory();
}

void TileRasterizerFP16::allocateMemory() {
    if (memory_allocated) return;
    
    int total_pixels = image_height * image_width;
    int total_alpha_size = total_pixels * max_prims_per_pixel;
    
    cudaError_t err;
    
    err = cudaMalloc(&pixel_alphas, total_alpha_size * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_alphas: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_colors_r, total_alpha_size * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_colors_r: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_colors_g, total_alpha_size * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_colors_g: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_colors_b, total_alpha_size * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_colors_b: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_T_values, total_alpha_size * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_T_values: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&pixel_prim_counts, total_pixels * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pixel_prim_counts: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&sigma_inv, num_primitives * 4 * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate sigma_inv: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_sigma, num_primitives * 4 * sizeof(__half));
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

    err = cudaMalloc(&out_color, total_pixels * 3 * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate out_color: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&out_alpha, total_pixels * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate out_alpha: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }    
    err = cudaMalloc(&grad_means2D, num_primitives * 2 * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_means2D: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_radii, num_primitives * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_radii: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_rotations, num_primitives * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_rotations: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_opacities, num_primitives * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_opacities: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&grad_colors, num_primitives * 3 * sizeof(__half));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate grad_colors: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory allocation failed");
    }
    
    // Initialize arrays to zero
    cudaMemset(pixel_alphas, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_colors_r, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_colors_g, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_colors_b, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_T_values, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_prim_counts, 0, total_pixels * sizeof(int));
    cudaMemset(sigma_inv, 0, num_primitives * 4 * sizeof(__half));
    cudaMemset(grad_sigma, 0, num_primitives * 4 * sizeof(__half));
    
    // Initialize tile arrays to zero
    cudaMemset(d_tile_offsets, 0, tile_offsets_size * sizeof(int));
    cudaMemset(d_tile_indices, 0, tile_indices_size * sizeof(int));
    
    printf("TileRasterizerFP16: Tile arrays initialized - offsets_size=%d, indices_size=%d\n", tile_offsets_size, tile_indices_size);
    
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
    
    printf("TileRasterizerFP16: Tile arrays populated - num_tiles=%d, num_primitives=%d\n", num_tiles, num_primitives);
    
    // Clean up temporary arrays
    delete[] h_tile_offsets_init;
    delete[] h_tile_indices_init;
    
    cudaMemset(out_color, 0, total_pixels * 3 * sizeof(__half));
    cudaMemset(out_alpha, 0, total_pixels * sizeof(__half));
    cudaMemset(grad_means2D, 0, num_primitives * 2 * sizeof(__half));
    cudaMemset(grad_radii, 0, num_primitives * sizeof(__half));
    cudaMemset(grad_rotations, 0, num_primitives * sizeof(__half));
    cudaMemset(grad_opacities, 0, num_primitives * sizeof(__half));
    cudaMemset(grad_colors, 0, num_primitives * 3 * sizeof(__half));
    printf("Gradient tensors allocated: means2D=%d, radii=%d, rotations=%d, opacities=%d, colors=%d\n", num_primitives * 2, num_primitives, num_primitives, num_primitives, num_primitives * 3);    
    
    memory_allocated = true;
    printf("TileRasterizerFP16 memory allocated successfully\n");
}

void TileRasterizerFP16::freeMemory() {
    if (!memory_allocated) return;
    
    printf("TileRasterizerFP16: Freeing memory...\n");
    
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
    
    printf("TileRasterizerFP16: Memory freed successfully\n");
}

std::tuple<torch::Tensor, torch::Tensor> TileRasterizerFP16::forward(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor colors_orig,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    float c_blend,
    torch::Tensor tile_primitive_mapping) {
    
    // Check if memory is allocated
    if (!memory_allocated) {
        throw std::runtime_error("TileRasterizerFP16 memory not allocated");
    }
    
    // Check for null pointers
    if (!pixel_alphas || !pixel_colors_r || !pixel_colors_g || !pixel_colors_b || 
        !pixel_T_values || !pixel_prim_counts || !sigma_inv || !grad_sigma) {
        throw std::runtime_error("TileRasterizerFP16 has null pointers");
    }
    
    const int num_primitives = radii.size(0);
    
    // Apply dynamic tile-primitive mapping if provided (must have data, not just be defined)
    if (tile_primitive_mapping.defined() && tile_primitive_mapping.numel() > 0) {
        // Extract tile offsets and indices from mapping
        // tile_primitive_mapping is a 1D tensor containing concatenated data
        // Format: [tile_offsets_size, tile_indices_size, tile_offsets..., tile_indices...]
        int tile_offsets_size_from_tensor = tile_primitive_mapping[0].item<int>();
        int tile_indices_size_from_tensor = tile_primitive_mapping[1].item<int>();
        
        // Extract tile_offsets and tile_indices from the 1D tensor using narrow
        auto tile_offsets = tile_primitive_mapping.narrow(0, 2, tile_offsets_size_from_tensor);
        auto tile_indices = tile_primitive_mapping.narrow(0, 2 + tile_offsets_size_from_tensor, tile_indices_size_from_tensor);
        
        // Copy new tile mapping to device
        cudaMemcpy(d_tile_offsets, tile_offsets.data_ptr<int>(), 
                   tile_offsets_size_from_tensor * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_tile_indices, tile_indices.data_ptr<int>(), 
                   tile_indices_size_from_tensor * sizeof(int), cudaMemcpyDeviceToDevice);
        
        // Update sizes
        tile_offsets_size = tile_offsets_size_from_tensor;
        tile_indices_size = tile_indices_size_from_tensor;

#if DEBUG_CUDA_KERNELS_FP16
        
        printf("TileRasterizerFP16::forward: Applied dynamic tile mapping - offsets_size=%d, indices_size=%d\n", tile_offsets_size, tile_indices_size);
        
        // Detailed tile mapping validation
        printf("TileRasterizerFP16::forward: Tile mapping validation:\n");
        printf("  - Total tiles: %d\n", tile_offsets_size - 1);
        printf("  - Total primitive indices: %d\n", tile_indices_size);
        
        // Print first few tile offsets for validation
        printf("  - First 5 tile offsets: ");
        for (int i = 0; i < std::min(5, tile_offsets_size); i++) {
            printf("%d ", tile_offsets[i].item<int>());
        }
        printf("\n");
        
        // Print last few tile offsets for validation
        if (tile_offsets_size > 5) {
            printf("  - Last 5 tile offsets: ");
            for (int i = std::max(0, tile_offsets_size - 5); i < tile_offsets_size; i++) {
                printf("%d ", tile_offsets[i].item<int>());
            }
            printf("\n");
        }
        
        // Print first few primitive indices for validation
        printf("  - First 10 primitive indices: ");
        for (int i = 0; i < std::min(10, tile_indices_size); i++) {
            printf("%d ", tile_indices[i].item<int>());
        }
        printf("\n");
        
        // Calculate and print primitive distribution per tile
        printf("  - Primitive distribution per tile:\n");
        for (int tile_idx = 0; tile_idx < std::min(10, tile_offsets_size - 1); tile_idx++) {
            int start_idx = tile_offsets[tile_idx].item<int>();
            int end_idx = tile_offsets[tile_idx + 1].item<int>();
            int num_prims = end_idx - start_idx;
            printf("    Tile %d: %d primitives (indices %d-%d)\n", tile_idx, num_prims, start_idx, end_idx - 1);
        }
        
        for (int tile_idx = tile_offsets_size - 4; tile_idx < tile_offsets_size - 1; tile_idx++) {
            int start_idx = tile_offsets[tile_idx].item<int>();
            int end_idx = tile_offsets[tile_idx + 1].item<int>();
            int num_prims = end_idx - start_idx;
            printf("    Tile %d: %d primitives (indices %d-%d)\n", tile_idx, num_prims, start_idx, end_idx - 1);
        }
        
        printf("    ... (showing first 10 tiles with last 3 tiles only)\n");
        
        // Verify total primitive count matches
        int total_mapped_prims = tile_offsets[tile_offsets_size - 1].item<int>();
        printf("  - Total mapped primitives: %d (should match indices_size: %d)\n", total_mapped_prims, tile_indices_size);
        
        if (total_mapped_prims != tile_indices_size) {
            printf("  ⚠️ WARNING: Total mapped primitives (%d) != indices_size (%d)\n", total_mapped_prims, tile_indices_size);
        }
        
        printf("TileRasterizerFP16::forward: Tile mapping validation completed.\n");
#endif
    }
    
#if DEBUG_CUDA_KERNELS_FP16
    printf("TileRasterizerFP16::forward: %d primitives, %dx%d image, tile_size=%d\n", num_primitives, image_width, image_height, tile_size);
#endif
    
    // Create configuration structures
    TileConfigFP16 tile_config(image_height, image_width, tile_size, sigma, alpha_upper_bound);
    PrimitiveConfigFP16 prim_config(num_primitives, primitive_templates.size(0), 
                               primitive_templates.size(1), primitive_templates.size(2), 
                               max_prims_per_pixel);
        
    // Clear global memory arrays for this forward pass
    int total_pixels = image_height * image_width;
    int total_alpha_size = total_pixels * max_prims_per_pixel;
    cudaMemset(pixel_alphas, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_colors_r, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_colors_g, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_colors_b, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_T_values, 0, total_alpha_size * sizeof(__half));
    cudaMemset(pixel_prim_counts, 0, total_pixels * sizeof(int));
    cudaMemset(sigma_inv, 0, num_primitives * 4 * sizeof(__half));
    cudaMemset(grad_sigma, 0, num_primitives * 4 * sizeof(__half));

    cudaMemset(out_color, 0, total_pixels * 3 * sizeof(__half));
    cudaMemset(out_alpha, 0, total_pixels * sizeof(__half));
    
    // Start timing for kernel execution only (similar to backward pass)
    forward_start_time = std::chrono::high_resolution_clock::now();
    
    // Launch CUDA forward kernel
    CudaRasterizeTilesForwardKernelFP16(
        InputTensorsFP16(
            reinterpret_cast<const __half*>(means2D.data_ptr()),
            reinterpret_cast<const __half*>(radii.data_ptr()),
            reinterpret_cast<const __half*>(rotations.data_ptr()),
            reinterpret_cast<const __half*>(opacities.data_ptr()),
            reinterpret_cast<const __half*>(colors.data_ptr()),
            reinterpret_cast<const __half*>(colors_orig.data_ptr()),
            reinterpret_cast<const __half*>(primitive_templates.data_ptr()),
            global_bmp_sel.data_ptr<int>(),
            c_blend
        ),
        OutputTensorsFP16(
            out_color,
            out_alpha,
            grad_means2D,
            grad_radii,
            grad_rotations,
            grad_opacities,
            grad_colors
        ),
        GlobalBuffersFP16(
            pixel_alphas, pixel_colors_r, pixel_colors_g, pixel_colors_b, 
            pixel_T_values, pixel_prim_counts, sigma_inv, grad_sigma,
            d_tile_offsets, d_tile_indices, tile_offsets_size, tile_indices_size
        ),
        tile_config,
        prim_config
    );
    
    // End timing for kernel execution only
    forward_end_time = std::chrono::high_resolution_clock::now();
    auto forward_duration = std::chrono::duration_cast<std::chrono::microseconds>(forward_end_time - forward_start_time);
    double forward_time_ms = forward_duration.count() / 1000.0;
    
    total_forward_time += forward_time_ms;
    forward_iteration_count++;
    
    // Create CUDA tensors directly (no CPU copy needed)
    auto out_color_tensor = torch::zeros({image_height, image_width, 3}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto out_alpha_tensor = torch::zeros({image_height, image_width}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    // Copy from GPU memory to CUDA tensors
    cudaError_t err;
    err = cudaMemcpy(out_color_tensor.data_ptr<at::Half>(), out_color, 
                     total_pixels * 3 * sizeof(__half), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy out_color to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(out_alpha_tensor.data_ptr<at::Half>(), out_alpha, 
                     total_pixels * sizeof(__half), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy out_alpha to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
#if DEBUG_CUDA_KERNELS_FP16
    printf("TileRasterizerFP16::forward: CUDA tensors created successfully\n");
#endif
    
    return std::make_tuple(out_color_tensor, out_alpha_tensor);
}

std::tuple<torch::Tensor, torch::Tensor> TileRasterizerFP16::forward_batch(
    torch::Tensor means2D,              // (B, N, 2)
    torch::Tensor radii,                // (B, N)
    torch::Tensor rotations,            // (B, N)
    torch::Tensor opacities,            // (B, N)
    torch::Tensor colors,               // (B, N, 3)
    torch::Tensor colors_orig,          // (B, N, H, W, 3)
    torch::Tensor primitive_templates,  // (P, H, W)
    torch::Tensor global_bmp_sel,       // (N,) - shared across batch
    float c_blend,
    torch::Tensor tile_primitive_mapping) {
    
    // Check if memory is allocated
    if (!memory_allocated) {
        throw std::runtime_error("TileRasterizerFP16 memory not allocated");
    }
    
    const int batch_size = means2D.size(0);
    const int num_prims = means2D.size(1);
    const int total_pixels = image_height * image_width;
    
#if DEBUG_CUDA_KERNELS_FP16
    printf("TileRasterizerFP16::forward_batch: batch_size=%d, num_prims=%d, image=%dx%d\n", 
           batch_size, num_prims, image_width, image_height);
#endif
    
    // Allocate output tensors for batch
    auto out_color_batch = torch::zeros({batch_size, image_height, image_width, 3}, 
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto out_alpha_batch = torch::zeros({batch_size, image_height, image_width}, 
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    // Process each candidate in the batch
    for (int b = 0; b < batch_size; b++) {
        // Extract single candidate tensors from batch
        auto means2D_b = means2D[b];          // (N, 2)
        auto radii_b = radii[b];              // (N,)
        auto rotations_b = rotations[b];      // (N,)
        auto opacities_b = opacities[b];      // (N,)
        auto colors_b = colors[b];            // (N, 3)
        auto colors_orig_b = colors_orig[b];  // (N, H, W, 3)
        
        // Call single forward for this candidate
        auto [out_color_single, out_alpha_single] = forward(
            means2D_b, radii_b, rotations_b, opacities_b, colors_b,
            colors_orig_b, primitive_templates, global_bmp_sel,
            c_blend, tile_primitive_mapping
        );
        
        // Copy results to batch output
        out_color_batch[b] = out_color_single;
        out_alpha_batch[b] = out_alpha_single;
    }
    
#if DEBUG_CUDA_KERNELS_FP16
    printf("TileRasterizerFP16::forward_batch: completed batch processing\n");
#endif
    
    return std::make_tuple(out_color_batch, out_alpha_batch);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TileRasterizerFP16::backward(
    torch::Tensor grad_out_color,
    torch::Tensor grad_out_alpha,
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor colors_orig,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    float c_blend,
    torch::Tensor lr_config) {
    
    // Check if memory is allocated
    if (!memory_allocated) {
        throw std::runtime_error("TileRasterizerFP16 memory not allocated");
    }
    
    // Check for null pointers
    if (!pixel_alphas || !pixel_colors_r || !pixel_colors_g || !pixel_colors_b || 
        !pixel_T_values || !pixel_prim_counts || !sigma_inv || !grad_sigma) {
        throw std::runtime_error("TileRasterizerFP16 has null pointers");
    }
    
    const int num_primitives = radii.size(0);
    
    // Create configuration structures
    TileConfigFP16 tile_config(image_height, image_width, tile_size, sigma, alpha_upper_bound);
    PrimitiveConfigFP16 prim_config(num_primitives, primitive_templates.size(0), 
                               primitive_templates.size(1), primitive_templates.size(2), 
                               max_prims_per_pixel);

    cudaMemset(grad_means2D, 0,  num_primitives* 2 * sizeof(__half));
    cudaMemset(grad_radii, 0,  num_primitives * sizeof(__half));
    cudaMemset(grad_rotations, 0,  num_primitives * sizeof(__half));
    cudaMemset(grad_opacities, 0,  num_primitives * sizeof(__half));
    cudaMemset(grad_colors, 0,  num_primitives * 3 * sizeof(__half));
    
    // Start timing for kernel execution only
    backward_start_time = std::chrono::high_resolution_clock::now();
    
    // Launch CUDA backward kernel using the stored global memory
    CudaRasterizeTilesBackwardKernelFP16(
        reinterpret_cast<const __half*>(grad_out_color.data_ptr()),
        reinterpret_cast<const __half*>(grad_out_alpha.data_ptr()),
        InputTensorsFP16(
            reinterpret_cast<const __half*>(means2D.data_ptr()),
            reinterpret_cast<const __half*>(radii.data_ptr()),
            reinterpret_cast<const __half*>(rotations.data_ptr()),
            reinterpret_cast<const __half*>(opacities.data_ptr()),
            reinterpret_cast<const __half*>(colors.data_ptr()),
            reinterpret_cast<const __half*>(colors_orig.data_ptr()),
            reinterpret_cast<const __half*>(primitive_templates.data_ptr()),
            global_bmp_sel.data_ptr<int>(),
            c_blend
        ),
        OutputTensorsFP16(
            out_color,
            out_alpha,
            grad_means2D,
            grad_radii,
            grad_rotations,
            grad_opacities,
            grad_colors
        ),
        GlobalBuffersFP16(
            pixel_alphas, pixel_colors_r, pixel_colors_g, pixel_colors_b, 
            pixel_T_values, pixel_prim_counts, sigma_inv, grad_sigma,
            d_tile_offsets, d_tile_indices, tile_offsets_size, tile_indices_size
        ),
        tile_config,
        prim_config,
        LearningRateConfigFP16(
            __float2half(lr_config[0].item<float>()), // default_lr
            __float2half(lr_config[1].item<float>()), // gain_x
            __float2half(lr_config[2].item<float>()), // gain_y
            __float2half(lr_config[3].item<float>()), // gain_r
            __float2half(lr_config[4].item<float>()), // gain_v
            __float2half(lr_config[5].item<float>()), // gain_theta
            __float2half(lr_config[6].item<float>())  // gain_c
        )
    );
    
    // End timing for kernel execution only
    backward_end_time = std::chrono::high_resolution_clock::now();
    auto backward_duration = std::chrono::duration_cast<std::chrono::microseconds>(backward_end_time - backward_start_time);
    double backward_time_ms = backward_duration.count() / 1000.0;
    
    total_backward_time += backward_time_ms;
    backward_iteration_count++;
    
    // Create tensors from CPU memory
    // Create CUDA tensors directly (no CPU copy needed)
    auto grad_means2D_tensor = torch::zeros({num_primitives, 2}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto grad_radii_tensor = torch::zeros({num_primitives}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto grad_rotations_tensor = torch::zeros({num_primitives}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto grad_opacities_tensor = torch::zeros({num_primitives}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto grad_colors_tensor = torch::zeros({num_primitives, 3}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    // Copy from GPU memory to CUDA tensors
    cudaError_t err;
    err = cudaMemcpy(grad_means2D_tensor.data_ptr<at::Half>(), grad_means2D, 
                     num_primitives * 2 * sizeof(__half), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_means2D to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(grad_radii_tensor.data_ptr<at::Half>(), grad_radii, 
                     num_primitives * sizeof(__half), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_radii to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(grad_rotations_tensor.data_ptr<at::Half>(), grad_rotations, 
                     num_primitives * sizeof(__half), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_rotations to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(grad_opacities_tensor.data_ptr<at::Half>(), grad_opacities, 
                     num_primitives * sizeof(__half), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_opacities to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }
    
    err = cudaMemcpy(grad_colors_tensor.data_ptr<at::Half>(), grad_colors, 
                     num_primitives * 3 * sizeof(__half), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy grad_colors to CUDA tensor: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy failed");
    }

#if DEBUG_CUDA_KERNELS_FP16
    printf("TileRasterizerFP16::backward: Tensors created successfully\n");
#endif
    
    return std::make_tuple(grad_means2D_tensor, grad_radii_tensor, grad_rotations_tensor, grad_opacities_tensor, grad_colors_tensor);
}

// Timing functions implementation
void TileRasterizerFP16::printTimingStats() {
    std::cout << "=== CUDA Tile Rasterizer FP16 Timing Statistics ===" << std::endl;
    std::cout << "Forward Pass:" << std::endl;
    std::cout << "  Total time: " << total_forward_time << " ms" << std::endl;
    std::cout << "  Iterations: " << forward_iteration_count << std::endl;
    if (forward_iteration_count > 0) {
        std::cout << "  Average time per iteration: " << (total_forward_time / forward_iteration_count) << " ms" << std::endl;
    }
    
    std::cout << "Backward Pass:" << std::endl;
    std::cout << "  Total time: " << total_backward_time << " ms" << std::endl;
    std::cout << "  Iterations: " << backward_iteration_count << std::endl;
    if (backward_iteration_count > 0) {
        std::cout << "  Average time per iteration: " << (total_backward_time / backward_iteration_count) << " ms" << std::endl;
    }
    
    std::cout << "Combined:" << std::endl;
    std::cout << "  Total time: " << (total_forward_time + total_backward_time) << " ms" << std::endl;
    std::cout << "  Total iterations: " << (forward_iteration_count + backward_iteration_count) << std::endl;
    if ((forward_iteration_count + backward_iteration_count) > 0) {
        std::cout << "  Average time per iteration: " << ((total_forward_time + total_backward_time) / (forward_iteration_count + backward_iteration_count)) << " ms" << std::endl;
    }
    std::cout << "===============================================" << std::endl;
}

void TileRasterizerFP16::resetTimingStats() {
    total_forward_time = 0.0;
    total_backward_time = 0.0;
    forward_iteration_count = 0;
    backward_iteration_count = 0;
    std::cout << "FP16 Timing statistics reset." << std::endl;
}

// Python binding functions for timing
void printCudaTimingStatsFP16() {
    if (global_tile_rasterizer_fp16) {
        global_tile_rasterizer_fp16->printTimingStats();
    } else {
        std::cout << "No TileRasterizerFP16 instance available for timing stats." << std::endl;
    }
}

void resetCudaTimingStatsFP16() {
    if (global_tile_rasterizer_fp16) {
        global_tile_rasterizer_fp16->resetTimingStats();
    } else {
        std::cout << "No TileRasterizerFP16 instance available for timing stats reset." << std::endl;
    }
}

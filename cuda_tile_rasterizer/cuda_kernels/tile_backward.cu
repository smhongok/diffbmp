#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "tile_backward.h"

__global__ void tile_rasterize_backward_kernel(
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
    int total_tiles) {
    
    int tile_id = blockIdx.x;
    int pixel_id = threadIdx.x;
    
    if (tile_id >= total_tiles) return;
    
    // Calculate tile coordinates
    int tiles_x = (image_width + tile_size - 1) / tile_size;
    int tile_x = tile_id % tiles_x;
    int tile_y = tile_id / tiles_x;
    
    // Calculate tile bounds
    int tile_start_x = tile_x * tile_size;
    int tile_start_y = tile_y * tile_size;
    int tile_end_x = min(tile_start_x + tile_size, image_width);
    int tile_end_y = min(tile_start_y + tile_size, image_height);
    
    int tile_width = tile_end_x - tile_start_x;
    int tile_height = tile_end_y - tile_start_y;
    int pixels_in_tile = tile_width * tile_height;
    
    if (pixel_id >= pixels_in_tile) return;
    
    // Calculate pixel coordinates within tile
    int local_x = pixel_id % tile_width;
    int local_y = pixel_id / tile_width;
    int global_x = tile_start_x + local_x;
    int global_y = tile_start_y + local_y;
    
    // Get output gradients for this pixel
    int output_idx = global_y * image_width + global_x;
    float grad_pixel_color[3] = {
        grad_out_color[output_idx * 3],
        grad_out_color[output_idx * 3 + 1],
        grad_out_color[output_idx * 3 + 2]
    };
    float grad_pixel_alpha = grad_out_alpha[output_idx];
    
    // For now, implement simplified backward pass
    // In a full implementation, this would compute gradients w.r.t. all parameters
    // This is a placeholder that sets gradients to zero
    for (int prim_id = 0; prim_id < num_primitives; prim_id++) {
        // Zero gradients (placeholder implementation)
        if (pixel_id == 0 && tile_id == 0) {
            grad_means2D[prim_id * 2] = 0.0f;
            grad_means2D[prim_id * 2 + 1] = 0.0f;
            grad_radii[prim_id] = 0.0f;
            grad_rotations[prim_id] = 0.0f;
            grad_opacities[prim_id] = 0.0f;
            grad_colors[prim_id * 3] = 0.0f;
            grad_colors[prim_id * 3 + 1] = 0.0f;
            grad_colors[prim_id * 3 + 2] = 0.0f;
        }
    }
}

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
    int total_tiles) {
    
    // Calculate maximum pixels per tile
    int max_pixels_per_tile = tile_size * tile_size;
    
    // Launch kernel: one block per tile, one thread per pixel
    dim3 grid(total_tiles);
    dim3 block(max_pixels_per_tile);
    
    tile_rasterize_backward_kernel<<<grid, block>>>(
        grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates,
        grad_means2D, grad_radii, grad_rotations, grad_opacities, grad_colors,
        num_primitives, template_height, template_width, image_height, image_width, tile_size, sigma, total_tiles);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling
        return;
    }
    
    // Synchronize
    cudaDeviceSynchronize();
}
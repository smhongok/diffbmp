#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "tile_forward.h"

__device__ float bilinear_sample(const float* data, int height, int width, float y, float x) {
    // Clamp coordinates
    x = fmaxf(0.0f, fminf(width - 1.0f, x));
    y = fmaxf(0.0f, fminf(height - 1.0f, y));
    
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    float wx = x - x0;
    float wy = y - y0;
    
    float v00 = data[y0 * width + x0];
    float v01 = data[y0 * width + x1];
    float v10 = data[y1 * width + x0];
    float v11 = data[y1 * width + x1];
    
    return v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + 
           v10 * (1 - wx) * wy + v11 * wx * wy;
}

__global__ void tile_rasterize_forward_kernel(
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
    
    // Initialize pixel values (start with transparent, background will be added later)
    float pixel_color[3] = {0.0f, 0.0f, 0.0f};
    float pixel_alpha = 0.0f;
    
    // Process all primitives for this pixel
    for (int prim_id = 0; prim_id < num_primitives; prim_id++) {
        float mean_x = means2D[prim_id * 2];
        float mean_y = means2D[prim_id * 2 + 1];
        float radius = radii[prim_id];
        float rotation = rotations[prim_id];
        float opacity_logit = opacities[prim_id];
        
        // Check if primitive affects this tile (conservative bounding box)
        float prim_left = mean_x - radius;
        float prim_right = mean_x + radius;
        float prim_top = mean_y - radius;
        float prim_bottom = mean_y + radius;
        
        if (prim_right < tile_start_x || prim_left >= tile_end_x ||
            prim_bottom < tile_start_y || prim_top >= tile_end_y) {
            continue; // Primitive doesn't affect this tile
        }
        
        // Transform pixel to primitive local coordinates (match PyTorch)
        // PyTorch uses normalized coordinates and different coordinate system
        float dx = (float)global_x - mean_x;
        float dy = (float)global_y - mean_y;
        
        // Normalize by radius (PyTorch: pos / r_exp)
        float norm_dx = dx / radius;
        float norm_dy = dy / radius;
        
        // Apply inverse rotation (PyTorch: R_inv)
        float cos_theta = cosf(rotation);
        float sin_theta = sinf(rotation);
        float u = cos_theta * norm_dx + sin_theta * norm_dy;
        float v = -sin_theta * norm_dx + cos_theta * norm_dy;
        
        // Convert to grid_sample coordinates [-1, 1] -> [0, template_size-1]
        // PyTorch grid_sample with align_corners=True
        float sample_u = (u + 1.0f) * 0.5f * (template_width - 1);
        float sample_v = (v + 1.0f) * 0.5f * (template_height - 1);
        
        // Template selection (match PyTorch periodic assignment with flip)
        // PyTorch: idx = torch.arange(B) % p; idx = idx.flip(0)
        int template_idx;
        if (num_templates > 1) {
            // Reverse indexing to match PyTorch flip(0)
            template_idx = (num_primitives - 1 - prim_id) % num_templates;
        } else {
            template_idx = 0;
        }
        
        // Sample primitive template (use corrected coordinates)
        float mask_value = bilinear_sample(
            &primitive_templates[template_idx * template_height * template_width],
            template_height, template_width, sample_v, sample_u);
        
        // Apply Gaussian blur if sigma > 0 (PyTorch applies blur to template, not per-pixel)
        // Note: PyTorch applies gaussian_blur to the entire template beforehand
        // For now, skip per-pixel blur to match PyTorch behavior
        // if (sigma > 0.0f) {
        //     float dist_sq = dx * dx + dy * dy;
        //     float gauss = expf(-0.5f * dist_sq / (sigma * sigma));
        //     mask_value *= gauss;
        // }
        
        // Convert opacity logit to probability (match PyTorch sigmoid)
        float opacity = 1.0f / (1.0f + expf(-opacity_logit));
        // Apply alpha_upper_bound (PyTorch: alpha_upper_bound * torch.sigmoid(v))
        float alpha = alpha_upper_bound * opacity * mask_value;
        
        // Convert color logits to RGB (match PyTorch sigmoid)
        float color_r = 1.0f / (1.0f + expf(-colors[prim_id * 3]));
        float color_g = 1.0f / (1.0f + expf(-colors[prim_id * 3 + 1]));
        float color_b = 1.0f / (1.0f + expf(-colors[prim_id * 3 + 2]));
        
        // Alpha compositing (Porter-Duff over)
        float one_minus_alpha = 1.0f - alpha;
        pixel_color[0] = pixel_color[0] * one_minus_alpha + color_r * alpha;
        pixel_color[1] = pixel_color[1] * one_minus_alpha + color_g * alpha;
        pixel_color[2] = pixel_color[2] * one_minus_alpha + color_b * alpha;
        pixel_alpha = pixel_alpha * one_minus_alpha + alpha;
    }
    
    // Apply background compositing (match PyTorch: result = color + (1 - alpha) * bg)
    // PyTorch uses white background: bg = torch.ones(...)
    float bg_r = 1.0f, bg_g = 1.0f, bg_b = 1.0f;  // White background
    float final_r = pixel_color[0] + (1.0f - pixel_alpha) * bg_r;
    float final_g = pixel_color[1] + (1.0f - pixel_alpha) * bg_g;
    float final_b = pixel_color[2] + (1.0f - pixel_alpha) * bg_b;
    
    // Write output
    int output_idx = global_y * image_width + global_x;
    out_color[output_idx * 3] = final_r;
    out_color[output_idx * 3 + 1] = final_g;
    out_color[output_idx * 3 + 2] = final_b;
    out_alpha[output_idx] = pixel_alpha;
}

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
    int total_tiles) {
    
    // Calculate maximum pixels per tile
    int max_pixels_per_tile = tile_size * tile_size;
    
    // Launch kernel: one block per tile, one thread per pixel
    dim3 grid(total_tiles);
    dim3 block(max_pixels_per_tile);
    
    tile_rasterize_forward_kernel<<<grid, block>>>(
        means2D, radii, rotations, opacities, colors, primitive_templates,
        out_color, out_alpha, num_primitives, num_templates, template_height, template_width,
        image_height, image_width, tile_size, sigma, alpha_upper_bound, total_tiles);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling without printf
        return;
    }
    
    // Synchronize
    cudaDeviceSynchronize();
}
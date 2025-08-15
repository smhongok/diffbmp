#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "tile_forward.h"

#define DEBUG_CUDA_KERNELS 1

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
    
    // Shared memory for caching frequently accessed data
    __shared__ float shared_templates[1024];  // Cache template data
    
    // Debug: Print kernel launch info (only from first thread)
#if DEBUG_CUDA_KERNELS
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("CUDA Kernel: num_primitives=%d, num_templates=%d, template_size=%dx%d, image_size=%dx%d\n",
               num_primitives, num_templates, template_width, template_height, image_width, image_height);
    }
#endif
    
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
    int primitives_processed = 0;
    int primitives_in_range = 0;
    
    // Early exit if no primitives are likely to affect this pixel
    // This reduces unnecessary computation for empty regions
    bool has_primitives_nearby = false;
    for (int prim_id = 0; prim_id < num_primitives; prim_id++) {
        float mean_x = means2D[prim_id * 2];
        float mean_y = means2D[prim_id * 2 + 1];
        float radius = radii[prim_id];
        
        // Quick distance check
        float dx = (float)global_x - mean_x;
        float dy = (float)global_y - mean_y;
        float dist_sq = dx * dx + dy * dy;
        
        if (dist_sq <= radius * radius * 2.0f) {
            has_primitives_nearby = true;
            break;
        }
    }
    
    if (!has_primitives_nearby) {
        // No primitives nearby, set transparent pixels (background will be applied by Python)
        int output_idx = global_y * image_width + global_x;
        out_color[output_idx * 3] = 0.0f;     // Transparent
        out_color[output_idx * 3 + 1] = 0.0f;
        out_color[output_idx * 3 + 2] = 0.0f;
        out_alpha[output_idx] = 0.0f;
        return;
    }
    
    // Process primitives that might affect this pixel
    for (int prim_id = 0; prim_id < num_primitives; prim_id++) {
        float mean_x = means2D[prim_id * 2];
        float mean_y = means2D[prim_id * 2 + 1];
        float radius = radii[prim_id];
        float rotation = rotations[prim_id];
        float opacity_logit = opacities[prim_id];
        
#if DEBUG_CUDA_KERNELS
        // Debug: Print first few primitives info (only from first pixel)
        if (global_x == 0 && global_y == 0 && prim_id < 3) {
            printf("Primitive %d: pos=(%.2f,%.2f), r=%.2f, theta=%.2f, opacity=%.2f\n",
                   prim_id, mean_x, mean_y, radius, rotation, opacity_logit);
        }
#endif
        
        // Quick distance check first (avoid expensive operations)
        float dx = (float)global_x - mean_x;
        float dy = (float)global_y - mean_y;
        float dist_sq = dx * dx + dy * dy;
        
        if (dist_sq > radius * radius * 2.0f) {
            continue; // Pixel too far from primitive
        }
        
        primitives_in_range++;
        
        
#if DEBUG_CUDA_KERNELS
        // Debug: Print primitive processing info (only from first pixel and first few primitives)
        if (global_x == 0 && global_y == 0 && prim_id < 3) {
            printf("Primitive %d processing: dist_sq=%.2f, radius_sq=%.2f, in_range=%s\n",
                   prim_id, dist_sq, radius * radius * 2.0f, (dist_sq <= radius * radius * 2.0f) ? "YES" : "NO");
        }
#endif
        
        // Normalize by radius
        float norm_dx = dx / radius;
        float norm_dy = dy / radius;
        
        // Apply inverse rotation
        float cos_theta = cosf(rotation);
        float sin_theta = sinf(rotation);
        float u = cos_theta * norm_dx + sin_theta * norm_dy;
        float v = -sin_theta * norm_dx + cos_theta * norm_dy;
        
        // Convert to template coordinates [-1, 1] -> [0, template_size-1]
        // PyTorch grid_sample with align_corners=True
        float sample_u = (u + 1.0f) * 0.5f * (template_width - 1);
        float sample_v = (v + 1.0f) * 0.5f * (template_height - 1);
        
        // Ensure coordinates are within bounds
        sample_u = fmaxf(0.0f, fminf(template_width - 1.0f, sample_u));
        sample_v = fmaxf(0.0f, fminf(template_height - 1.0f, sample_v));
        
        // Template selection (match PyTorch periodic assignment with flip)
        int template_idx;
        if (num_templates > 1) {
            template_idx = (num_primitives - 1 - prim_id) % num_templates;
            // Ensure template_idx is within bounds
            if (template_idx >= num_templates) {
                template_idx = 0;
            }
        } else {
            template_idx = 0;
        }
        
        // Sample primitive template
        float mask_value = 0.0f;
        if (template_idx >= 0 && template_idx < num_templates) {
            mask_value = bilinear_sample(
                &primitive_templates[template_idx * template_height * template_width],
                template_height, template_width, sample_v, sample_u);
            
            // Clamp mask_value to valid range
            mask_value = fmaxf(0.0f, fminf(1.0f, mask_value));
            
#if DEBUG_CUDA_KERNELS
            // Debug: Print template sampling info (only from first pixel and first primitive)
            if (global_x == 0 && global_y == 0 && prim_id == 0) {
                printf("Template sampling: idx=%d, u=%.2f, v=%.2f, sample_u=%.2f, sample_v=%.2f, mask=%.4f\n",
                       template_idx, u, v, sample_u, sample_v, mask_value);
            }
            
            // Debug: Check if mask_value is reasonable
            if (global_x == 0 && global_y == 0 && prim_id == 0) {
                printf("Template check: template_idx=%d, num_templates=%d, mask_value=%.4f\n",
                       template_idx, num_templates, mask_value);
            }
            

#endif
        }
        
        // Convert opacity logit to probability
        float opacity = 1.0f / (1.0f + expf(-opacity_logit));
        float alpha = alpha_upper_bound * opacity * mask_value;
        
#if DEBUG_CUDA_KERNELS
        // Debug: Print more detailed info for first few primitives
        if (global_x == 0 && global_y == 0 && prim_id < 3) {
            printf("Primitive %d: template_idx=%d, mask=%.4f, opacity=%.4f, alpha=%.4f\n",
                   prim_id, template_idx, mask_value, opacity, alpha);
        }
#endif
        
        // Convert color logits to RGB
        float color_r = 1.0f / (1.0f + expf(-colors[prim_id * 3]));
        float color_g = 1.0f / (1.0f + expf(-colors[prim_id * 3 + 1]));
        float color_b = 1.0f / (1.0f + expf(-colors[prim_id * 3 + 2]));
        
        // Alpha compositing (Porter-Duff over)
        float one_minus_alpha = 1.0f - alpha;
        pixel_color[0] = pixel_color[0] * one_minus_alpha + color_r * alpha;
        pixel_color[1] = pixel_color[1] * one_minus_alpha + color_g * alpha;
        pixel_color[2] = pixel_color[2] * one_minus_alpha + color_b * alpha;
        pixel_alpha = pixel_alpha * one_minus_alpha + alpha;
        
#if DEBUG_CUDA_KERNELS
        // Debug: Print alpha compositing info (only from first pixel and first primitive)
        if (global_x == 0 && global_y == 0 && prim_id == 0) {
            printf("Alpha compositing: alpha=%.4f, color=(%.4f,%.4f,%.4f), final_color=(%.4f,%.4f,%.4f)\n",
                   alpha, color_r, color_g, color_b, pixel_color[0], pixel_color[1], pixel_color[2]);
        }
        
        // Debug: Print intermediate values (only from first pixel and first primitive)
        if (global_x == 0 && global_y == 0 && prim_id == 0) {
            printf("Intermediate: opacity_logit=%.2f, opacity=%.4f, mask_value=%.4f, alpha=%.4f\n",
                   opacity_logit, opacity, mask_value, alpha);
        }
#endif
        
        primitives_processed++;
    }
    
    // Write output (HWC format) - no background compositing here, handled by Python
    int output_idx = global_y * image_width + global_x;
    out_color[output_idx * 3] = pixel_color[0];
    out_color[output_idx * 3 + 1] = pixel_color[1];
    out_color[output_idx * 3 + 2] = pixel_color[2];
    out_alpha[output_idx] = pixel_alpha;
    
#if DEBUG_CUDA_KERNELS
    // Debug: Print final output info (only from first pixel)
    if (global_x == 0 && global_y == 0) {
        printf("Final output: pixel=(%d,%d), color=(%.4f,%.4f,%.4f), alpha=%.4f, primitives_processed=%d, primitives_in_range=%d\n",
               global_x, global_y, pixel_color[0], pixel_color[1], pixel_color[2], pixel_alpha, primitives_processed, primitives_in_range);
    }
#endif
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
    
#if DEBUG_CUDA_KERNELS
    // Debug: Print launch configuration
    printf("CudaRasterizeTilesForwardKernel: grid=%d, block=%d, max_pixels_per_tile=%d\n", total_tiles, max_pixels_per_tile, max_pixels_per_tile);
#endif
    // Optimize block size for performance
    int block_size = 1024;  // Use maximum block size for better occupancy
    
    // Launch kernel: one block per tile
    dim3 grid(total_tiles);
    dim3 block(block_size);
    
    tile_rasterize_forward_kernel<<<grid, block>>>(
        means2D, radii, rotations, opacities, colors, primitive_templates,
        out_color, out_alpha, num_primitives, num_templates, template_height, template_width,
        image_height, image_width, tile_size, sigma, alpha_upper_bound, total_tiles);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
#if DEBUG_CUDA_KERNELS
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
#endif
    
    // Synchronize
    cudaDeviceSynchronize();
    
    // Check for runtime errors
    err = cudaGetLastError();
#if DEBUG_CUDA_KERNELS
    if (err != cudaSuccess) {
        printf("CUDA kernel runtime error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("CUDA kernel execution completed successfully\n");
#endif
}
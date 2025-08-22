#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "tile_backward_fp16.h"

#define DEBUG_CUDA_KERNELS_FP16_BACKWARD 0

// FP16 version of the tile rasterization backward kernel
__global__ void tile_rasterize_backward_kernel_fp16(
    const __half* grad_out_color,
    const __half* grad_out_alpha,
    const __half* means2D,
    const __half* radii,
    const __half* rotations,
    const __half* opacities,
    const __half* colors,
    const __half* primitive_templates,
    __half* grad_means2D,
    __half* grad_radii,
    __half* grad_rotations,
    __half* grad_opacities,
    __half* grad_colors,
    int num_primitives,
    int template_height,
    int template_width,
    int image_height,
    int image_width,
    int tile_size,
    __half sigma,
    int total_tiles) {
    
    // Debug: Print kernel launch info (only from first thread)
#if DEBUG_CUDA_KERNELS_FP16_BACKWARD
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("CUDA Backward Kernel FP16: num_primitives=%d, num_templates=%d, template_size=%dx%d, image_size=%dx%d\n",
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
    
    // Get gradient for this pixel
    int pixel_idx = global_y * image_width + global_x;
    __half grad_pixel_color_r = grad_out_color[pixel_idx * 3];
    __half grad_pixel_color_g = grad_out_color[pixel_idx * 3 + 1];
    __half grad_pixel_color_b = grad_out_color[pixel_idx * 3 + 2];
    __half grad_pixel_alpha = grad_out_alpha[pixel_idx];
    
    // Process all primitives that might affect this pixel
    for (int prim_id = 0; prim_id < num_primitives; prim_id++) {
        __half mean_x = means2D[prim_id * 2];
        __half mean_y = means2D[prim_id * 2 + 1];
        __half radius = radii[prim_id];
        __half rotation = rotations[prim_id];
        __half opacity_logit = opacities[prim_id];
        
        // Quick distance check first
        __half dx = __hsub(__int2half_rn(global_x), mean_x);
        __half dy = __hsub(__int2half_rn(global_y), mean_y);
        /*
        __half dist_sq = hadd(__hmul(dx, dx), __hmul(dy, dy));
        
        if (__hgt(dist_sq, __hmul(__hmul(radius, radius), __float2half(2.0f)))) {
            continue; // Pixel too far from primitive
        }
        */
        
        // Normalize distance
        __half norm_dx = __hdiv(dx, radius);
        __half norm_dy = __hdiv(dy, radius);
        
        // Apply rotation
        __half cos_theta = hcos(rotation);
        __half sin_theta = hsin(rotation);
        __half u = __hmul(cos_theta, norm_dx);
        u = __hadd(u, __hmul(sin_theta, norm_dy));
        __half v = __hmul(__hneg(sin_theta), norm_dx);
        v = __hadd(v, __hmul(cos_theta, norm_dy));
        
        // Convert to template coordinates
        __half sample_u = __hmul(__hadd(u, __float2half(1.0f)), __float2half(0.5f));
        sample_u = __hmul(sample_u, __float2half(template_width - 1));
        __half sample_v = __hmul(__hadd(v, __float2half(1.0f)), __float2half(0.5f));
        sample_v = __hmul(sample_v, __float2half(template_height - 1));
        
        // Ensure coordinates are within bounds
        sample_u = __hmax(__float2half(0.0f), __hmin(__float2half(template_width - 1.0f), sample_u));
        sample_v = __hmax(__float2half(0.0f), __hmin(__float2half(template_height - 1.0f), sample_v));
        
        // Template selection (match forward pass)
        int template_idx = 0; // Simplified for now
        if (template_idx >= 0) {
            // Sample primitive template (simplified bilinear)
            __half mask_value = __float2half(0.0f);
            int x0 = __half2int_rd(sample_u);
            int y0 = __half2int_rd(sample_v);
            int x1 = min(x0 + 1, template_width - 1);
            int y1 = min(y0 + 1, template_height - 1);
            
            __half wx = __hsub(sample_u, sample_u);
            __half wy = __hsub(sample_v, sample_v);
            
            __half v00 = primitive_templates[template_idx * template_height * template_width + y0 * template_width + x0];
            __half v01 = primitive_templates[template_idx * template_height * template_width + y0 * template_width + x1];
            __half v10 = primitive_templates[template_idx * template_height * template_width + y1 * template_width + x0];
            __half v11 = primitive_templates[template_idx * template_height * template_width + y1 * template_width + x1];
            
            mask_value = __hmul(__hmul(v00, __hsub(__float2half(1.0f), wx)), __hsub(__float2half(1.0f), wy));
            mask_value = __hadd(mask_value, __hmul(v01, __hmul(wx, __hsub(__float2half(1.0f), wy))));
            mask_value = __hadd(mask_value, __hmul(v10, __hmul(__hsub(__float2half(1.0f), wx), wy)));
            mask_value = __hadd(mask_value, __hmul(v11, __hmul(wx, wy)));
            
            mask_value = __hmax(__float2half(0.0f), __hmin(__float2half(1.0f), mask_value));
            
            // Convert opacity logit to probability
            __half opacity = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(opacity_logit))));
            __half alpha = __hmul(__float2half(0.5f), __hmul(opacity, mask_value)); // alpha_upper_bound = 0.5
            
            // Convert color logits to RGB
            __half color_r = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(colors[prim_id * 3]))));
            __half color_g = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(colors[prim_id * 3 + 1]))));
            __half color_b = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(colors[prim_id * 3 + 2]))));
            
            // Compute gradients using atomic operations to avoid race conditions
            // Gradient w.r.t. opacity logit
            __half grad_opacity = __hmul(grad_pixel_alpha, __hmul(__hmul(__float2half(0.5f), mask_value), __hmul(opacity, __hsub(__float2half(1.0f), opacity))));
            atomicAdd(reinterpret_cast<__half*>(&grad_opacities[prim_id]), grad_opacity);
            
            // Gradient w.r.t. color logits
            __half grad_color_r = __hmul(grad_pixel_color_r, __hmul(alpha, __hmul(color_r, __hsub(__float2half(1.0f), color_r))));
            __half grad_color_g = __hmul(grad_pixel_color_g, __hmul(alpha, __hmul(color_g, __hsub(__float2half(1.0f), color_g))));
            __half grad_color_b = __hmul(grad_pixel_color_b, __hmul(alpha, __hmul(color_b, __hsub(__float2half(1.0f), color_b))));
            
            atomicAdd(&grad_colors[prim_id * 3], grad_color_r);
            atomicAdd(&grad_colors[prim_id * 3 + 1], grad_color_g);
            atomicAdd(&grad_colors[prim_id * 3 + 2], grad_color_b);
            
            // Gradient w.r.t. position (simplified - only consider alpha contribution)
            __half grad_alpha_wrt_pos = __hmul(grad_pixel_alpha, __hmul(__hmul(__float2half(0.5f), opacity), mask_value));
            // This is a simplified gradient - in practice, we'd need to compute ∂mask_value/∂position
            __half grad_pos_scale = __hdiv(grad_alpha_wrt_pos, radius);
            
            atomicAdd(&grad_means2D[prim_id * 2], __hneg(__hmul(grad_pos_scale, dx)));
            atomicAdd(&grad_means2D[prim_id * 2 + 1], __hneg(__hmul(grad_pos_scale, dy)));
            
            // Gradient w.r.t. radius (simplified)
            __half grad_radius = __hmul(__hmul(grad_alpha_wrt_pos, mask_value), __hdiv(__hneg(__hadd(__hmul(dx, dx), __hmul(dy, dy))), __hmul(__hmul(radius, radius), radius)));
            atomicAdd(&grad_radii[prim_id], grad_radius);
            
            // Gradient w.r.t. rotation (simplified)
            __half grad_rotation = __hmul(__hmul(grad_alpha_wrt_pos, mask_value), __hsub(__hneg(__hmul(sin_theta, norm_dx)), __hmul(cos_theta, norm_dy)));
            atomicAdd(&grad_rotations[prim_id], grad_rotation);
        }
    }
}

// FP16 version of the main backward kernel function
void CudaRasterizeTilesBackwardKernelFP16(
    const __half* grad_out_color,
    const __half* grad_out_alpha,
    const __half* means2D,
    const __half* radii,
    const __half* rotations,
    const __half* opacities,
    const __half* colors,
    const __half* primitive_templates,
    __half* grad_means2D,
    __half* grad_radii,
    __half* grad_rotations,
    __half* grad_opacities,
    __half* grad_colors,
    int num_primitives,
    int template_height,
    int template_width,
    int image_height,
    int image_width,
    int tile_size,
    __half sigma,
    int total_tiles) {
    
    // Calculate maximum pixels per tile
    int max_pixels_per_tile = tile_size * tile_size;
    
    // Launch kernel: one block per tile
    dim3 grid(total_tiles);
    dim3 block(max_pixels_per_tile);
    
    tile_rasterize_backward_kernel_fp16<<<grid, block>>>(
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
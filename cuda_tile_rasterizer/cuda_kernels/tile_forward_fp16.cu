#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "tile_forward_fp16.h"

#define DEBUG_CUDA_KERNELS_FP16 0

__device__ __half bilinear_sample(const __half* data, int height, int width, __half y, __half x) {
    // Clamp coordinates
    x = __hmax(__hmin(x, __float2half(width - 1.0f)), __float2half(0.0f));
    y = __hmax(__hmin(y, __float2half(height - 1.0f)), __float2half(0.0f));
    
    int x0 = __half2int_rd(x);
    int y0 = __half2int_rd(y);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    __half wx = __hsub(x, __int2half_rn(x0));
    __half wy = __hsub(y, __int2half_rn(y0));
    
    __half v00 = data[y0 * width + x0];
    __half v01 = data[y0 * width + x1];
    __half v10 = data[y1 * width + x0];
    __half v11 = data[y1 * width + x1];

    __half result = __hmul(__hmul(v00, __hsub(__float2half(1.0f), wx)), __hsub(__float2half(1.0f), wy));
    result = __hadd(result, __hmul(v01, __hmul(wx, __hsub(__float2half(1.0f), wy))));
    result = __hadd(result, __hmul(v10, __hmul(__hsub(__float2half(1.0f), wx), wy)));
    result = __hadd(result, __hmul(v11, __hmul(wx, wy)));
    return result;
}

__global__ void tile_rasterize_forward_kernel_fp16(
    const __half* means2D,
    const __half* radii,
    const __half* rotations,
    const __half* opacities,
    const __half* colors,
    const __half* primitive_templates,
    __half* out_color,
    __half* out_alpha,
    // Global memory arrays for transmit_over compositing
    __half* pixel_alphas,      // [image_height * image_width * max_prims_per_pixel]
    __half* pixel_colors_r,    // [image_height * image_width * max_prims_per_pixel]
    __half* pixel_colors_g,    // [image_height * image_width * max_prims_per_pixel]
    __half* pixel_colors_b,    // [image_height * image_width * max_prims_per_pixel]
    int* pixel_prim_counts,   // [image_height * image_width]
    int max_prims_per_pixel,
    int num_primitives,
    int num_templates,
    int template_height,
    int template_width,
    int image_height,
    int image_width,
    int tile_size,
    __half sigma,
    __half alpha_upper_bound,
    int total_tiles) {
    
    // Debug: Print kernel launch info (only from first thread)
#if DEBUG_CUDA_KERNELS_FP16
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("CUDA Kernel FP16: num_primitives=%d, num_templates=%d, template_size=%dx%d, image_size=%dx%d\n",
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
    __half pixel_color[3] = {__float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
    __half pixel_alpha = __float2half(0.0f);
    
    // Process all primitives for this pixel
    int primitives_processed = 0;
    int primitives_in_range = 0;
    
    // Early exit if no primitives are likely to affect this pixel
    bool has_primitives_nearby = false;
    for (int prim_id = 0; prim_id < num_primitives; prim_id++) {
        __half mean_x = means2D[prim_id * 2];
        __half mean_y = means2D[prim_id * 2 + 1];
        __half radius = radii[prim_id];
        
        // Quick distance check
        __half dx = __hsub(__float2half((float)global_x), mean_x);
        __half dy = __hsub(__float2half((float)global_y), mean_y);
        __half dist_sq = __hadd(__hmul(dx, dx), __hmul(dy, dy));
        
        if (__hle(dist_sq, __hmul(__hmul(radius, radius), __float2half(2.0f)))) {
            has_primitives_nearby = true;
            break;
        }
    }
    
    if (!has_primitives_nearby) {
        // No primitives nearby, set transparent pixels
        int output_idx = global_y * image_width + global_x;
        out_color[output_idx * 3] = __float2half(0.0f);
        out_color[output_idx * 3 + 1] = __float2half(0.0f);
        out_color[output_idx * 3 + 2] = __float2half(0.0f);
        out_alpha[output_idx] = __float2half(0.0f);
        return;
    }
    
    // Process primitives that might affect this pixel
    for (int prim_id = 0; prim_id < num_primitives; prim_id++) {
        __half mean_x = means2D[prim_id * 2];
        __half mean_y = means2D[prim_id * 2 + 1];
        __half radius = radii[prim_id];
        __half rotation = rotations[prim_id];
        __half opacity_logit = opacities[prim_id];
        
#if DEBUG_CUDA_KERNELS_FP16
        // Debug: Print first few primitives info (only from first pixel)
        if (global_x == 0 && global_y == 0 && prim_id < 3) {
            printf("Primitive %d FP16: pos=(%.2f,%.2f), r=%.2f, theta=%.2f, opacity=%.2f\n",
                   prim_id, __half2float(mean_x), __half2float(mean_y), __half2float(radius), __half2float(rotation), __half2float(opacity_logit));
        }
#endif
        
        // Quick distance check first
        __half dx = __hsub(__int2half_rn((float)global_x), mean_x);
        __half dy = __hsub(__int2half_rn((float)global_y), mean_y);
        __half dist_sq = __hadd(__hmul(dx, dx), __hmul(dy, dy));
        
        if (__hgt(dist_sq, __hmul(__hmul(radius, radius), __float2half(2.0f)))) {
            continue; // Pixel too far from primitive
        }

        primitives_in_range++;
        
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
        
        // Clamp to valid range
        // u = __hmax(hmin(u, __float2half(1.0f)), __float2half(-1.0f));
        // v = __hmax(hmin(v, __float2half(1.0f)), __float2half(-1.0f));
        
        // Convert to template coordinates
        __half sample_u = __hmul(__hadd(u, __float2half(1.0f)), __float2half(0.5f));
        sample_u = __hmul(sample_u, __float2half(template_width - 1));
        __half sample_v = __hmul(__hadd(v, __float2half(1.0f)), __float2half(0.5f));
        sample_v = __hmul(sample_v, __float2half(template_height - 1));
        
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
        __half mask_value = __float2half(0.0f);
        if (template_idx >= 0 && template_idx < num_templates) {
            mask_value = bilinear_sample(
                &primitive_templates[template_idx * template_height * template_width],
                template_height, template_width, sample_v, sample_u);
            
            // Clamp mask_value to valid range
            mask_value = __hmax(__float2half(0.0f),  __hmin(__float2half(1.0f), mask_value));
            
#if DEBUG_CUDA_KERNELS_FP16
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
        
        // Convert opacity logit to probability using sigmoid
        __half opacity = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(opacity_logit))));
        __half alpha = __hmul(__hmul(alpha_upper_bound, opacity), mask_value);
        
        // Convert color logits to RGB using sigmoid
        __half color_r = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(colors[prim_id * 3]))));
        __half color_g = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(colors[prim_id * 3 + 1]))));
        __half color_b = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(colors[prim_id * 3 + 2]))));
        
        // For transmit_over compositing, store primitive data in global memory
        int pixel_global_idx = global_y * image_width + global_x;
        int pixel_data_offset = pixel_global_idx * max_prims_per_pixel;
        
        // Store current primitive's data in global memory
        int local_prim_idx = primitives_in_range;
        if (local_prim_idx < max_prims_per_pixel) {
            int global_alpha_idx = pixel_data_offset + local_prim_idx;
            
            pixel_alphas[global_alpha_idx] = alpha;
            pixel_colors_r[global_alpha_idx] = color_r;
            pixel_colors_g[global_alpha_idx] = color_g;
            pixel_colors_b[global_alpha_idx] = color_b;
        }
        
        primitives_in_range++;
        
        // Only process transmit_over when we've processed all primitives
        if (prim_id == num_primitives - 1) {
            // Store the count of primitives for this pixel
            pixel_prim_counts[pixel_global_idx] = min(primitives_in_range, max_prims_per_pixel);
            
            // Implement transmit_over compositing (VectorRenderer style)
            __half T_partial = __float2half(1.0f);  // T[0] = 1
            __half comp_color_r = __float2half(0.0f);
            __half comp_color_g = __float2half(0.0f);
            __half comp_color_b = __float2half(0.0f);
            
            // Process all primitives that affect this pixel
            int num_prims_this_pixel = min(primitives_in_range, max_prims_per_pixel);
            for (int k = 0; k < num_prims_this_pixel; k++) {
                int global_alpha_idx = pixel_data_offset + k;
                
                __half alpha_k = pixel_alphas[global_alpha_idx];
                __half color_r_k = pixel_colors_r[global_alpha_idx];
                __half color_g_k = pixel_colors_g[global_alpha_idx];
                __half color_b_k = pixel_colors_b[global_alpha_idx];
                
                // T[k] = T[k-1] * (1 - alpha[k-1])
                __half T_k = T_partial;
                
                // Composite color: C_out += T[k] * color[k] * alpha[k]
                comp_color_r = __hadd(comp_color_r, __hmul(__hmul(T_k, color_r_k), alpha_k));
                comp_color_g = __hadd(comp_color_g, __hmul(__hmul(T_k, color_g_k), alpha_k));
                comp_color_b = __hadd(comp_color_b, __hmul(__hmul(T_k, color_b_k), alpha_k));
                
                // Update T_partial for next iteration: T[k+1] = T[k] * (1 - alpha[k])
                T_partial = __hmul(T_partial, __hsub(__float2half(1.0f), alpha_k));
            }
            
            // Final composite alpha: A_out = 1 - T[N]
            __half comp_alpha = __hsub(__float2half(1.0f), T_partial);
            
            // Store final composite result
            pixel_color[0] = comp_color_r;
            pixel_color[1] = comp_color_g;
            pixel_color[2] = comp_color_b;
            pixel_alpha = comp_alpha;
        }
                
        
#if DEBUG_CUDA_KERNELS_FP16
        // Debug: Print transmit_over compositing info (only from first pixel and first primitive)
        if (global_x == 0 && global_y == 0 && prim_id == 0) {
            printf("Transmit_over: alpha=%.4f, color=(%.4f,%.4f,%.4f), final_color=(%.4f,%.4f,%.4f), final_alpha=%.4f\n",
                   __half2float(alpha), __half2float(color_r), __half2float(color_g), __half2float(color_b), __half2float(pixel_color[0]), __half2float(pixel_color[1]), __half2float(pixel_color[2]), __half2float(pixel_alpha));
        }
        
        // Debug: Print intermediate values (only from first pixel and first primitive)
        if (global_x == 0 && global_y == 0 && prim_id == 0) {
            printf("Intermediate: opacity_logit=%.2f, opacity=%.4f, mask_value=%.4f, alpha=%.4f\n",
                   __half2float(opacity_logit), __half2float(opacity), __half2float(mask_value), __half2float(alpha));
        }
#endif
        
        primitives_processed++;
    }
    
    // Write output (HWC format)
    int output_idx = global_y * image_width + global_x;
    out_color[output_idx * 3] = pixel_color[0];
    out_color[output_idx * 3 + 1] = pixel_color[1];
    out_color[output_idx * 3 + 2] = pixel_color[2];
    out_alpha[output_idx] = pixel_alpha;
    
#if DEBUG_CUDA_KERNELS_FP16
    // Debug: Print final output info (only from first pixel)
    if (global_x == 0 && global_y == 0) {
        printf("Final output FP16: pixel=(%d,%d), color=(%.4f,%.4f,%.4f), alpha=%.4f, primitives_processed=%d, primitives_in_range=%d\n",
               global_x, global_y, __half2float(pixel_color[0]), __half2float(pixel_color[1]), __half2float(pixel_color[2]), __half2float(pixel_alpha), primitives_processed, primitives_in_range);
    }
#endif
}

// FP16 version of the main kernel function
void CudaRasterizeTilesForwardKernelFP16(
    const __half* means2D,
    const __half* radii,
    const __half* rotations,
    const __half* opacities,
    const __half* colors,
    const __half* primitive_templates,
    __half* out_color,
    __half* out_alpha,
    int num_primitives,
    int num_templates,
    int template_height,
    int template_width,
    int image_height,
    int image_width,
    int tile_size,
    __half sigma,
    __half alpha_upper_bound,
    int total_tiles) {
    
    // Calculate maximum pixels per tile
    int max_pixels_per_tile = tile_size * tile_size;

#if DEBUG_CUDA_KERNELS_FP16
    // Debug: Print launch configuration
    printf("CudaRasterizeTilesForwardKernelFP16: grid=%d, block=%d\n", total_tiles, tile_size * tile_size);
#endif
    
    // Launch kernel: one block per tile
    dim3 grid(total_tiles);
    dim3 block(max_pixels_per_tile);
    
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
    tile_rasterize_forward_kernel_fp16<<<grid, block>>>(
        means2D, radii, rotations, opacities, colors, primitive_templates,
        out_color, out_alpha, pixel_alphas, pixel_colors_r, pixel_colors_g, pixel_colors_b,
        pixel_prim_counts, max_prims_per_pixel, num_primitives, num_templates, template_height, template_width,
        image_height, image_width, tile_size, sigma, alpha_upper_bound, total_tiles);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
#if DEBUG_CUDA_KERNELS_FP16
    if (err != cudaSuccess) {
        printf("CUDA kernel FP16 launch error: %s\n", cudaGetErrorString(err));
        return;
    }
#endif
    
    // Synchronize
    cudaDeviceSynchronize();
    
    // Check for runtime errors
    err = cudaGetLastError();
#if DEBUG_CUDA_KERNELS_FP16
    if (err != cudaSuccess) {
        printf("CUDA kernel FP16 runtime error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("CUDA kernel FP16 execution completed successfully\n");
#endif
}

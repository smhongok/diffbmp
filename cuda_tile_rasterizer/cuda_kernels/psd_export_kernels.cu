#include "psd_export_common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <cstdint>

// =======================================================
// Stage 1: Bounding Box Computation Kernel
// =======================================================

__global__ void compute_bounding_boxes_kernel(
    const float* means2D,
    const float* radii,
    const float* rotations,
    const float* primitive_templates,
    const int* global_bmp_sel,
    int* output_bounding_boxes,
    int* cropped_sizes,
    PSDExportConfig config
) {
    int prim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (prim_id >= config.N) return;
    
    // Get primitive parameters
    float x = means2D[prim_id * 2 + 0] * config.scale_factor;
    float y = means2D[prim_id * 2 + 1] * config.scale_factor;
    float r = radii[prim_id] * config.scale_factor;
    
    // More accurate bounding box calculation considering template size
    // Use template size for more accurate bounds
    float template_radius = r * 1.5f; // Add some padding for rotation
    int min_x = max(0, (int)floorf(x - template_radius));
    int min_y = max(0, (int)floorf(y - template_radius));
    int max_x = min(config.W, (int)ceilf(x + template_radius));
    int max_y = min(config.H, (int)ceilf(y + template_radius));
    
    // Check if valid region exists
    if (max_x <= min_x || max_y <= min_y) {
        // Empty bounding box - set minimal 1x1 region
        output_bounding_boxes[prim_id * 4 + 0] = 0;
        output_bounding_boxes[prim_id * 4 + 1] = 0;
        output_bounding_boxes[prim_id * 4 + 2] = 1;
        output_bounding_boxes[prim_id * 4 + 3] = 1;
        cropped_sizes[prim_id * 2 + 0] = 1;
        cropped_sizes[prim_id * 2 + 1] = 1;
    } else {
        output_bounding_boxes[prim_id * 4 + 0] = min_x;
        output_bounding_boxes[prim_id * 4 + 1] = min_y;
        output_bounding_boxes[prim_id * 4 + 2] = max_x;
        output_bounding_boxes[prim_id * 4 + 3] = max_y;
        cropped_sizes[prim_id * 2 + 0] = max_x - min_x;
        cropped_sizes[prim_id * 2 + 1] = max_y - min_y;
    }
}

// =======================================================
// Stage 2: Cropped Layer Generation Kernel
// =======================================================

__global__ void generate_cropped_layers_kernel(
    const float* means2D,
    const float* radii,
    const float* rotations,
    const float* colors,
    const float* visibility,
    const float* primitive_templates,
    const int* global_bmp_sel,
    const int* bounding_boxes,
    const int* layer_offsets,
    uint8_t* cropped_output_buffer,
    PSDExportConfig config
) {
    // Block dimension: primitives (threadIdx.x = primitive within block)
    // Grid dimension: (primitive_blocks, spatial_y, spatial_x)
    
    int block_prim_id = threadIdx.x;
    int global_prim_id = blockIdx.x * blockDim.x + block_prim_id;
    
    // Spatial coordinates from grid
    int tile_size = 16;
    int tile_y = blockIdx.y;
    int tile_x = blockIdx.z;
    
    if (global_prim_id >= config.N) return;
    
    // Get this primitive's bounding box
    int bbox_left = bounding_boxes[global_prim_id * 4 + 0];
    int bbox_top = bounding_boxes[global_prim_id * 4 + 1];
    int bbox_right = bounding_boxes[global_prim_id * 4 + 2];
    int bbox_bottom = bounding_boxes[global_prim_id * 4 + 3];
    int bbox_w = bbox_right - bbox_left;
    int bbox_h = bbox_bottom - bbox_top;
    
    // Get primitive parameters
    float x = means2D[global_prim_id * 2 + 0] * config.scale_factor;
    float y = means2D[global_prim_id * 2 + 1] * config.scale_factor;
    float r = radii[global_prim_id] * config.scale_factor;
    float theta = rotations[global_prim_id];
    
    // Get RGB colors and visibility (already sigmoid applied)
    float rgb_r = colors[global_prim_id * 3 + 0];
    float rgb_g = colors[global_prim_id * 3 + 1];
    float rgb_b = colors[global_prim_id * 3 + 2];
    float vis = visibility[global_prim_id];
    
    int buffer_offset = layer_offsets[global_prim_id];
    
    // Process tile region for this primitive
    int start_x = tile_x * tile_size;
    int start_y = tile_y * tile_size;
    int end_x = min(start_x + tile_size, config.W);
    int end_y = min(start_y + tile_size, config.H);
    
    for (int py = start_y; py < end_y; py++) {
        for (int px = start_x; px < end_x; px++) {
            // Check if pixel is within this primitive's bounding box
            if (px < bbox_left || px >= bbox_right || py < bbox_top || py >= bbox_bottom) {
                continue;
            }
            
            // Convert to local bounding box coordinates
            int local_x = px - bbox_left;
            int local_y = py - bbox_top;
            
            // Grid sampling to get template value
            float template_val = psd_grid_sample_bilinear(
                primitive_templates, global_bmp_sel[global_prim_id],
                px, py, x, y, r, theta, config
            );
            
            // Apply alpha upper bound (matching psd_exporter.py logic)
            float alpha_val = template_val * vis * config.alpha_upper_bound;
            
            // Skip pixels with very low alpha to reduce noise
            if (alpha_val < 0.001f) {
                // Store transparent pixel
                int pixel_offset = (local_y * bbox_w + local_x) * 4;
                cropped_output_buffer[buffer_offset + pixel_offset + 0] = 0;
                cropped_output_buffer[buffer_offset + pixel_offset + 1] = 0;
                cropped_output_buffer[buffer_offset + pixel_offset + 2] = 0;
                cropped_output_buffer[buffer_offset + pixel_offset + 3] = 0;
                continue;
            }
            
            // Calculate uint8 RGBA values with binary mask
            float mask = template_val > 0.0f ? 1.0f : 0.0f;
            uint8_t final_r = float_to_uint8(mask*rgb_r );
            uint8_t final_g = float_to_uint8(mask*rgb_g );
            uint8_t final_b = float_to_uint8(mask*rgb_b );
            uint8_t final_a = float_to_uint8(alpha_val );
            
            // Store in cropped buffer
            int pixel_offset = (local_y * bbox_w + local_x) * 4;
            cropped_output_buffer[buffer_offset + pixel_offset + 0] = final_r;
            cropped_output_buffer[buffer_offset + pixel_offset + 1] = final_g;
            cropped_output_buffer[buffer_offset + pixel_offset + 2] = final_b;
            cropped_output_buffer[buffer_offset + pixel_offset + 3] = final_a;
        }
    }
}

// =======================================================
// Host wrapper functions
// =======================================================

extern "C" {

void launch_compute_bounding_boxes(
    const float* means2D,
    const float* radii,
    const float* rotations,
    const float* primitive_templates,
    const int* global_bmp_sel,
    int* output_bounding_boxes,
    int* cropped_sizes,
    int N, int H, int W, int template_height, int template_width,
    float scale_factor, float alpha_upper_bound
) {
    PSDExportConfig config(N, H, W, template_height, template_width, scale_factor, alpha_upper_bound);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    compute_bounding_boxes_kernel<<<grid, block>>>(
        means2D, radii, rotations, primitive_templates, global_bmp_sel,
        output_bounding_boxes, cropped_sizes, config
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in compute_bounding_boxes_kernel: %s\n", cudaGetErrorString(err));
    }
}

void launch_generate_cropped_layers(
    const float* means2D,
    const float* radii,
    const float* rotations,
    const float* colors,
    const float* visibility,
    const float* primitive_templates,
    const int* global_bmp_sel,
    const int* bounding_boxes,
    const int* layer_offsets,
    uint8_t* cropped_output_buffer,
    int N, int H, int W, int template_height, int template_width,
    float scale_factor, float alpha_upper_bound,
    int max_bbox_w, int max_bbox_h
) {
    PSDExportConfig config(N, H, W, template_height, template_width, scale_factor, alpha_upper_bound);
    
    // Block: primitives (up to 1024 threads per block limit)
    // Grid: spatial subdivision of canvas
    int primitives_per_block = min(N, 256);  // Safe limit for block size
    int num_blocks_primitives = (N + primitives_per_block - 1) / primitives_per_block;
    
    // Spatial subdivision: 16x16 tiles for canvas
    int tile_size = 16;
    int grid_x = (W + tile_size - 1) / tile_size;
    int grid_y = (H + tile_size - 1) / tile_size;
    
    dim3 block(primitives_per_block, 1, 1);
    dim3 grid(num_blocks_primitives, grid_y, grid_x);
    
    generate_cropped_layers_kernel<<<grid, block>>>(
        means2D, radii, rotations, colors, visibility,
        primitive_templates, global_bmp_sel, bounding_boxes,
        layer_offsets, cropped_output_buffer, config
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in generate_cropped_layers_kernel: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"

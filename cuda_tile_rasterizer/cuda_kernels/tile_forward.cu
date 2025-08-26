#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "tile_forward.h"
#include "tile_common.h"

// Enable deterministic math operations
#pragma STDC FP_CONTRACT OFF

#define DEBUG_CUDA_KERNELS 0
#define DEBUG_SEQUENTIAL 0  // Set to 1 for sequential debugging, 0 for parallel execution

// Forward kernel:
// - Per pixel: iterate tile's primitive list (global ids)
// - Push into per-pixel caches (alpha, color, T placeholder) in the SAME order as forward condition
//   * Push condition == radius check ONLY (to keep k mapping with backward)
// - Compute T[k] and composite with OVER: C = Σ T[k] * (c_k * α_k),  A = 1 - Π(1-α_k)
__global__ void tile_rasterize_forward_kernel(
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const TileConfig tile_config,
    const PrimitiveConfig prim_config)
{
    const int W = tile_config.image_width;
    const int H = tile_config.image_height;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Tile mapping
    const int tilesX = (W + tile_config.tile_size - 1) / tile_config.tile_size;
    const int tx = x / tile_config.tile_size;
    const int ty = y / tile_config.tile_size;
    const int tile_id = ty * tilesX + tx;
    
#if DEBUG_CUDA_KERNELS
    // Debug tile access
    if (x == 0 && y == 0) {
        printf("CUDA Kernel: Debug - tilesX=%d, total_tiles=%d, tile_id=%d\n", tilesX, tile_config.total_tiles, tile_id);
        printf("CUDA Kernel: Debug - d_tile_offsets ptr=%p, d_tile_indices ptr=%p\n", buffers.d_tile_offsets, buffers.d_tile_indices);
        printf("CUDA Kernel: Debug - tile_offsets_size=%d, tile_indices_size=%d\n", buffers.tile_offsets_size, buffers.tile_indices_size);
    }
    
    // Bounds checking for tile access
    if (tile_id >= tile_config.total_tiles) {
        printf("CUDA Kernel: Invalid tile ID: tile_id=%d, total_tiles=%d\n", 
               tile_id, tile_config.total_tiles);
        return;
    }

    // Additional bounds checking for tile offsets
    if (tile_id >= buffers.tile_offsets_size - 1) {
        printf("CUDA Kernel: Invalid tile ID for offsets: tile_id=%d, offsets_size=%d\n", 
               tile_id, buffers.tile_offsets_size);
        return;
    }
#endif

    // Tile list
    const int start = buffers.d_tile_offsets[tile_id + 0];
    const int end   = buffers.d_tile_offsets[tile_id + 1];
    
#if DEBUG_CUDA_KERNELS
    // Debug tile offsets
    if (x == 0 && y == 0) {
        printf("CUDA Kernel: Debug - tile_id=%d, start=%d, end=%d\n", tile_id, start, end);
    }
    
    if (end < start) { // 방어
        if (x==0 && y==0) printf("Forward: bad tile offsets tile=%d start=%d end=%d\n", tile_id, start, end);
        return;
    }
    
    // Bounds checking for tile indices
    if (end > buffers.tile_indices_size) {
        printf("CUDA Kernel: Invalid tile indices end: end=%d, indices_size=%d\n", 
               end, buffers.tile_indices_size);
        return;
    }
#endif
    
    const int Ktile = end - start;
    const int* prim_ids = buffers.d_tile_indices + start;

    // Pixel index and per-pixel cache base
    const int pixel_idx = y * W + x;
    const int base = pixel_idx * prim_config.max_prims_per_pixel;
    
    // Bounds checking for base index
    const int total_pixels = W * H;
    const int max_base = total_pixels * prim_config.max_prims_per_pixel;
#if DEBUG_CUDA_KERNELS
    if (base >= max_base) {
        printf("CUDA Kernel: Invalid base index: pixel_idx=%d, base=%d, max_base=%d\n", pixel_idx, base, max_base);
        return;
    }
#endif

    // Reset count (optional if already zeroed by caller)
    int local_k = 0;

    // Phase 1: fill per-pixel caches in forward push order (radius check only)
    for (int kk = 0; kk < Ktile; ++kk) {
        const int n = prim_ids[kk];                     // global primitive id
        if (n < 0 || n >= prim_config.num_primitives)   // safety
            continue;

        // Forward push condition: template-aware bounding box check
        const float mx = inputs.means2D[2*n + 0];
        const float my = inputs.means2D[2*n + 1];
        const float r  = inputs.radii[n];
        const float dx = (float)x - mx;
        const float dy = (float)y - my;
        
        // Template-aware bounding box check instead of circular check
        const int template_idx = inputs.global_bmp_sel[n];
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            // Use template's actual dimensions for bounding box
            const float template_width = (float)prim_config.template_width;
            const float template_height = (float)prim_config.template_height;
            const float max_dim = fmaxf(template_width, template_height);
            const float scale_factor = r / (max_dim * 0.5f);  // r as scale factor
            
            // Calculate scaled template bounds
            const float half_width = template_width * 0.5f * scale_factor;
            const float half_height = template_height * 0.5f * scale_factor;
            
            // Check if pixel is outside template's bounding box
            if (fabsf(dx) > half_width || fabsf(dy) > half_height) {
                continue;  // pixel outside template bounds
            }
        } else {
            // Fallback to circular check for invalid template index
            if (dx*dx + dy*dy > r*r) {
                continue;  // not pushed, keep local_k unchanged
            }
        }
        

        if (local_k >= prim_config.max_prims_per_pixel) {
            // cache overflow guard: ignore remaining
            printf("@@@@@@@@@@@@@@@@@@@@@@@@@[ERROR]CUDA Kernel: cache overflow guard: ignore remaining@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
            break;
        }

        // Opacity: logits -> prob via sigmoid
        const float op_logit = inputs.opacities[n];
        const float s_op     = sigmoidf_safe(op_logit);           // in [0,1]
        const float alpha_max= tile_config.alpha_upper_bound;

        // Template mask via bilinear
        float mask_value = 0.f;
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const float r_inv = r > 1e-6f ? (1.f / r) : 1e6f;     // avoid div0
            const float phi = inputs.rotations[n];
            const float c = __cosf(phi), s = __sinf(phi);
            const float ndx = dx * r_inv;
            const float ndy = dy * r_inv;
            const float u =  c*ndx + s*ndy;                        // [-?,?]
            const float v = -s*ndx + c*ndy;

            const float tex_x = (u + 1.f) * 0.5f * (prim_config.template_width  - 1); // [0..W-1]
            const float tex_y = (v + 1.f) * 0.5f * (prim_config.template_height - 1); // [0..H-1]

            const float* tex = &inputs.primitive_templates[template_idx * prim_config.template_height * prim_config.template_width];
            mask_value = bilinear_sample(tex, prim_config.template_height, prim_config.template_width, tex_y, tex_x);
            // Convert to binary mask (0 or 1) after bilinear sampling
            //mask_value = mask_value > 0.5f ? 1.0f : 0.0f;
            //mask_value = fminf(fmaxf(mask_value, 0.f), 1.f);
            mask_value = fmaxf(0.f, fminf(1.f, mask_value));
        }

        if (mask_value <= 1e-6f) {
            continue;  // foreground가 아니면 모든 계산 건너뛰기
        }

        // Alpha and color for this primitive at this pixel
        const float a_k = alpha_max * s_op * mask_value;          // α_k
        const float sr  = sigmoidf_safe(inputs.colors[3*n + 0]);  // c_k (premult src)
        const float sg  = sigmoidf_safe(inputs.colors[3*n + 1]);
        const float sb  = sigmoidf_safe(inputs.colors[3*n + 2]);

        const int idx = base + local_k;
        
#if DEBUG_CUDA_KERNELS
        // Bounds checking for cache access
        if (idx >= max_base) {
            printf("CUDA Kernel: Invalid cache index: idx=%d, max_base=%d\n", idx, max_base);
            break;
        }
#endif

        // Cache: α_k, colors (non-premult), and T placeholder (set later)
        buffers.pixel_alphas[idx]   = a_k;
        buffers.pixel_colors_r[idx] = sr;
        buffers.pixel_colors_g[idx] = sg;
        buffers.pixel_colors_b[idx] = sb;
        // T[k] will be written after we know previous alphas
        // buffers.pixel_T_values[idx] = ???  (set in Phase 2)

        local_k++;
    }

    // Store per-pixel count
    buffers.pixel_prim_counts[pixel_idx] = local_k;

    float T = 1.f;
    float comp_r = 0.f, comp_g = 0.f, comp_b = 0.f;

    // Phase 2: compute T[k] and OVER composite
    for (int k = 0; k < local_k; ++k) {
        const int idx = base + k;
        
#if DEBUG_CUDA_KERNELS
        // Bounds checking for cache access
        if (idx >= max_base) {
            printf("CUDA Kernel: Invalid cache index in Phase 2: idx=%d, max_base=%d\n", idx, max_base);
            break;
        }
#endif
        
        const float a_k = buffers.pixel_alphas[idx];
        const float cr  = buffers.pixel_colors_r[idx];
        const float cg  = buffers.pixel_colors_g[idx];
        const float cb  = buffers.pixel_colors_b[idx];

        buffers.pixel_T_values[idx] = T;

        comp_r += T * cr * a_k;
        comp_g += T * cg * a_k;
        comp_b += T * cb * a_k;

        T *= fmaxf(1.f - a_k, 0.f);
    }
    const float comp_a = 1.f - T;

    outputs.out_color[pixel_idx * 3 + 0] = comp_r;
    outputs.out_color[pixel_idx * 3 + 1] = comp_g;
    outputs.out_color[pixel_idx * 3 + 2] = comp_b;
    outputs.out_alpha[pixel_idx]         = comp_a;
}

// DEBUG MODE: Sequential version of the forward kernel for debugging
__global__ void tile_rasterize_forward_kernel_debug(
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const TileConfig tile_config,
    const PrimitiveConfig prim_config)
{
    const int W = tile_config.image_width;
    const int H = tile_config.image_height;
    
    // Only thread (0,0) does the work
    if (threadIdx.x != 0 || threadIdx.y != 0 || blockIdx.x != 0 || blockIdx.y != 0) {
        return;
    }
    
    printf("CUDA Kernel: DEBUG MODE - Processing all pixels sequentially\n");
    
    // Process all pixels sequentially
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // Tile mapping
            const int tilesX = (W + tile_config.tile_size - 1) / tile_config.tile_size;
            const int tx = x / tile_config.tile_size;
            const int ty = y / tile_config.tile_size;
            const int tile_id = ty * tilesX + tx;
            
            // Debug tile access (only for first few pixels)
            if (x < 2 && y < 2) {
                printf("CUDA Kernel: Debug - pixel(%d,%d), tilesX=%d, total_tiles=%d, tile_id=%d\n", 
                       x, y, tilesX, tile_config.total_tiles, tile_id);
                printf("CUDA Kernel: Debug - d_tile_offsets ptr=%p, d_tile_indices ptr=%p\n", 
                       buffers.d_tile_offsets, buffers.d_tile_indices);
                printf("CUDA Kernel: Debug - tile_offsets_size=%d, tile_indices_size=%d\n", 
                       buffers.tile_offsets_size, buffers.tile_indices_size);
            }

            // Bounds checking for tile access
            if (tile_id >= tile_config.total_tiles) {
                printf("CUDA Kernel: Invalid tile ID: pixel(%d,%d), tile_id=%d, total_tiles=%d\n", 
                       x, y, tile_id, tile_config.total_tiles);
                continue;
            }

            // Additional bounds checking for tile offsets
            if (tile_id >= buffers.tile_offsets_size - 1) {
                printf("CUDA Kernel: Invalid tile ID for offsets: pixel(%d,%d), tile_id=%d, offsets_size=%d\n", 
                       x, y, tile_id, buffers.tile_offsets_size);
                continue;
            }

            // Tile list
            const int start = buffers.d_tile_offsets[tile_id + 0];
            const int end   = buffers.d_tile_offsets[tile_id + 1];
            
            // Debug tile offsets (only for first few pixels)
            if (x < 2 && y < 2) {
                printf("CUDA Kernel: Debug - pixel(%d,%d), tile_id=%d, start=%d, end=%d\n", x, y, tile_id, start, end);
            }
            
            if (end < start) { // 방어
                if (x < 2 && y < 2) printf("Forward: bad tile offsets pixel(%d,%d) tile=%d start=%d end=%d\n", x, y, tile_id, start, end);
                continue;
            }
            
            // Bounds checking for tile indices
            if (end > buffers.tile_indices_size) {
                printf("CUDA Kernel: Invalid tile indices end: pixel(%d,%d), end=%d, indices_size=%d\n", 
                       x, y, end, buffers.tile_indices_size);
                continue;
            }
            
            const int Ktile = end - start;
            const int* prim_ids = buffers.d_tile_indices + start;

            // Pixel index and per-pixel cache base
            const int pixel_idx = y * W + x;
            const int base = pixel_idx * prim_config.max_prims_per_pixel;
            
            // Bounds checking for base index
            const int total_pixels = W * H;
            const int max_base = total_pixels * prim_config.max_prims_per_pixel;
            if (base >= max_base) {
                printf("CUDA Kernel: Invalid base index: pixel(%d,%d), pixel_idx=%d, base=%d, max_base=%d\n", 
                       x, y, pixel_idx, base, max_base);
                continue;
            }

            // Reset count (optional if already zeroed by caller)
            int local_k = 0;

            // Phase 1: fill per-pixel caches in forward push order (radius check only)
            for (int kk = 0; kk < Ktile; ++kk) {
                const int n = prim_ids[kk];                     // global primitive id
                if (n < 0 || n >= prim_config.num_primitives)   // safety
                    continue;

                // Forward push condition: template-aware bounding box check
                const float mx = inputs.means2D[2*n + 0];
                const float my = inputs.means2D[2*n + 1];
                const float r  = inputs.radii[n];
                const float dx = (float)x - mx;
                const float dy = (float)y - my;
                
                // Template-aware bounding box check instead of circular check
                const int template_idx = inputs.global_bmp_sel[n];
                if (template_idx >= 0 && template_idx < prim_config.num_templates) {
                    // Use template's actual dimensions for bounding box
                    const float template_width = (float)prim_config.template_width;
                    const float template_height = (float)prim_config.template_height;
                    const float max_dim = fmaxf(template_width, template_height);
                    const float scale_factor = r / (max_dim * 0.5f);  // r as scale factor
                    
                    // Calculate scaled template bounds
                    const float half_width = template_width * 0.5f * scale_factor;
                    const float half_height = template_height * 0.5f * scale_factor;
                    
                    // Check if pixel is outside template's bounding box
                    if (fabsf(dx) > half_width || fabsf(dy) > half_height) {
                        continue;  // pixel outside template bounds
                    }
                } else {
                    // Fallback to circular check for invalid template index
                    if (dx*dx + dy*dy > r*r) {
                        continue;  // not pushed, keep local_k unchanged
                    }
                }

                if (local_k >= prim_config.max_prims_per_pixel) {
                    // cache overflow guard: ignore remaining
                    printf("CUDA Kernel: DEBUG MODE - cache overflow guard: ignore remaining\n");
                    break;
                }

                // Opacity: logits -> prob via sigmoid
                const float op_logit = inputs.opacities[n];
                const float s_op     = sigmoidf_safe(op_logit);           // in [0,1]
                const float alpha_max= tile_config.alpha_upper_bound;

                // Template mask via bilinear
                float mask_value = 0.f;
                if (template_idx >= 0 && template_idx < prim_config.num_templates) {
                    const float r_inv = r > 1e-6f ? (1.f / r) : 1e6f;     // avoid div0
                    const float phi = inputs.rotations[n];
                    const float c = __cosf(phi), s = __sinf(phi);
                    const float ndx = dx * r_inv;
                    const float ndy = dy * r_inv;
                    const float u =  c*ndx + s*ndy;                        // [-?,?]
                    const float v = -s*ndx + c*ndy;

                    const float tex_x = (u + 1.f) * 0.5f * (prim_config.template_width  - 1); // [0..W-1]
                    const float tex_y = (v + 1.f) * 0.5f * (prim_config.template_height - 1); // [0..H-1]

                    const float* tex = &inputs.primitive_templates[template_idx * prim_config.template_height * prim_config.template_width];
                    mask_value = bilinear_sample(tex, prim_config.template_height, prim_config.template_width, tex_y, tex_x);
                    //mask_value = fmaxf(0.f, fminf(1.f, mask_value));
                    //mask_value = fminf(fmaxf(mask_value, 0.f), 1.f);
                    mask_value = fmaxf(0.f, fminf(1.f, mask_value));
                }

                // Alpha and color for this primitive at this pixel
                const float a_k = alpha_max * s_op * mask_value;          // α_k
                const float sr  = sigmoidf_safe(inputs.colors[3*n + 0]);  // c_k (premult src)
                const float sg  = sigmoidf_safe(inputs.colors[3*n + 1]);
                const float sb  = sigmoidf_safe(inputs.colors[3*n + 2]);

                const int idx = base + local_k;
                
                // Bounds checking for cache access
                if (idx >= max_base) {
                    printf("CUDA Kernel: Invalid cache index: pixel(%d,%d), idx=%d, max_base=%d\n", x, y, idx, max_base);
                    break;
                }

                // Cache: α_k, colors (non-premult), and T placeholder (set later)
                buffers.pixel_alphas[idx]   = a_k;
                buffers.pixel_colors_r[idx] = sr;
                buffers.pixel_colors_g[idx] = sg;
                buffers.pixel_colors_b[idx] = sb;
                // T[k] will be written after we know previous alphas
                // buffers.pixel_T_values[idx] = ???  (set in Phase 2)

                local_k++;
            }

            // Store per-pixel count
            buffers.pixel_prim_counts[pixel_idx] = local_k;

            float T = 1.f;
            float comp_r = 0.f, comp_g = 0.f, comp_b = 0.f;

            // Phase 2: compute T[k] and OVER composite
            for (int k = 0; k < local_k; ++k) {
                const int idx = base + k;
                
                // Bounds checking for cache access
                if (idx >= max_base) {
                    printf("CUDA Kernel: Invalid cache index in Phase 2: pixel(%d,%d), idx=%d, max_base=%d\n", x, y, idx, max_base);
                    break;
                }
                
                const float a_k = buffers.pixel_alphas[idx];
                const float cr  = buffers.pixel_colors_r[idx];
                const float cg  = buffers.pixel_colors_g[idx];
                const float cb  = buffers.pixel_colors_b[idx];

                // cache T[k] BEFORE updating by (1-α_k)
                buffers.pixel_T_values[idx] = T;

                // OVER accumulation with premultiplied color m_k = c_k * α_k
                comp_r += T * cr * a_k;
                comp_g += T * cg * a_k;
                comp_b += T * cb * a_k;

                T *= fmaxf(1.f - a_k, 0.f);
            }
            const float comp_a = 1.f - T;

            outputs.out_color[pixel_idx * 3 + 0] = comp_r;
            outputs.out_color[pixel_idx * 3 + 1] = comp_g;
            outputs.out_color[pixel_idx * 3 + 2] = comp_b;
            outputs.out_alpha[pixel_idx]         = comp_a;
        }
    }
    
    printf("CUDA Kernel: DEBUG MODE - Sequential processing completed\n");
}

void CudaRasterizeTilesForwardKernel(
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const TileConfig tile_config,
    const PrimitiveConfig prim_config) {

#if DEBUG_CUDA_KERNELS
    printf("CUDA Forward Kernel: Launching with %dx%d image, tile_size=%d\n", tile_config.image_width, tile_config.image_height, tile_config.tile_size);
#endif

#if DEBUG_SEQUENTIAL
    // DEBUG MODE: Sequential execution for debugging
    // Use only 1 block with 1 thread to avoid race conditions
    dim3 grid(1, 1);  // Single block
    dim3 block(1, 1); // Single thread
    
    printf("CUDA Forward Kernel: DEBUG MODE - Grid size: %dx%d, Block size: %dx%d\n", grid.x, grid.y, block.x, block.y);
    
    printf("CUDA Forward Kernel: About to launch debug kernel...\n");
    
    tile_rasterize_forward_kernel_debug<<<grid, block>>>(inputs, outputs, buffers, tile_config, prim_config);
#else
    // NORMAL MODE: Parallel execution
    // Calculate grid and block dimensions
    int grid_x = (tile_config.image_width + tile_config.tile_size - 1) / tile_config.tile_size;
    int grid_y = (tile_config.image_height + tile_config.tile_size - 1) / tile_config.tile_size;
    
#if DEBUG_CUDA_KERNELS
    printf("CUDA Forward Kernel: NORMAL MODE - Grid size: %dx%d, Block size: %dx%d\n", grid_x, grid_y, tile_config.tile_size, tile_config.tile_size);
#endif

    // Launch kernel: one block per tile
    dim3 grid(grid_x, grid_y);
    dim3 block(tile_config.tile_size, tile_config.tile_size);
    
    tile_rasterize_forward_kernel<<<grid, block>>>(inputs, outputs, buffers, tile_config, prim_config);
#endif
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize to ensure kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }
    
#if DEBUG_CUDA_KERNELS
    printf("CUDA Forward Kernel: Kernel completed successfully\n");
#endif
}
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <math.h>
#include "tile_forward_fp16.h"
#include "tile_common_fp16.h"

// Enable deterministic math operations
#ifdef __CUDACC__
    #pragma STDC FP_CONTRACT OFF
#endif

#define DEBUG_CUDA_KERNELS_FP16 0
#define DEBUG_SEQUENTIAL_FP16 0  // Set to 1 for sequential debugging, 0 for parallel execution

// Forward kernel:
// - Per pixel: iterate tile's primitive list (global ids)
// - Push into per-pixel caches (alpha, color, T placeholder) in the SAME order as forward condition
//   * Push condition == radius check ONLY (to keep k mapping with backward)
// - Compute T[k] and composite with OVER: C = Σ T[k] * (c_k * α_k),  A = 1 - Π(1-α_k)
__global__ void tile_rasterize_forward_kernel_fp16(
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config)
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
    
#if DEBUG_CUDA_KERNELS_FP16
    // Debug tile access
    if (x == 0 && y == 0) {
        printf("CUDA Kernel FP16: Debug - tilesX=%d, total_tiles=%d, tile_id=%d\n", tilesX, tile_config.total_tiles, tile_id);
        printf("CUDA Kernel FP16: Debug - d_tile_offsets ptr=%p, d_tile_indices ptr=%p\n", buffers.d_tile_offsets, buffers.d_tile_indices);
        printf("CUDA Kernel FP16: Debug - tile_offsets_size=%d, tile_indices_size=%d\n", buffers.tile_offsets_size, buffers.tile_indices_size);
    }
    
    // Bounds checking for tile access
    if (tile_id >= tile_config.total_tiles) {
        printf("CUDA Kernel FP16: Invalid tile ID: tile_id=%d, total_tiles=%d\n", 
               tile_id, tile_config.total_tiles);
        return;
    }

    // Additional bounds checking for tile offsets
    if (tile_id >= buffers.tile_offsets_size - 1) {
        printf("CUDA Kernel FP16: Invalid tile ID for offsets: tile_id=%d, offsets_size=%d\n", 
               tile_id, buffers.tile_offsets_size);
        return;
    }
#endif

    // Tile list
    const int start = buffers.d_tile_offsets[tile_id + 0];
    const int end   = buffers.d_tile_offsets[tile_id + 1];
    
#if DEBUG_CUDA_KERNELS_FP16
    // Debug tile offsets
    if (x == 0 && y == 0) {
        printf("CUDA Kernel FP16: Debug - tile_id=%d, start=%d, end=%d\n", tile_id, start, end);
    }
    
    if (end < start) { // 방어
        if (x==0 && y==0) printf("Forward FP16: bad tile offsets tile=%d start=%d end=%d\n", tile_id, start, end);
        return;
    }
    
    // Bounds checking for tile indices
    if (end > buffers.tile_indices_size) {
        printf("CUDA Kernel FP16: Invalid tile indices end: end=%d, indices_size=%d\n", 
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
#if DEBUG_CUDA_KERNELS_FP16
    if (base >= max_base) {
        printf("CUDA Kernel FP16: Invalid base index: pixel_idx=%d, base=%d, max_base=%d\n", pixel_idx, base, max_base);
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
        const __half mx = inputs.means2D[2*n + 0];
        const __half my = inputs.means2D[2*n + 1];
        const __half r  = inputs.radii[n];
        const __half dx = __hsub(__float2half((float)x), mx);
        const __half dy = __hsub(__float2half((float)y), my);
        
        // Template-aware bounding box check instead of circular check
        const int template_idx = inputs.global_bmp_sel[n];
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            // Use template's actual dimensions for bounding box
            const __half template_width = __float2half((float)prim_config.template_width);
            const __half template_height = __float2half((float)prim_config.template_height);
            const __half max_dim = __float2half(fmaxf(__half2float(template_width), __half2float(template_height)));
            const __half scale_factor = __hdiv(r, __hmul(max_dim, __float2half(0.5f)));  // r as scale factor
            
            // Calculate scaled template bounds
            const __half half_width = __hmul(__hmul(template_width, __float2half(0.5f)), scale_factor);
            const __half half_height = __hmul(__hmul(template_height, __float2half(0.5f)), scale_factor);
            
            // Check if pixel is outside template's bounding box
            if (__hgt(__habs(dx), half_width) || __hgt(__habs(dy), half_height)) {
                continue;  // pixel outside template bounds
            }
        } else {
            // Fallback to circular check for invalid template index
            if (__hgt(__hadd(__hmul(dx, dx), __hmul(dy, dy)), __hmul(r, r))) {
                continue;  // not pushed, keep local_k unchanged
            }
        }
        

        if (local_k >= prim_config.max_prims_per_pixel) {
            // cache overflow guard: ignore remaining
            printf("@@@@@@@@@@@@@@@@@@@@@@@@@[ERROR]CUDA Kernel FP16: cache overflow guard: ignore remaining@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
            break;
        }

        // Opacity: logits -> prob via sigmoid
        const __half op_logit = inputs.opacities[n];
        const __half s_op     = sigmoidf_safe_fp16(op_logit);           // in [0,1]
        const __half alpha_max= tile_config.alpha_upper_bound;

        // Template mask via bilinear
        __half mask_value = __float2half(0.0f);
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const __half r_inv = __hgt(r, __float2half(1e-6f)) ? __hdiv(__float2half(1.0f), r) : __float2half(1e6f);     // avoid div0
            const __half phi = inputs.rotations[n];
            const __half c = __float2half(cosf(__half2float(phi))), s = __float2half(sinf(__half2float(phi)));
            const __half ndx = __hmul(dx, r_inv);
            const __half ndy = __hmul(dy, r_inv);
            const __half u =  __hadd(__hmul(c, ndx), __hmul(s, ndy));                        // [-?,?]
            const __half v = __hadd(__hmul(__hneg(s), ndx), __hmul(c, ndy));

            const __half tex_x = __hmul(__hmul(__hadd(u, __float2half(1.0f)), __float2half(0.5f)), __float2half(prim_config.template_width  - 1)); // [0..W-1]
            const __half tex_y = __hmul(__hmul(__hadd(v, __float2half(1.0f)), __float2half(0.5f)), __float2half(prim_config.template_height - 1)); // [0..H-1]

            const __half* tex = &inputs.primitive_templates[template_idx * prim_config.template_height * prim_config.template_width];
            mask_value = bilinear_sample_fp16(tex, prim_config.template_height, prim_config.template_width, tex_y, tex_x);
            // Convert to binary mask (0 or 1) after bilinear sampling
            //mask_value = mask_value > 0.5f ? 1.0f : 0.0f;
            //mask_value = fminf(fmaxf(mask_value, 0.f), 1.f);
            mask_value = clampf_fp16(mask_value, __float2half(0.0f), __float2half(1.0f));
        }

        if (__hle(mask_value, __float2half(1e-6f))) {
            continue;  // foreground가 아니면 모든 계산 건너뛰기
        }

        // Alpha and color for this primitive at this pixel
        const __half a_k = __hmul(__hmul(alpha_max, s_op), mask_value);          // α_k
        const __half sr  = sigmoidf_safe_fp16(inputs.colors[3*n + 0]);  // c_k (premult src)
        const __half sg  = sigmoidf_safe_fp16(inputs.colors[3*n + 1]);
        const __half sb  = sigmoidf_safe_fp16(inputs.colors[3*n + 2]);

        const int idx = base + local_k;
        
#if DEBUG_CUDA_KERNELS_FP16
        // Bounds checking for cache access
        if (idx >= max_base) {
            printf("CUDA Kernel FP16: Invalid cache index: idx=%d, max_base=%d\n", idx, max_base);
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

    __half T = __float2half(1.0f);
    __half comp_r = __float2half(0.0f), comp_g = __float2half(0.0f), comp_b = __float2half(0.0f);

    // Phase 2: compute T[k] and OVER composite
    for (int k = 0; k < local_k; ++k) {
        const int idx = base + k;
        
#if DEBUG_CUDA_KERNELS_FP16
        // Bounds checking for cache access
        if (idx >= max_base) {
            printf("CUDA Kernel FP16: Invalid cache index in Phase 2: idx=%d, max_base=%d\n", idx, max_base);
            break;
        }
#endif
        
        const __half a_k = buffers.pixel_alphas[idx];
        const __half cr  = buffers.pixel_colors_r[idx];
        const __half cg  = buffers.pixel_colors_g[idx];
        const __half cb  = buffers.pixel_colors_b[idx];

        buffers.pixel_T_values[idx] = T;

        comp_r = __hadd(comp_r, __hmul(__hmul(T, cr), a_k));
        comp_g = __hadd(comp_g, __hmul(__hmul(T, cg), a_k));
        comp_b = __hadd(comp_b, __hmul(__hmul(T, cb), a_k));

        T = __hmul(T, __hmax(__hsub(__float2half(1.0f), a_k), __float2half(0.0f)));
    }
    const __half comp_a = __hsub(__float2half(1.0f), T);

    outputs.out_color[pixel_idx * 3 + 0] = comp_r;
    outputs.out_color[pixel_idx * 3 + 1] = comp_g;
    outputs.out_color[pixel_idx * 3 + 2] = comp_b;
    outputs.out_alpha[pixel_idx]         = comp_a;
}

// DEBUG MODE: Sequential version of the forward kernel for debugging
__global__ void tile_rasterize_forward_kernel_fp16_debug(
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config)
{
    const int W = tile_config.image_width;
    const int H = tile_config.image_height;
    
    // Only thread (0,0) does the work
    if (threadIdx.x != 0 || threadIdx.y != 0 || blockIdx.x != 0 || blockIdx.y != 0) {
        return;
    }
    
    printf("CUDA Kernel FP16: DEBUG MODE - Processing all pixels sequentially\n");
    
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
                printf("CUDA Kernel FP16: Debug - pixel(%d,%d), tilesX=%d, total_tiles=%d, tile_id=%d\n", 
                       x, y, tilesX, tile_config.total_tiles, tile_id);
                printf("CUDA Kernel FP16: Debug - d_tile_offsets ptr=%p, d_tile_indices ptr=%p\n", 
                       buffers.d_tile_offsets, buffers.d_tile_indices);
                printf("CUDA Kernel FP16: Debug - tile_offsets_size=%d, tile_indices_size=%d\n", 
                       buffers.tile_offsets_size, buffers.tile_indices_size);
            }

            // Bounds checking for tile access
            if (tile_id >= tile_config.total_tiles) {
                printf("CUDA Kernel FP16: Invalid tile ID: pixel(%d,%d), tile_id=%d, total_tiles=%d\n", 
                       x, y, tile_id, tile_config.total_tiles);
                continue;
            }

            // Additional bounds checking for tile offsets
            if (tile_id >= buffers.tile_offsets_size - 1) {
                printf("CUDA Kernel FP16: Invalid tile ID for offsets: pixel(%d,%d), tile_id=%d, offsets_size=%d\n", 
                       x, y, tile_id, buffers.tile_offsets_size);
                continue;
            }

            // Tile list
            const int start = buffers.d_tile_offsets[tile_id + 0];
            const int end   = buffers.d_tile_offsets[tile_id + 1];
            
            // Debug tile offsets (only for first few pixels)
            if (x < 2 && y < 2) {
                printf("CUDA Kernel FP16: Debug - pixel(%d,%d), tile_id=%d, start=%d, end=%d\n", x, y, tile_id, start, end);
            }
            
            if (end < start) { // 방어
                if (x < 2 && y < 2) printf("Forward FP16: bad tile offsets pixel(%d,%d) tile=%d start=%d end=%d\n", x, y, tile_id, start, end);
                continue;
            }
            
            // Bounds checking for tile indices
            if (end > buffers.tile_indices_size) {
                printf("CUDA Kernel FP16: Invalid tile indices end: pixel(%d,%d), end=%d, indices_size=%d\n", 
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
                printf("CUDA Kernel FP16: Invalid base index: pixel(%d,%d), pixel_idx=%d, base=%d, max_base=%d\n", 
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
                const __half mx = inputs.means2D[2*n + 0];
                const __half my = inputs.means2D[2*n + 1];
                const __half r  = inputs.radii[n];
                const __half dx = __hsub(__float2half((float)x), mx);
                const __half dy = __hsub(__float2half((float)y), my);
                
                // Template-aware bounding box check instead of circular check
                const int template_idx = inputs.global_bmp_sel[n];
                if (template_idx >= 0 && template_idx < prim_config.num_templates) {
                    // Use template's actual dimensions for bounding box
                    const __half template_width = __float2half((float)prim_config.template_width);
                    const __half template_height = __float2half((float)prim_config.template_height);
                    const __half max_dim = __float2half(fmaxf(__half2float(template_width), __half2float(template_height)));
                    const __half scale_factor = __hdiv(r, __hmul(max_dim, __float2half(0.5f)));  // r as scale factor
                    
                    // Calculate scaled template bounds
                    const __half half_width = __hmul(__hmul(template_width, __float2half(0.5f)), scale_factor);
                    const __half half_height = __hmul(__hmul(template_height, __float2half(0.5f)), scale_factor);
                    
                    // Check if pixel is outside template's bounding box
                    if (__hgt(__habs(dx), half_width) || __hgt(__habs(dy), half_height)) {
                        continue;  // pixel outside template bounds
                    }
                } else {
                    // Fallback to circular check for invalid template index
                    if (__hgt(__hadd(__hmul(dx, dx), __hmul(dy, dy)), __hmul(r, r))) {
                        continue;  // not pushed, keep local_k unchanged
                    }
                }

                if (local_k >= prim_config.max_prims_per_pixel) {
                    // cache overflow guard: ignore remaining
                    printf("CUDA Kernel FP16: DEBUG MODE - cache overflow guard: ignore remaining\n");
                    break;
                }

                // Opacity: logits -> prob via sigmoid
                const __half op_logit = inputs.opacities[n];
                const __half s_op     = sigmoidf_safe_fp16(op_logit);           // in [0,1]
                const __half alpha_max= tile_config.alpha_upper_bound;

                // Template mask via bilinear
                __half mask_value = __float2half(0.0f);
                if (template_idx >= 0 && template_idx < prim_config.num_templates) {
                    const __half r_inv = __hgt(r, __float2half(1e-6f)) ? __hdiv(__float2half(1.0f), r) : __float2half(1e6f);     // avoid div0
                    const __half phi = inputs.rotations[n];
                    const __half c = __float2half(cosf(__half2float(phi))), s = __float2half(sinf(__half2float(phi)));
                    const __half ndx = __hmul(dx, r_inv);
                    const __half ndy = __hmul(dy, r_inv);
                    const __half u =  __hadd(__hmul(c, ndx), __hmul(s, ndy));                        // [-?,?]
                    const __half v = __hadd(__hmul(__hneg(s), ndx), __hmul(c, ndy));

                    const __half tex_x = __hmul(__hmul(__hadd(u, __float2half(1.0f)), __float2half(0.5f)), __float2half(prim_config.template_width  - 1)); // [0..W-1]
                    const __half tex_y = __hmul(__hmul(__hadd(v, __float2half(1.0f)), __float2half(0.5f)), __float2half(prim_config.template_height - 1)); // [0..H-1]

                    const __half* tex = &inputs.primitive_templates[template_idx * prim_config.template_height * prim_config.template_width];
                    mask_value = bilinear_sample_fp16(tex, prim_config.template_height, prim_config.template_width, tex_y, tex_x);
                    //mask_value = fmaxf(0.f, fminf(1.f, mask_value));
                    //mask_value = fminf(fmaxf(mask_value, 0.f), 1.f);
                    mask_value = clampf_fp16(mask_value, __float2half(0.0f), __float2half(1.0f));
                }

                // Alpha and color for this primitive at this pixel
                const __half a_k = __hmul(__hmul(alpha_max, s_op), mask_value);          // α_k
                const __half sr  = sigmoidf_safe_fp16(inputs.colors[3*n + 0]);  // c_k (premult src)
                const __half sg  = sigmoidf_safe_fp16(inputs.colors[3*n + 1]);
                const __half sb  = sigmoidf_safe_fp16(inputs.colors[3*n + 2]);

                const int idx = base + local_k;
                
                // Bounds checking for cache access
                if (idx >= max_base) {
                    printf("CUDA Kernel FP16: Invalid cache index: pixel(%d,%d), idx=%d, max_base=%d\n", x, y, idx, max_base);
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

            __half T = __float2half(1.0f);
            __half comp_r = __float2half(0.0f), comp_g = __float2half(0.0f), comp_b = __float2half(0.0f);

            // Phase 2: compute T[k] and OVER composite
            for (int k = 0; k < local_k; ++k) {
                const int idx = base + k;
                
                // Bounds checking for cache access
                if (idx >= max_base) {
                    printf("CUDA Kernel FP16: Invalid cache index in Phase 2: pixel(%d,%d), idx=%d, max_base=%d\n", x, y, idx, max_base);
                    break;
                }
                
                const __half a_k = buffers.pixel_alphas[idx];
                const __half cr  = buffers.pixel_colors_r[idx];
                const __half cg  = buffers.pixel_colors_g[idx];
                const __half cb  = buffers.pixel_colors_b[idx];

                // cache T[k] BEFORE updating by (1-α_k)
                buffers.pixel_T_values[idx] = T;

                // OVER accumulation with premultiplied color m_k = c_k * α_k
                comp_r = __hadd(comp_r, __hmul(__hmul(T, cr), a_k));
                comp_g = __hadd(comp_g, __hmul(__hmul(T, cg), a_k));
                comp_b = __hadd(comp_b, __hmul(__hmul(T, cb), a_k));

                T = __hmul(T, __hmax(__hsub(__float2half(1.0f), a_k), __float2half(0.0f)));
            }
            const __half comp_a = __hsub(__float2half(1.0f), T);

            outputs.out_color[pixel_idx * 3 + 0] = comp_r;
            outputs.out_color[pixel_idx * 3 + 1] = comp_g;
            outputs.out_color[pixel_idx * 3 + 2] = comp_b;
            outputs.out_alpha[pixel_idx]         = comp_a;
        }
    }
    
    printf("CUDA Kernel FP16: DEBUG MODE - Sequential processing completed\n");
}

void CudaRasterizeTilesForwardKernelFP16(
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config) {

#if DEBUG_CUDA_KERNELS_FP16
    printf("CUDA Forward Kernel FP16: Launching with %dx%d image, tile_size=%d\n", tile_config.image_width, tile_config.image_height, tile_config.tile_size);
#endif

#if DEBUG_SEQUENTIAL_FP16
    // DEBUG MODE: Sequential execution for debugging
    // Use only 1 block with 1 thread to avoid race conditions
    dim3 grid(1, 1);  // Single block
    dim3 block(1, 1); // Single thread
    
    printf("CUDA Forward Kernel FP16: DEBUG MODE - Grid size: %dx%d, Block size: %dx%d\n", grid.x, grid.y, block.x, block.y);
    
    printf("CUDA Forward Kernel FP16: About to launch debug kernel...\n");
    
    tile_rasterize_forward_kernel_fp16_debug<<<grid, block>>>(inputs, outputs, buffers, tile_config, prim_config);
#else
    // NORMAL MODE: Parallel execution
    // Calculate grid and block dimensions
    int grid_x = (tile_config.image_width + tile_config.tile_size - 1) / tile_config.tile_size;
    int grid_y = (tile_config.image_height + tile_config.tile_size - 1) / tile_config.tile_size;
    
#if DEBUG_CUDA_KERNELS_FP16
    printf("CUDA Forward Kernel FP16: NORMAL MODE - Grid size: %dx%d, Block size: %dx%d\n", grid_x, grid_y, tile_config.tile_size, tile_config.tile_size);
#endif

    // Launch kernel: one block per tile
    dim3 grid(grid_x, grid_y);
    dim3 block(tile_config.tile_size, tile_config.tile_size);
    
    tile_rasterize_forward_kernel_fp16<<<grid, block>>>(inputs, outputs, buffers, tile_config, prim_config);
#endif
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel FP16 launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize to ensure kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel FP16 execution error: %s\n", cudaGetErrorString(err));
        return;
    }
    
#if DEBUG_CUDA_KERNELS_FP16
    printf("CUDA Forward Kernel FP16: Kernel completed successfully\n");
#endif
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <stdio.h>
#include <math.h>
#include "tile_backward_fp16.h"
#include "tile_common_fp16.h"

#ifdef __CUDACC__
    #pragma STDC FP_CONTRACT OFF
#endif

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) (x)
#endif

#define DEBUG_CUDA_KERNELS_FP16_BACKWARD 0
#define DEBUG_SEQUENTIAL_FP16 0  // Set to 1 for sequential debugging, 0 for parallel execution
#define EPS1_FP16 __float2half(1e-6f)

// Per-pixel backward with OVER rule and per-pixel local-k mapping
template<bool HAVE_ALPHA_CACHE, bool HAVE_T_CACHE>
__device__ inline void backward_over_one_pixel_fp16(
    // pixel coord & config
    int x, int y, const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config,
    // primitive set for this tile
    const int* __restrict__ prim_ids,  // [K]
    int K,
    // upstream grads (dL/dC and dL/dA at this pixel)
    __half gCx, __half gCy, __half gCz, __half gA,
    // structured IO
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const LearningRateConfigFP16 lr_config
){
    const int W = tile_config.image_width;
    const int H = tile_config.image_height;
    const int pixel_idx = y * W + x;
    const int base = pixel_idx * prim_config.max_prims_per_pixel;
    const int num_k = min(buffers.pixel_prim_counts[pixel_idx], prim_config.max_prims_per_pixel);
    if (num_k <= 0) return;

    // Rebuild local k by using the SAME push condition as forward (radius check only).
    int k_running = 0;

    for (int kk = 0; kk < K; ++kk){
        const int n = prim_ids[kk];
        if (n < 0 || n >= prim_config.num_primitives) continue;

        // Forward push condition: radius check only
        const __half mx = inputs.means2D[2*n+0];
        const __half my = inputs.means2D[2*n+1];
        const __half r  = inputs.radii[n];
        const __half dx = __hsub(__float2half((float)x), mx);
        const __half dy = __hsub(__float2half((float)y), my);
        const int template_idx = inputs.global_bmp_sel[n];
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const __half template_width = __float2half((float)prim_config.template_width);
            const __half template_height = __float2half((float)prim_config.template_height);
            const __half max_dim = __float2half(fmaxf(__half2float(template_width), __half2float(template_height)));
            const __half scale_factor = __hdiv(r, __hmul(max_dim, __float2half(0.5f)));
            
            const __half half_width = __hmul(__hmul(template_width, __float2half(0.5f)), scale_factor);
            const __half half_height = __hmul(__hmul(template_height, __float2half(0.5f)), scale_factor);
            
            if (__hgt(__habs(dx), half_width) || __hgt(__habs(dy), half_height)) {
                continue;
            }
        } else {
            if (__hgt(__hfma(dy, dy, __hmul(dx, dx)), __hmul(r, r))) {
                continue;
            }
        }

        const int k = k_running;
        if (k >= num_k) { k_running++; continue; } // safety

        // Read caches written by forward
        const int idx = base + k;
        const __half a_k = buffers.pixel_alphas[idx];
        const __half T_k = buffers.pixel_T_values[idx];
        const __half cr  = buffers.pixel_colors_r[idx];
        const __half cg  = buffers.pixel_colors_g[idx];
        const __half cb  = buffers.pixel_colors_b[idx];

        // Get mask value early for gradient computation decisions
        __half mask_val = __float2half(0.0f);
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const __half inv_r = __hgt(r, __float2half(1e-6f)) ? __hdiv(__float2half(1.0f), r) : __float2half(1e6f);
            const __half phi = inputs.rotations[n];
            const __half c = __float2half(cosf(__half2float(phi))), s = __float2half(sinf(__half2float(phi)));
            const __half ndx = __hmul(dx, inv_r);
            const __half ndy = __hmul(dy, inv_r);
            const __half u =  __hfma(s, ndy, __hmul(c, ndx));
            const __half v = __hfma(c, ndy, __hmul(__hneg(s), ndx));
            const __half tex_x = __hmul(__hfma(u, __float2half(0.5f), __float2half(0.5f)), __float2half(prim_config.template_width  - 1));
            const __half tex_y = __hmul(__hfma(v, __float2half(0.5f), __float2half(0.5f)), __float2half(prim_config.template_height - 1));

            const __half* tex = &inputs.primitive_templates[
                template_idx * prim_config.template_height * prim_config.template_width];
            mask_val = bilinear_sample_fp16(tex, prim_config.template_height, prim_config.template_width, tex_y, tex_x);
            mask_val = clampf_fp16(mask_val, __float2half(0.0f), __float2half(1.0f));
        }
        
        if (__hle(mask_val, __float2half(1e-6f))) {
            // This primitive doesn't affect this pixel, skip
            continue;
        }
            
        // Build suffix S and back-product B:
        // S = Σ_{m>k} c_m α_m T_m,  B = Π_{m>k}(1-α_m)
        __half Sx=__float2half(0.0f), Sy=__float2half(0.0f), Sz=__float2half(0.0f), B=__float2half(1.0f);
        for (int m = k+1; m < num_k; ++m){
            const int midx = base + m;
            const __half am = buffers.pixel_alphas[midx];
            const __half Tm = buffers.pixel_T_values[midx];
            const __half crm= buffers.pixel_colors_r[midx];
            const __half cgm= buffers.pixel_colors_g[midx];
            const __half cbm= buffers.pixel_colors_b[midx];
            // Fused multiply-add in half to reduce rounding
            Sx = __hfma(__hmul(crm, am), Tm, Sx);
            Sy = __hfma(__hmul(cgm, am), Tm, Sy);
            Sz = __hfma(__hmul(cbm, am), Tm, Sz);
            B  = __hmul(B, __hmax(__hsub(__float2half(1.0f), am), __float2half(0.0f)));
        }

        // (1) dL/d(color logits): ∂L/∂c = gC * (α_k T_k); ∂c/∂z = σ(z)(1-σ(z))
        // Only compute color gradients if mask value is significant (not transparent)
        if (__hgt(mask_val, __float2half(1e-6f))) {
            const __half sr = sigmoidf_safe_fp16(inputs.colors[3*n+0]);
            const __half sg = sigmoidf_safe_fp16(inputs.colors[3*n+1]);
            const __half sb = sigmoidf_safe_fp16(inputs.colors[3*n+2]);
            // Use FMA for gC * (a_k * T_k)
            const __half aT = __hmul(a_k, T_k);
            const __half dcr = __hmul(gCx, aT);
            const __half dcg = __hmul(gCy, aT);
            const __half dcb = __hmul(gCz, aT);
            atomicAdd(&outputs.grad_colors[3*n+0], __hmul(dcr, __hmul(sr, __hsub(__float2half(1.0f), sr))));
            atomicAdd(&outputs.grad_colors[3*n+1], __hmul(dcg, __hmul(sg, __hsub(__float2half(1.0f), sg))));
            atomicAdd(&outputs.grad_colors[3*n+2], __hmul(dcb, __hmul(sb, __hsub(__float2half(1.0f), sb))));
        }

        // (2) dL/dα_k (OVER):
        // ∂C/∂α_k = T_k c_k - S/(1-α_k),  ∂A/∂α_k = T_k * B
        const __half inv1m = __hdiv(__float2half(1.0f), __hmax(__hsub(__float2half(1.0f), a_k), EPS1_FP16));
        const __half dCda_x = __hsub(__hmul(T_k, cr), __hmul(Sx, inv1m));
        const __half dCda_y = __hsub(__hmul(T_k, cg), __hmul(Sy, inv1m));
        const __half dCda_z = __hsub(__hmul(T_k, cb), __hmul(Sz, inv1m));
        // Accumulate dLdalpha with fused adds
        __half dLdalpha = __hfma(gCx, dCda_x, __float2half(0.0f));
        dLdalpha = __hfma(gCy, dCda_y, dLdalpha);
        dLdalpha = __hfma(gCz, dCda_z, dLdalpha);
        dLdalpha = __hfma(gA, __hmul(T_k, B), dLdalpha);

        // (3) α = α_max * sigmoid(v_op) * mask(u,v)
        //     opacities are LOGITS  → dα/dv = α_max * mask * σ(v)*(1-σ(v))
        const __half v_op = inputs.opacities[n];
        const __half s_op = sigmoidf_safe_fp16(v_op);
        __half dmask_dx_tex=__float2half(0.0f), dmask_dy_tex=__float2half(0.0f);

        // Template coords (forward-identical): (u,v) from rotation φ and scale r
        const __half inv_r = __hgt(r, __float2half(1e-6f)) ? __hdiv(__float2half(1.0f), r) : __float2half(1e6f);
        const __half phi = inputs.rotations[n];
        const __half c = __float2half(cosf(__half2float(phi))), s = __float2half(sinf(__half2float(phi)));
        const __half ndx = __hmul(dx, inv_r);
        const __half ndy = __hmul(dy, inv_r);
        const __half u =  __hfma(s, ndy, __hmul(c, ndx));
        const __half v = __hfma(c, ndy, __hmul(__hneg(s), ndx));
        const __half tex_x = __hmul(__hfma(u, __float2half(0.5f), __float2half(0.5f)), __float2half(prim_config.template_width  - 1));
        const __half tex_y = __hmul(__hfma(v, __float2half(0.5f), __float2half(0.5f)), __float2half(prim_config.template_height - 1));

        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const __half* tex = &inputs.primitive_templates[
                template_idx * prim_config.template_height * prim_config.template_width];
            mask_val = bilinear_value_and_grad_xy_fp16(
                tex, prim_config.template_height, prim_config.template_width,
                tex_y, tex_x, dmask_dx_tex, dmask_dy_tex);
            mask_val = clampf_fp16(mask_val, __float2half(0.0f), __float2half(1.0f));
        } else {
            mask_val = __float2half(0.0f);
            dmask_dx_tex = __float2half(0.0f);
            dmask_dy_tex = __float2half(0.0f);
        }
        

        // Only compute gradients if mask value is significant (not transparent)
        if (__hgt(mask_val, __float2half(1e-6f))) {
            const __half d_alpha_dv = __hmul(__hmul(__hmul(tile_config.alpha_upper_bound, mask_val), s_op), __hsub(__float2half(1.0f), s_op));
            atomicAdd(&outputs.grad_opacities[n], __hmul(dLdalpha, d_alpha_dv));

            // (4) mask path: dα/dmask = α_max * σ(v_op)
            const __half dalpha_dmask = __hmul(tile_config.alpha_upper_bound, s_op);

            // (5) ∂mask/∂(u,v) from texture-space grads:
            // tex_x = 0.5*(u+1)*(W-1), tex_y = 0.5*(v+1)*(H-1)
            const __half du2x = __float2half(0.5f * (prim_config.template_width  - 1));
            const __half dv2y = __float2half(0.5f * (prim_config.template_height - 1));
            const __half dmask_du = __hmul(dmask_dx_tex, du2x);
            const __half dmask_dv = __hmul(dmask_dy_tex, dv2y);

            const __half dL_dmask = __hmul(dLdalpha, dalpha_dmask);
            const __half dL_du = __hmul(dL_dmask, dmask_du);
            const __half dL_dv = __hmul(dL_dmask, dmask_dv);

            // (6) ∂(u,v)/∂(μx,μy,r,φ)   (r is scale)
            // u = ( c*dx + s*dy)/r,   v = (-s*dx + c*dy)/r
            const __half du_dmx = __hneg(__hmul(c, inv_r)),  du_dmy = __hneg(__hmul(s, inv_r));
            const __half dv_dmx =  __hmul(s, inv_r),  dv_dmy = __hneg(__hmul(c, inv_r));
            const __half du_dr  = __hneg(__hmul(u, inv_r)),  dv_dr  = __hneg(__hmul(v, inv_r));
            const __half du_dphi= __hfma(c, ndy, __hmul(__hneg(s), ndx));
            const __half dv_dphi= __hsub(__hmul(__hneg(c), ndx), __hmul(s, ndy));

            atomicAdd(&outputs.grad_means2D[2*n+0], __hfma(dL_dv, dv_dmx, __hmul(dL_du, du_dmx)));
            atomicAdd(&outputs.grad_means2D[2*n+1], __hfma(dL_dv, dv_dmy, __hmul(dL_du, du_dmy)));
            atomicAdd(&outputs.grad_radii[n],       __hfma(dL_dv, dv_dr,  __hmul(dL_du, du_dr)));
            atomicAdd(&outputs.grad_rotations[n],   __hfma(dL_dv, dv_dphi, __hmul(dL_du, du_dphi)));
        }

        // advance local k just like forward push order
        k_running++;
        if (k_running >= num_k) break;
    }

    //printf("CUDA Backward Kernel FP16: DEBUG MODE - pixel(%d,%d), K=%d, k_running=%d\n", x, y, K, k_running);
}

// Backward tile kernel (Original parallel version)
__global__ void tile_backward_over_tilelist_kernel_fp16(
    const __half* grad_out_color,
    const __half* grad_out_alpha,
    const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config,
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const LearningRateConfigFP16 lr_config
) {
    const int W = tile_config.image_width;
    const int H = tile_config.image_height;
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;

    const int tilesX = (W + tile_config.tile_size - 1) / tile_config.tile_size;
    const int tx = x / tile_config.tile_size;
    const int ty = y / tile_config.tile_size;
    const int tile_id = ty*tilesX + tx;

    const int start = buffers.d_tile_offsets[tile_id+0];
    const int end   = buffers.d_tile_offsets[tile_id+1];
    const int K     = end - start;
    const int* prim_ids = buffers.d_tile_indices + start;

    const int gidx = (y * W + x) * 3;
    const __half gCx = grad_out_color[gidx + 0];
    const __half gCy = grad_out_color[gidx + 1];
    const __half gCz = grad_out_color[gidx + 2];
    const __half gA  = grad_out_alpha ? grad_out_alpha[y * W + x] : __float2half(0.0f);

    backward_over_one_pixel_fp16<true,true>(
        x, y, tile_config, prim_config,
        prim_ids, K,
        gCx, gCy, gCz, gA,
        inputs, outputs, buffers,
        lr_config
    );
}

// DEBUG MODE: Sequential version of the backward kernel for debugging
__global__ void tile_backward_over_tilelist_kernel_fp16_debug(
    const __half* grad_out_color,
    const __half* grad_out_alpha,
    const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config,
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const LearningRateConfigFP16 lr_config
) {
    const int W = tile_config.image_width;
    const int H = tile_config.image_height;
    
    // Only thread (0,0) does the work
    if (threadIdx.x != 0 || threadIdx.y != 0 || blockIdx.x != 0 || blockIdx.y != 0) {
        return;
    }
    
    printf("CUDA Backward Kernel FP16: DEBUG MODE - Processing all pixels sequentially\n");
    
    // Process all pixels sequentially
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int tilesX = (W + tile_config.tile_size - 1) / tile_config.tile_size;
            const int tx = x / tile_config.tile_size;
            const int ty = y / tile_config.tile_size;
            const int tile_id = ty*tilesX + tx;

            const int start = buffers.d_tile_offsets[tile_id+0];
            const int end   = buffers.d_tile_offsets[tile_id+1];
            const int K     = end - start;
            const int* prim_ids = buffers.d_tile_indices + start;

            const int gidx = (y * W + x) * 3;
            const __half gCx = grad_out_color[gidx + 0];
            const __half gCy = grad_out_color[gidx + 1];
            const __half gCz = grad_out_color[gidx + 2];
            const __half gA  = grad_out_alpha ? grad_out_alpha[y * W + x] : __float2half(0.0f);

            // Debug print for first few pixels
            if (x < 2 && y < 2) {
                printf("CUDA Backward Kernel FP16: Debug - pixel(%d,%d), tile_id=%d, K=%d, grad_color=[%.4f,%.4f,%.4f], grad_alpha=%.4f\n", 
                       x, y, tile_id, K, __half2float(gCx), __half2float(gCy), __half2float(gCz), __half2float(gA));
            }

            backward_over_one_pixel_fp16<true,true>(
                x, y, tile_config, prim_config,
                prim_ids, K,
                gCx, gCy, gCz, gA,
                inputs, outputs, buffers,
                lr_config
            );
        }
    }
    printf("CUDA Backward Kernel FP16: DEBUG MODE - Sequential processing completed\n");
}

// =======================================================
// tile_backward_fp16.cu와 동일한 인터페이스를 가진 CudaRasterizeTilesBackwardKernelFP16 함수
// =======================================================
void CudaRasterizeTilesBackwardKernelFP16(
    const __half* grad_out_color,
    const __half* grad_out_alpha,
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config,
    const LearningRateConfigFP16 lr_config) {
    
#if DEBUG_CUDA_KERNELS_FP16_BACKWARD
    printf("CUDA Backward Kernel FP16: Starting with %d primitives, %dx%d image\n", prim_config.num_primitives, tile_config.image_width, tile_config.image_height);
#endif

#if DEBUG_SEQUENTIAL_FP16
    // DEBUG MODE: Sequential execution for debugging
    // Use only 1 block with 1 thread to avoid race conditions
    dim3 grid(1, 1);  // Single block
    dim3 block(1, 1); // Single thread
    
    printf("CUDA Backward Kernel FP16: DEBUG MODE - Grid size: %dx%d, Block size: %dx%d\n", grid.x, grid.y, block.x, block.y);
    
    printf("CUDA Backward Kernel FP16: About to launch debug kernel...\n");
    
    tile_backward_over_tilelist_kernel_fp16_debug<<<grid, block>>>(
        grad_out_color, grad_out_alpha,
        tile_config, prim_config,
        inputs, outputs, buffers,
        lr_config
    );
#else
    dim3 block(tile_config.tile_size, tile_config.tile_size);
    dim3 grid((tile_config.image_width  + tile_config.tile_size - 1) / tile_config.tile_size,
              (tile_config.image_height + tile_config.tile_size - 1) / tile_config.tile_size);
    
#if DEBUG_CUDA_KERNELS_FP16_BACKWARD
    printf("CUDA Backward Kernel FP16: NORMAL MODE - Grid size: %dx%d, Block size: %dx%d\n", grid.x, grid.y, block.x, block.y);
    
    printf("CUDA Backward Kernel FP16: About to launch normal kernel...\n");
#endif
    
    tile_backward_over_tilelist_kernel_fp16<<<grid, block>>>(
        grad_out_color, grad_out_alpha,
        tile_config, prim_config,
        inputs, outputs, buffers,
        lr_config
    );
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Backward Kernel FP16: Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize to ensure kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Backward Kernel FP16: Kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }
    
#if DEBUG_CUDA_KERNELS_FP16_BACKWARD
    printf("CUDA Backward Kernel FP16: Kernel completed successfully\n");
#endif
}
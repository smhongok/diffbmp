// ===============================
// tile_backward.cu  (추가/대체 스니펫)
// Gsplat-style OVER compositing backward (2D SVGSplat)
// ===============================
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <stdio.h>
#include "tile_backward.h"
#include "tile_common.h"

#pragma STDC FP_CONTRACT OFF

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) (x)
#endif

#define DEBUG_CUDA_KERNELS 0
#define DEBUG_SEQUENTIAL 0  // Set to 1 for sequential debugging, 0 for parallel execution
#define EPS1 1e-6f


// Per-pixel backward with OVER rule and per-pixel local-k mapping
template<bool HAVE_ALPHA_CACHE, bool HAVE_T_CACHE>
__device__ inline void backward_over_one_pixel(
    // pixel coord & config
    int x, int y, const TileConfig tile_config,
    const PrimitiveConfig prim_config,
    // primitive set for this tile
    const int* __restrict__ prim_ids,  // [K]
    int K,
    // upstream grads (dL/dC and dL/dA at this pixel)
    float gCx, float gCy, float gCz, float gA,
    // structured IO
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const LearningRateConfig lr_config
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
        const float mx = inputs.means2D[2*n+0];
        const float my = inputs.means2D[2*n+1];
        const float r  = inputs.radii[n];
        const float dx = (float)x - mx;
        const float dy = (float)y - my;
        const int template_idx = inputs.global_bmp_sel[n];
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const float template_width = (float)prim_config.template_width;
            const float template_height = (float)prim_config.template_height;
            const float max_dim = fmaxf(template_width, template_height);
            const float scale_factor = r / (max_dim * 0.5f);
            
            const float half_width = template_width * 0.5f * scale_factor;
            const float half_height = template_height * 0.5f * scale_factor;
            
            if (fabsf(dx) > half_width || fabsf(dy) > half_height) {
                continue;
            }
        } else {
            if (dx*dx + dy*dy > r*r) {
                continue;
            }
        }

        const int k = k_running;
        if (k >= num_k) { k_running++; continue; } // safety

        // Read caches written by forward
        const int idx = base + k;
        const float a_k = buffers.pixel_alphas[idx];
        const float T_k = buffers.pixel_T_values[idx];
        const float cr  = buffers.pixel_colors_r[idx];
        const float cg  = buffers.pixel_colors_g[idx];
        const float cb  = buffers.pixel_colors_b[idx];

        // Get mask value early for gradient computation decisions
        float mask_val = 0.f;
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const float inv_r = (r > 1e-6f) ? (1.f/r) : 1e6f;
            const float phi = inputs.rotations[n];
            const float c = __cosf(phi), s = __sinf(phi);
            const float ndx = dx * inv_r;
            const float ndy = dy * inv_r;
            const float u =  c*ndx + s*ndy;
            const float v = -s*ndx + c*ndy;
            const float tex_x = (u + 1.f) * 0.5f * (prim_config.template_width  - 1);
            const float tex_y = (v + 1.f) * 0.5f * (prim_config.template_height - 1);

            const float* tex = &inputs.primitive_templates[
                template_idx * prim_config.template_height * prim_config.template_width];
            mask_val = bilinear_sample(tex, prim_config.template_height, prim_config.template_width, tex_y, tex_x);
            mask_val = fmaxf(0.f, fminf(1.f, mask_val));
        }
        
        if (mask_val <= 1e-6f) {
            // This primitive doesn't affect this pixel, skip
            continue;
        }
            
        // Build suffix S and back-product B:
        // S = Σ_{m>k} c_m α_m T_m,  B = Π_{m>k}(1-α_m)
        float Sx=0.f, Sy=0.f, Sz=0.f, B=1.f;
        for (int m = k+1; m < num_k; ++m){
            const int midx = base + m;
            const float am = buffers.pixel_alphas[midx];
            const float Tm = buffers.pixel_T_values[midx];
            const float crm= buffers.pixel_colors_r[midx];
            const float cgm= buffers.pixel_colors_g[midx];
            const float cbm= buffers.pixel_colors_b[midx];
            Sx += crm * am * Tm;
            Sy += cgm * am * Tm;
            Sz += cbm * am * Tm;
            B  *= fmaxf(1.f - am, 0.f);
        }

        // (1) dL/d(color logits): ∂L/∂c = gC * (α_k T_k); ∂c/∂z = σ(z)(1-σ(z))
        // Only compute color gradients if mask value is significant (not transparent)
        if (mask_val > 1e-6f) {
            const float sr = sigmoidf_safe(inputs.colors[3*n+0]);
            const float sg = sigmoidf_safe(inputs.colors[3*n+1]);
            const float sb = sigmoidf_safe(inputs.colors[3*n+2]);
            const float dcr = gCx * (a_k * T_k);
            const float dcg = gCy * (a_k * T_k);
            const float dcb = gCz * (a_k * T_k);
            atomicAdd(&outputs.grad_colors[3*n+0], (dcr * sr*(1.f-sr)));
            atomicAdd(&outputs.grad_colors[3*n+1], (dcg * sg*(1.f-sg)));
            atomicAdd(&outputs.grad_colors[3*n+2], (dcb * sb*(1.f-sb)));
        }

        // (2) dL/dα_k (OVER):
        // ∂C/∂α_k = T_k c_k - S/(1-α_k),  ∂A/∂α_k = T_k * B
        const float inv1m = 1.f / fmaxf(1.f - a_k, EPS1);
        const float dCda_x = T_k * cr - Sx * inv1m;
        const float dCda_y = T_k * cg - Sy * inv1m;
        const float dCda_z = T_k * cb - Sz * inv1m;
        float dLdalpha = gCx*dCda_x + gCy*dCda_y + gCz*dCda_z;
        dLdalpha += gA * (T_k * B);

        // (3) α = α_max * sigmoid(v_op) * mask(u,v)
        //     opacities are LOGITS  → dα/dv = α_max * mask * σ(v)*(1-σ(v))
        const float v_op = inputs.opacities[n];
        const float s_op = sigmoidf_safe(v_op);
        float dmask_dx_tex=0.f, dmask_dy_tex=0.f;

        // Template coords (forward-identical): (u,v) from rotation φ and scale r
        const float inv_r = (r > 1e-6f) ? (1.f/r) : 1e6f;
        const float phi = inputs.rotations[n];
        const float c = __cosf(phi), s = __sinf(phi);
        const float ndx = dx * inv_r;
        const float ndy = dy * inv_r;
        const float u =  c*ndx + s*ndy;
        const float v = -s*ndx + c*ndy;
        const float tex_x = (u + 1.f) * 0.5f * (prim_config.template_width  - 1);
        const float tex_y = (v + 1.f) * 0.5f * (prim_config.template_height - 1);

        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const float* tex = &inputs.primitive_templates[
                template_idx * prim_config.template_height * prim_config.template_width];
            mask_val = bilinear_value_and_grad_xy(
                tex, prim_config.template_height, prim_config.template_width,
                tex_y, tex_x, dmask_dx_tex, dmask_dy_tex);
            mask_val = fmaxf(0.f, fminf(1.f, mask_val));
        } else {
            mask_val = 0.f;
            dmask_dx_tex = 0.f;
            dmask_dy_tex = 0.f;
        }
        

        // Only compute gradients if mask value is significant (not transparent)
        if (mask_val > 1e-6f) {
            const float d_alpha_dv = tile_config.alpha_upper_bound * mask_val * s_op * (1.f - s_op);
            atomicAdd(&outputs.grad_opacities[n], (dLdalpha * d_alpha_dv));

            // (4) mask path: dα/dmask = α_max * σ(v_op)
            const float dalpha_dmask = tile_config.alpha_upper_bound * s_op;

            // (5) ∂mask/∂(u,v) from texture-space grads:
            // tex_x = 0.5*(u+1)*(W-1), tex_y = 0.5*(v+1)*(H-1)
            const float du2x = 0.5f * (prim_config.template_width  - 1);
            const float dv2y = 0.5f * (prim_config.template_height - 1);
            const float dmask_du = dmask_dx_tex * du2x;
            const float dmask_dv = dmask_dy_tex * dv2y;

            const float dL_dmask = dLdalpha * dalpha_dmask;
            const float dL_du = dL_dmask * dmask_du;
            const float dL_dv = dL_dmask * dmask_dv;

            // (6) ∂(u,v)/∂(μx,μy,r,φ)   (r is scale)
            // u = ( c*dx + s*dy)/r,   v = (-s*dx + c*dy)/r
            const float du_dmx = -c*inv_r,  du_dmy = -s*inv_r;
            const float dv_dmx =  s*inv_r,  dv_dmy = -c*inv_r;
            const float du_dr  = -u*inv_r,  dv_dr  = -v*inv_r;
            const float du_dphi= (-s*ndx +  c*ndy);
            const float dv_dphi= (-c*ndx -  s*ndy);
            
            atomicAdd(&outputs.grad_means2D[2*n+0], (dL_du*du_dmx + dL_dv*dv_dmx));
            atomicAdd(&outputs.grad_means2D[2*n+1], (dL_du*du_dmy + dL_dv*dv_dmy));
            atomicAdd(&outputs.grad_radii[n],       (dL_du*du_dr  + dL_dv*dv_dr));
            atomicAdd(&outputs.grad_rotations[n],   (dL_du*du_dphi+ dL_dv*dv_dphi));
        }

        // advance local k just like forward push order
        k_running++;
        if (k_running >= num_k) break;
    }

    //printf("CUDA Backward Kernel: DEBUG MODE - pixel(%d,%d), K=%d, k_running=%d\n", x, y, K, k_running);
}

// Backward tile kernel (Original parallel version)
__global__ void tile_backward_over_tilelist_kernel(
    const float* grad_out_color,
    const float* grad_out_alpha,
    const TileConfig tile_config,
    const PrimitiveConfig prim_config,
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const LearningRateConfig lr_config
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
    const float gCx = grad_out_color[gidx + 0];
    const float gCy = grad_out_color[gidx + 1];
    const float gCz = grad_out_color[gidx + 2];
    const float gA  = grad_out_alpha ? grad_out_alpha[y * W + x] : 0.f;

    backward_over_one_pixel<true,true>(
        x, y, tile_config, prim_config,
        prim_ids, K,
        gCx, gCy, gCz, gA,
        inputs, outputs, buffers,
        lr_config
    );
}

// DEBUG MODE: Sequential version of the backward kernel for debugging
__global__ void tile_backward_over_tilelist_kernel_debug(
    const float* grad_out_color,
    const float* grad_out_alpha,
    const TileConfig tile_config,
    const PrimitiveConfig prim_config,
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const LearningRateConfig lr_config
) {
    const int W = tile_config.image_width;
    const int H = tile_config.image_height;
    
    // Only thread (0,0) does the work
    if (threadIdx.x != 0 || threadIdx.y != 0 || blockIdx.x != 0 || blockIdx.y != 0) {
        return;
    }
    
    printf("CUDA Backward Kernel: DEBUG MODE - Processing all pixels sequentially\n");
    
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
            const float gCx = grad_out_color[gidx + 0];
            const float gCy = grad_out_color[gidx + 1];
            const float gCz = grad_out_color[gidx + 2];
            const float gA  = grad_out_alpha ? grad_out_alpha[y * W + x] : 0.f;

            // Debug print for first few pixels
            if (x < 2 && y < 2) {
                printf("CUDA Backward Kernel: Debug - pixel(%d,%d), tile_id=%d, K=%d, grad_color=[%.4f,%.4f,%.4f], grad_alpha=%.4f\n", 
                       x, y, tile_id, K, gCx, gCy, gCz, gA);
            }

            backward_over_one_pixel<true,true>(
                x, y, tile_config, prim_config,
                prim_ids, K,
                gCx, gCy, gCz, gA,
                inputs, outputs, buffers,
                lr_config
            );
        }
    }
    printf("CUDA Backward Kernel: DEBUG MODE - Sequential processing completed\n");
}


// =======================================================
// tile_backward.cu와 동일한 인터페이스를 가진 CudaRasterizeTilesBackwardKernel 함수
// =======================================================
void CudaRasterizeTilesBackwardKernel(
    const float* grad_out_color,
    const float* grad_out_alpha,
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const TileConfig tile_config,
    const PrimitiveConfig prim_config,
    const LearningRateConfig lr_config) {
    
#if DEBUG_CUDA_KERNELS
    printf("CUDA Backward Kernel: Starting with %d primitives, %dx%d image\n", prim_config.num_primitives, tile_config.image_width, tile_config.image_height);
#endif

#if DEBUG_SEQUENTIAL
    // DEBUG MODE: Sequential execution for debugging
    // Use only 1 block with 1 thread to avoid race conditions
    dim3 grid(1, 1);  // Single block
    dim3 block(1, 1); // Single thread
    
    printf("CUDA Backward Kernel: DEBUG MODE - Grid size: %dx%d, Block size: %dx%d\n", grid.x, grid.y, block.x, block.y);
    
    printf("CUDA Backward Kernel: About to launch debug kernel...\n");
    
    tile_backward_over_tilelist_kernel_debug<<<grid, block>>>(
        grad_out_color, grad_out_alpha,
        tile_config, prim_config,
        inputs, outputs, buffers,
        lr_config
    );
#else
    dim3 block(tile_config.tile_size, tile_config.tile_size);
    dim3 grid((tile_config.image_width  + tile_config.tile_size - 1) / tile_config.tile_size,
              (tile_config.image_height + tile_config.tile_size - 1) / tile_config.tile_size);
    
#if DEBUG_CUDA_KERNELS
    printf("CUDA Backward Kernel: NORMAL MODE - Grid size: %dx%d, Block size: %dx%d\n", grid.x, grid.y, block.x, block.y);
    
    printf("CUDA Backward Kernel: About to launch normal kernel...\n");
#endif
    
    tile_backward_over_tilelist_kernel<<<grid, block>>>(
        grad_out_color, grad_out_alpha,
        tile_config, prim_config,
        inputs, outputs, buffers,
        lr_config
    );
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Backward Kernel: Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize to ensure kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Backward Kernel: Kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }
    
#if DEBUG_CUDA_KERNELS
    printf("CUDA Backward Kernel: Kernel completed successfully\n");
#endif
}

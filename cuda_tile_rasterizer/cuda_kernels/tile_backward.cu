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
#define SIZE_BASED_OPTIM 1  // Set to 1 to use size-based c_blend and rotation (image_width * SIZE_THRESHOLD_RATIO threshold), 0 for original global c_blend
#define SIZE_THRESHOLD_RATIO 0.2f  // Threshold ratio for size-based optimization (image_width * SIZE_THRESHOLD_RATIO)
#define C_BLEND_INVERSE 0  // Set to 1 to apply c_blend to large primitives (s > threshold), 0 to apply to small primitives (s <= threshold)

// Color map sampling function (same as tile_forward.cu)
__device__ __forceinline__ float3 sample_color_map(
    const float* color_map, int map_h, int map_w, 
    float x, float y, float radius, float rotation) {
    
    // Transform coordinates to color map space
    // 1. Normalize by primitive radius to get [-1, 1] range
    float r_inv = radius > 1e-6f ? (1.0f / radius) : 1e6f;
    float norm_x = x * r_inv;
    float norm_y = y * r_inv;
    
    // 2. Rotate
    float cos_theta = cosf(rotation);
    float sin_theta = sinf(rotation);
    float rot_x =  cos_theta * norm_x + sin_theta * norm_y;
    float rot_y = -sin_theta * norm_x + cos_theta * norm_y;
    
    // 3. Convert normalized [-1, 1] coordinates to color map pixel coordinates [0, W-1], [0, H-1]
    float coord_x = (rot_x + 1.0f) * 0.5f * (map_w - 1);
    float coord_y = (rot_y + 1.0f) * 0.5f * (map_h - 1);
    
    // Clamp coordinates
    coord_x = fmaxf(0.0f, fminf(map_w - 1, coord_x));
    coord_y = fmaxf(0.0f, fminf(map_h - 1, coord_y));
    
    // Bilinear interpolation
    int x0 = (int)coord_x;
    int y0 = (int)coord_y;
    int x1 = min(x0 + 1, map_w - 1);
    int y1 = min(y0 + 1, map_h - 1);
    
    float fx = coord_x - x0;
    float fy = coord_y - y0;
    
    // Sample colors
    float3 c00 = make_float3(
        color_map[3 * (y0 * map_w + x0) + 0],
        color_map[3 * (y0 * map_w + x0) + 1],
        color_map[3 * (y0 * map_w + x0) + 2]
    );
    float3 c01 = make_float3(
        color_map[3 * (y1 * map_w + x0) + 0],
        color_map[3 * (y1 * map_w + x0) + 1],
        color_map[3 * (y1 * map_w + x0) + 2]
    );
    float3 c10 = make_float3(
        color_map[3 * (y0 * map_w + x1) + 0],
        color_map[3 * (y0 * map_w + x1) + 1],
        color_map[3 * (y0 * map_w + x1) + 2]
    );
    float3 c11 = make_float3(
        color_map[3 * (y1 * map_w + x1) + 0],
        color_map[3 * (y1 * map_w + x1) + 1],
        color_map[3 * (y1 * map_w + x1) + 2]
    );
    
    // Interpolate
    float3 c0 = make_float3(
        c00.x * (1.0f - fx) + c10.x * fx,
        c00.y * (1.0f - fx) + c10.y * fx,
        c00.z * (1.0f - fx) + c10.z * fx
    );
    float3 c1 = make_float3(
        c01.x * (1.0f - fx) + c11.x * fx,
        c01.y * (1.0f - fx) + c11.y * fx,
        c01.z * (1.0f - fx) + c11.z * fx
    );
    
    return make_float3(
        c0.x * (1.0f - fy) + c1.x * fy,
        c0.y * (1.0f - fy) + c1.y * fy,
        c0.z * (1.0f - fy) + c1.z * fy
    );
}

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
#if SIZE_BASED_OPTIM
        // Calculate rotation_scale early for use in mask computation
        const float threshold = (float)tile_config.image_width * SIZE_THRESHOLD_RATIO;
        const float s_radius = inputs.radii[n];
        // const float c_blend_local = fmaxf(0.0f, fminf(1.0f, 1.0f - (s_radius / threshold)));
        // const float rotation_scale = 1.0f - c_blend_local;
#if C_BLEND_INVERSE
        // Binary c_blend: s > threshold -> inputs.c_blend, s <= threshold -> 0 (apply to large primitives)
        const float c_blend_local = (s_radius > threshold) ? inputs.c_blend : 0.0f;
        const float rotation_scale = (s_radius > threshold) ? 1.0f : 0.0f;
#else
        // Binary c_blend: s > threshold -> 0, s <= threshold -> inputs.c_blend (apply to small primitives)
        const float c_blend_local = (s_radius <= threshold) ? inputs.c_blend : 0.0f;
        const float rotation_scale = (s_radius <= threshold) ? 1.0f : 0.0f;
#endif
#endif
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const float inv_r = (r > 1e-6f) ? (1.f/r) : 1e6f;
#if SIZE_BASED_OPTIM
            const float phi_original = inputs.rotations[n];
            const float phi = phi_original * rotation_scale;  // Apply size-based rotation scaling
#else
            const float phi = inputs.rotations[n];
#endif
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
#if SIZE_BASED_OPTIM
            // Size-based c_blend and rotation: threshold = image_width * SIZE_THRESHOLD_RATIO
            const float threshold = (float)tile_config.image_width * SIZE_THRESHOLD_RATIO;
            const float s_radius = inputs.radii[n];
            // const float c_blend_local = fmaxf(0.0f, fminf(1.0f, 1.0f - (s_radius / threshold)));
            // const float rotation_scale = 1.0f - c_blend_local;  // Same as c_blend: large primitives get 0 rotation, small get full rotation
#if C_BLEND_INVERSE
            // Binary c_blend: s > threshold -> inputs.c_blend, s <= threshold -> 0 (apply to large primitives)
            const float c_blend_local = (s_radius > threshold) ? inputs.c_blend : 0.0f;
            const float rotation_scale = (s_radius > threshold) ? 1.0f : 0.0f;  // Large primitives get full rotation, small get 0 rotation
#else
            // Binary c_blend: s > threshold -> 0, s <= threshold -> inputs.c_blend (apply to small primitives)
            const float c_blend_local = (s_radius <= threshold) ? inputs.c_blend : 0.0f;
            const float rotation_scale = (s_radius <= threshold) ? 1.0f : 0.0f;  // Small primitives get full rotation, large get 0 rotation
#endif
            // c_i gradient: ∂L/∂c_i = gCx * (1 - c_blend) * (α_k T_k)
            const float dcr = gCx * (a_k * T_k) * (1.0f - c_blend_local);
            const float dcg = gCy * (a_k * T_k) * (1.0f - c_blend_local);
            const float dcb = gCz * (a_k * T_k) * (1.0f - c_blend_local);
#else
            // Original: no c_blend consideration for c_i gradient
            const float dcr = gCx * (a_k * T_k);
            const float dcg = gCy * (a_k * T_k);
            const float dcb = gCz * (a_k * T_k);
#endif
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
#if SIZE_BASED_OPTIM
        const float phi_original = inputs.rotations[n];
        const float phi = phi_original * rotation_scale;  // Apply size-based rotation scaling
#else
        const float phi = inputs.rotations[n];
#endif
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
#if SIZE_BASED_OPTIM
            // rotation gradient: ∂L/∂phi_original = ∂L/∂phi * ∂phi/∂phi_original = ∂L/∂phi * rotation_scale
            atomicAdd(&outputs.grad_rotations[n],   (dL_du*du_dphi+ dL_dv*dv_dphi) * rotation_scale);
#else
            atomicAdd(&outputs.grad_rotations[n],   (dL_du*du_dphi+ dL_dv*dv_dphi));
#endif
            
#if SIZE_BASED_OPTIM
            // Additional gradient for r through c_blend
            // ∂L/∂r += ∂L/∂sr * ∂sr/∂c_blend * ∂c_blend/∂r * (α_k T_k)
            if (inputs.colors_orig != nullptr) {
                const float threshold = (float)tile_config.image_width * SIZE_THRESHOLD_RATIO;
                const float s_radius = inputs.radii[n];
                // const float c_blend_local = fmaxf(0.0f, fminf(1.0f, 1.0f - (s_radius / threshold)));
                // const float rotation_scale = 1.0f - c_blend_local;
#if C_BLEND_INVERSE
                // Binary c_blend: s > threshold -> inputs.c_blend, s <= threshold -> 0 (apply to large primitives)
                const float c_blend_local = (s_radius > threshold) ? inputs.c_blend : 0.0f;
                const float rotation_scale = (s_radius > threshold) ? 1.0f : 0.0f;
#else
                // Binary c_blend: s > threshold -> 0, s <= threshold -> inputs.c_blend (apply to small primitives)
                const float c_blend_local = (s_radius <= threshold) ? inputs.c_blend : 0.0f;
                const float rotation_scale = (s_radius <= threshold) ? 1.0f : 0.0f;
#endif
                
                // Only compute gradient if c_blend is active (s <= threshold and inputs.c_blend > 0)
                if (c_blend_local > 0.0f) {
                    // Forward에서 사용한 c_i와 c_o 재계산
                    const float c_i_r = sigmoidf_safe(inputs.colors[3*n+0]);
                    const float c_i_g = sigmoidf_safe(inputs.colors[3*n+1]);
                    const float c_i_b = sigmoidf_safe(inputs.colors[3*n+2]);
                    
                    // c_o 샘플링 (forward와 동일한 로직)
                    float3 c_o = sample_color_map(
                        inputs.colors_orig + n * prim_config.template_height * prim_config.template_width * 3,
                        prim_config.template_height, prim_config.template_width,
                        (float)x - inputs.means2D[2*n + 0], (float)y - inputs.means2D[2*n + 1],
                        inputs.radii[n], 
#if SIZE_BASED_OPTIM
                        inputs.rotations[n] * rotation_scale
#else
                        inputs.rotations[n]
#endif
                    );
                    
                    // ∂c_blend/∂r = -1/threshold (s < threshold일 때만, 아니면 0)
                    // const float dc_blend_dr = (-1.0f / threshold);

                    // ∂c_blend/∂r: For binary c_blend, gradient is 0 at threshold (step function)
                    // But we can approximate: if s is near threshold, small change in r can affect c_blend
                    // Actually, for binary case, gradient is infinite at threshold, but we use 0 for simplicity
                    // Or we can use a smoothed approximation near threshold
                    const float dc_blend_dr = 0.0f;  // Binary function has zero gradient except at threshold
                    
                    // ∂L/∂r += ∂L/∂sr * ∂sr/∂c_blend * ∂c_blend/∂r * (α_k T_k)
                    // = gCx * (c_o - c_i) * dc_blend_dr * (α_k T_k)
                    const float dLdr_blend = (gCx * (c_o.x - c_i_r) + 
                                              gCy * (c_o.y - c_i_g) + 
                                              gCz * (c_o.z - c_i_b)) * dc_blend_dr * (a_k * T_k);
                    atomicAdd(&outputs.grad_radii[n], dLdr_blend);
                }
            }
#endif
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

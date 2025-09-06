#ifndef TILE_COMMON_FP16_H
#define TILE_COMMON_FP16_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <math.h>

#ifdef __CUDACC__
    #define HD __host__ __device__
    #define DEV __device__ __forceinline__
#else
    #define HD
    #define DEV
#endif

HD static inline int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// =======================================================
// Common structures and utilities for tile-based rasterization (FP16)
// =======================================================

// --------- Matrix/Vector utilities ----------
struct half2x2 { 
    __half a00, a01, a10, a11; 
};

// =======================================================
// Configuration structures
// =======================================================

// Tile configuration
class TileConfigFP16 {
public:
    int image_height;
    int image_width;
    int tile_size;
    __half sigma;
    __half alpha_upper_bound;
    int total_tiles;
    int tile_width;
    int tile_height;
    
    __host__ __device__ TileConfigFP16(int image_height, int image_width, int tile_size, __half sigma, __half alpha_upper_bound);
};

// Primitive configuration
class PrimitiveConfigFP16 {
public:
    int num_primitives;
    int num_templates;
    int template_height;
    int template_width;
    int max_prims_per_pixel;
    
    __host__ __device__ PrimitiveConfigFP16(int num_primitives, int num_templates, int template_height, int template_width, int max_prims_per_pixel);
};

// Input tensor group for cleaner function signatures
class InputTensorsFP16 {
public:
    const __half* means2D;           // [P, 2] - primitive positions
    const __half* radii;             // [P] - primitive radii
    const __half* rotations;         // [P] - primitive rotations
    const __half* opacities;         // [P] - opacity logits
    const __half* colors;            // [P, 3] - color logits
    const __half* primitive_templates; // [T, Ht, Wt] - template masks
    const int* global_bmp_sel;      // [P] - template selection indices
    
    __host__ __device__ InputTensorsFP16(
        const __half* means2D, const __half* radii, const __half* rotations,
        const __half* opacities, const __half* colors, const __half* primitive_templates,
        const int* global_bmp_sel);
};

// Output tensor group for gradients
class OutputTensorsFP16 {
public:
    __half* out_color;         // [H*W*3] - output color
    __half* out_alpha;         // [H*W] - output alpha
    __half* grad_means2D;      // [P, 2] - gradients for positions
    __half* grad_radii;        // [P] - gradients for radii
    __half* grad_rotations;    // [P] - gradients for rotations
    __half* grad_opacities;    // [P] - gradients for opacities
    __half* grad_colors;       // [P, 3] - gradients for colors
    
    __host__ __device__ OutputTensorsFP16(
        __half* out_color, __half* out_alpha,
        __half* grad_means2D, __half* grad_radii, __half* grad_rotations,
        __half* grad_opacities, __half* grad_colors);
};

// Global memory buffers for transmit-over compositing
class GlobalBuffersFP16 {
public:
    __half* pixel_alphas;      // [H*W*Kmax] - per-pixel alpha values
    __half* pixel_colors_r;    // [H*W*Kmax] - per-pixel red values
    __half* pixel_colors_g;    // [H*W*Kmax] - per-pixel green values
    __half* pixel_colors_b;    // [H*W*Kmax] - per-pixel blue values
    __half* pixel_T_values;    // [H*W*Kmax] - transmittance values
    int* pixel_prim_counts;   // [H*W] - primitive counts per pixel
    __half* sigma_inv;         // [H*W] - inverse of sigma
    __half* grad_sigma;        // [H*W] - gradient of sigma
    
    // Tile-related buffers for backward pass
    int* d_tile_offsets;      // [num_tiles + 1] - device tile offsets
    int* d_tile_indices;      // [total_prims] - device tile indices
    int tile_offsets_size;    // Size of tile_offsets array
    int tile_indices_size;    // Size of tile_indices array
    
    __host__ __device__ GlobalBuffersFP16(
        __half* pixel_alphas, __half* pixel_colors_r, __half* pixel_colors_g, __half* pixel_colors_b,
        __half* pixel_T_values, int* pixel_prim_counts, __half* sigma_inv, __half* grad_sigma,
        int* d_tile_offsets = nullptr, int* d_tile_indices = nullptr,
        int tile_offsets_size = 0, int tile_indices_size = 0);
};

// Learning rate configuration structure
class LearningRateConfigFP16 {
public:
    __half default_lr;
    __half gain_x;
    __half gain_y;
    __half gain_r;
    __half gain_v;
    __half gain_theta;
    __half gain_c;
    
    __host__ __device__ LearningRateConfigFP16(__half lr, __half gx, __half gy, __half gr, __half gv, __half gt, __half gc);
};

// --------- Math utilities ----------
__device__ __forceinline__ __half sigmoidf_safe_fp16(__half x) {
    if (__half2float(x) >= 0.0f) {
        __half z = __float2half(expf(-__half2float(x)));
        return __float2half(1.0f / (1.0f + __half2float(z)));
    } else {
        __half z = __float2half(expf(__half2float(x)));
        return __float2half(__half2float(z) / (1.0f + __half2float(z)));
    }
}

__device__ __forceinline__ __half clampf_fp16(__half x, __half min_val, __half max_val) {
    float x_f = __half2float(x);
    float min_f = __half2float(min_val);
    float max_f = __half2float(max_val);
    return __float2half(fmaxf(min_f, fminf(max_f, x_f)));
}

// --------- Atomic operations ----------
// __device__ __forceinline__ void atomicAdd2_fp16(__half* g, __half x, __half y) {
//     atomicAdd(g, x);
//     atomicAdd(g + 1, y);
// }

// __device__ __forceinline__ void atomicAdd3_fp16(__half* g, __half x, __half y, __half z) {
//     atomicAdd(g, x);
//     atomicAdd(g + 1, y);
//     atomicAdd(g + 2, z);
// }

// __device__ __forceinline__ void atomicAddM2_fp16(__half* g, __half m00, __half m01, __half m10, __half m11) {
//     atomicAdd(g, m00);
//     atomicAdd(g + 1, m01);
//     atomicAdd(g + 2, m10);
//     atomicAdd(g + 3, m11);
// }

// __device__ __forceinline__ half2 mul_fp16(const half2x2& a, const half2& b) {
//     float a00_f = __half2float(a.a00), a01_f = __half2float(a.a01);
//     float a10_f = __half2float(a.a10), a11_f = __half2float(a.a11);
//     float bx_f = __half2float(b.x), by_f = __half2float(b.y);
    
//     float result_x = a00_f * bx_f + a01_f * by_f;
//     float result_y = a10_f * bx_f + a11_f * by_f;
    
//     return make_half2(__float2half(result_x), __float2half(result_y));
// }

// ---- FP16 atomic helpers (device-only) ----
#ifdef __CUDACC__  // Hide from host compiler (g++)
DEV void atomicAdd2_fp16(__half* g, __half x, __half y) {
    atomicAdd(&g[0], x);
    atomicAdd(&g[1], y);
}
DEV void atomicAdd3_fp16(__half* g, __half x, __half y, __half z) {
    atomicAdd(&g[0], x);
    atomicAdd(&g[1], y);
    atomicAdd(&g[2], z);
}
DEV void atomicAddM2_fp16(__half* g, __half m00, __half m01, __half m10, __half m11) {
    atomicAdd(&g[0], m00);
    atomicAdd(&g[1], m01);
    atomicAdd(&g[2], m10);
    atomicAdd(&g[3], m11);
}
DEV half2 mul_fp16(const half2x2& a, const half2& b) {
    float a00_f = __half2float(a.a00), a01_f = __half2float(a.a01);
    float a10_f = __half2float(a.a10), a11_f = __half2float(a.a11);
    float bx_f = __half2float(b.x), by_f = __half2float(b.y);
    
    float result_x = a00_f * bx_f + a01_f * by_f;
    float result_y = a10_f * bx_f + a11_f * by_f;
    
    return make_half2(__float2half(result_x), __float2half(result_y));
}
#endif

// --------- Indexing utilities ----------
__device__ __forceinline__ int idx_nhw(int n, int h, int w, int H, int W) {
    return n * H * W + h * W + w;
}

__device__ __forceinline__ int idx_hwc(int h, int w, int W, int c) {
    return (h * W + w) * 3 + c;
}

// =======================================================
// Bilinear sampling utilities (FP16)
// =======================================================

// Simple bilinear sample (for forward pass)
// __device__ __forceinline__ __half bilinear_sample_fp16(const __half* data, int height, int width, __half y, __half x) {
HD inline __half bilinear_sample_fp16(const __half* data, int height, int width, __half y, __half x) {
    // 완전 OOB는 즉시 0 반환 (int 캐스팅 전에 컷)
    float x_f = __half2float(x);
    float y_f = __half2float(y);
    if (x_f < 0.0f || x_f > (float)(width-1) || y_f < 0.0f || y_f > (float)(height-1))
        return __float2half(0.0f);

    int x0 = (int)floorf(x_f);
    int y0 = (int)floorf(y_f);
    x0 = clampi(x0, 0, width - 1);
    y0 = clampi(y0, 0, height - 1);

    float wx = x_f - x0;
    float wy = y_f - y0;
    
    auto in = [&](int yy, int xx){ return (0 <= xx && xx < width && 0 <= yy && yy < height); };
    auto at = [&](int yy, int xx){ return in(yy,xx) ? __half2float(data[yy*width + xx]) : 0.0f; };

    float v00 = at(y0,   x0   );
    float v01 = at(y0,   x0+1 );
    float v10 = at(y0+1, x0   );
    float v11 = at(y0+1, x0+1 );

    float result = v00 * (1.0f - wx) * (1.0f - wy) + v01 * wx * (1.0f - wy) +
                   v10 * (1.0f - wx) * wy + v11 * wx * wy;
    
    return __float2half(result);
}

// Bilinear sample + gradients w.r.t. x (u) and y (v) in template space
// __device__ __forceinline__ __half bilinear_value_and_grad_xy_fp16(const __half* data, int height, int width, __half y, __half x, __half& grad_x, __half& grad_y) {
HD inline __half bilinear_value_and_grad_xy_fp16(const __half* data, int height, int width, __half y, __half x, __half& grad_x, __half& grad_y) {
    // 완전 OOB는 값/그라드 0
    float x_f = __half2float(x);
    float y_f = __half2float(y);
    if (x_f < 0.0f || x_f > (float)(width-1) || y_f < 0.0f || y_f > (float)(height-1)) {
        grad_x = __float2half(0.0f); grad_y = __float2half(0.0f);
        return __float2half(0.0f);
    }

    int x0 = (int)floorf(x_f);
    int y0 = (int)floorf(y_f);
    x0 = clampi(x0, 0, width - 1);
    y0 = clampi(y0, 0, height - 1);

    float wx = x_f - x0;
    float wy = y_f - y0;

    auto in = [&](int yy, int xx){ return (0 <= xx && xx < width && 0 <= yy && yy < height); };
    auto at = [&](int yy, int xx){ return in(yy,xx) ? __half2float(data[yy*width + xx]) : 0.0f; };

    float v00 = at(y0,   x0   );
    float v01 = at(y0,   x0+1 );
    float v10 = at(y0+1, x0   );
    float v11 = at(y0+1, x0+1 );

    // value
    float f = v00 * (1.0f-wx)*(1.0f-wy) +
              v01 * wx     *(1.0f-wy) +
              v10 * (1.0f-wx)*wy      +
              v11 * wx     *wy;

    // grads: OOB 이웃은 값이 0이므로 그대로 선형식 사용하면 됨
    float grad_x_f = (v01 - v00) * (1.0f - wy) + (v11 - v10) * wy;
    float grad_y_f = (v10 - v00) * (1.0f - wx) + (v11 - v01) * wx;

    // 완전히 OOB인 경우(네 모서리 모두 OOB) PyTorch는 출력/그래드 0
    if (!in(y0, x0) && !in(y0, x0+1) && !in(y0+1, x0) && !in(y0+1, x0+1)) {
        grad_x_f = 0.0f; grad_y_f = 0.0f;
    }
    
    grad_x = __float2half(grad_x_f);
    grad_y = __float2half(grad_y_f);
    return __float2half(f);
}

#endif // TILE_COMMON_FP16_H

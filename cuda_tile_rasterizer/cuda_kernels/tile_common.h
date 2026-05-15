#ifndef TILE_COMMON_H
#define TILE_COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#define DCT_DEFORM_BASIS_COUNT 8

// =======================================================
// Common structures and utilities for tile-based rasterization
// =======================================================

// --------- Matrix/Vector utilities ----------
struct float2x2 { 
    float a00, a01, a10, a11; 
};


// =======================================================
// Configuration structures
// =======================================================

// Tile configuration
class TileConfig {
public:
    int image_height;
    int image_width;
    int tile_size;
    float sigma;
    float alpha_upper_bound;
    int total_tiles;
    int tile_width;
    int tile_height;
    
    __host__ __device__ TileConfig(int image_height, int image_width, int tile_size, float sigma, float alpha_upper_bound);
};

// Primitive configuration
class PrimitiveConfig {
public:
    int num_primitives;
    int num_templates;
    int template_height;
    int template_width;
    int max_prims_per_pixel;
    
    __host__ __device__ PrimitiveConfig(int num_primitives, int num_templates, int template_height, int template_width, int max_prims_per_pixel);
};

// Input tensor group for cleaner function signatures
class InputTensors {
public:
    const float* means2D;           // [P, 2] - primitive positions
    const float* radii;             // [P] - primitive radii
    const float* rotations;         // [P] - primitive rotations
    const float* opacities;         // [P] - opacity logits
    const float* colors;            // [P, 3] - color logits
    const float* colors_orig;       // [P, H, W, 3] - original color map logits (no gradient)
    const float* primitive_templates; // [T, Ht, Wt] - template masks
    const float* deform_coeffs;     // [P, 8, 2] - DCT local displacement coefficients
    const int* global_bmp_sel;      // [P] - template selection indices
    float c_blend;                  // Color blending factor
    float deform_max_disp;          // Maximum local displacement in normalized primitive coords
    
    __host__ __device__ InputTensors(
        const float* means2D, const float* radii, const float* rotations,
        const float* opacities, const float* colors, const float* colors_orig, 
        const float* primitive_templates, const float* deform_coeffs,
        const int* global_bmp_sel, float c_blend, float deform_max_disp);
};

// Output tensor group for gradients
class OutputTensors {
public:
    float* out_color;         // [H*W*3] - output color
    float* out_alpha;         // [H*W] - output alpha
    float* grad_means2D;      // [P, 2] - gradients for positions
    float* grad_radii;        // [P] - gradients for radii
    float* grad_rotations;    // [P] - gradients for rotations
    float* grad_opacities;    // [P] - gradients for opacities
    float* grad_colors;       // [P, 3] - gradients for colors
    float* grad_deform_coeffs; // [P, 8, 2] - gradients for DCT deformation coeffs
    
    __host__ __device__ OutputTensors(
        float* out_color, float* out_alpha,
        float* grad_means2D, float* grad_radii, float* grad_rotations,
        float* grad_opacities, float* grad_colors, float* grad_deform_coeffs);
};

// Global memory buffers for transmit-over compositing
class GlobalBuffers {
public:
    float* pixel_alphas;      // [H*W*Kmax] - per-pixel alpha values
    float* pixel_colors_r;    // [H*W*Kmax] - per-pixel red values
    float* pixel_colors_g;    // [H*W*Kmax] - per-pixel green values
    float* pixel_colors_b;    // [H*W*Kmax] - per-pixel blue values
    float* pixel_T_values;    // [H*W*Kmax] - transmittance values
    int* pixel_prim_counts;   // [H*W] - primitive counts per pixel
    float* sigma_inv;         // [H*W] - inverse of sigma
    float* grad_sigma;        // [H*W] - gradient of sigma
    
    // Tile-related buffers for backward pass
    int* d_tile_offsets;      // [num_tiles + 1] - device tile offsets
    int* d_tile_indices;      // [total_prims] - device tile indices
    int tile_offsets_size;    // Size of tile_offsets array
    int tile_indices_size;    // Size of tile_indices array
    
    __host__ __device__ GlobalBuffers(
        float* pixel_alphas, float* pixel_colors_r, float* pixel_colors_g, float* pixel_colors_b,
        float* pixel_T_values, int* pixel_prim_counts, float* sigma_inv, float* grad_sigma,
        int* d_tile_offsets = nullptr, int* d_tile_indices = nullptr,
        int tile_offsets_size = 0, int tile_indices_size = 0);
};

// Learning rate configuration structure
class LearningRateConfig {
public:
    float default_lr;
    float gain_x;
    float gain_y;
    float gain_r;
    float gain_v;
    float gain_theta;
    float gain_c;
    
    __host__ __device__ LearningRateConfig(float lr, float gx, float gy, float gr, float gv, float gt, float gc);
};

// --------- Math utilities ----------
__device__ __forceinline__ float sigmoidf_safe(float x) {
    if (x >= 0) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

__device__ __forceinline__ float clampf(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

__device__ __forceinline__ float2 mul(const float2x2& a, const float2& b) {
    return make_float2(a.a00 * b.x + a.a01 * b.y, a.a10 * b.x + a.a11 * b.y);
}

// --------- Atomic operations ----------
__device__ __forceinline__ void atomicAdd2(float* g, float x, float y) {
    atomicAdd(g, x);
    atomicAdd(g + 1, y);
}
__device__ __forceinline__ void atomicAdd3(float* g, float x, float y, float z) {
    atomicAdd(g, x);
    atomicAdd(g + 1, y);
    atomicAdd(g + 2, z);
}

__device__ __forceinline__ void atomicAddM2(float* g, float m00, float m01, float m10, float m11) {
    atomicAdd(g, m00);
    atomicAdd(g + 1, m01);
    atomicAdd(g + 2, m10);
    atomicAdd(g + 3, m11);
}

// --------- DCT deformation utilities ----------
__device__ __forceinline__ void dct_deform_mode(int k, int& fu, int& fv) {
    const int modes_u[DCT_DEFORM_BASIS_COUNT] = {1, 0, 1, 2, 0, 2, 1, 2};
    const int modes_v[DCT_DEFORM_BASIS_COUNT] = {0, 1, 1, 0, 2, 1, 2, 2};
    fu = modes_u[k];
    fv = modes_v[k];
}

__device__ __forceinline__ void dct_basis_and_grad(
    int k, float u, float v, float& phi, float& dphi_du, float& dphi_dv) {
    int fu, fv;
    dct_deform_mode(k, fu, fv);
    const float uu = 0.5f * (u + 1.0f);
    const float vv = 0.5f * (v + 1.0f);
    const float au = CUDART_PI_F * (float)fu * uu;
    const float av = CUDART_PI_F * (float)fv * vv;
    const float cu = __cosf(au);
    const float cv = __cosf(av);
    const float su = __sinf(au);
    const float sv = __sinf(av);
    phi = cu * cv;
    dphi_du = (fu == 0) ? 0.0f : -0.5f * CUDART_PI_F * (float)fu * su * cv;
    dphi_dv = (fv == 0) ? 0.0f : -0.5f * CUDART_PI_F * (float)fv * cu * sv;
}

__device__ __forceinline__ void apply_dct_deformation(
    const InputTensors& inputs, int prim_idx, float u, float v,
    float& u_def, float& v_def) {
    u_def = u;
    v_def = v;
    if (inputs.deform_coeffs == nullptr || inputs.deform_max_disp <= 0.0f) {
        return;
    }

    float raw_u = 0.0f;
    float raw_v = 0.0f;
    const int coeff_base = prim_idx * DCT_DEFORM_BASIS_COUNT * 2;
    for (int k = 0; k < DCT_DEFORM_BASIS_COUNT; ++k) {
        float phi, dphi_du, dphi_dv;
        dct_basis_and_grad(k, u, v, phi, dphi_du, dphi_dv);
        raw_u += inputs.deform_coeffs[coeff_base + 2 * k + 0] * phi;
        raw_v += inputs.deform_coeffs[coeff_base + 2 * k + 1] * phi;
    }
    u_def = u + inputs.deform_max_disp * tanhf(raw_u);
    v_def = v + inputs.deform_max_disp * tanhf(raw_v);
}

// --------- Indexing utilities ----------
__device__ __forceinline__ int idx_nhw(int n, int h, int w, int H, int W) {
    return n * H * W + h * W + w;
}

__device__ __forceinline__ int idx_hwc(int h, int w, int W, int c) {
    return (h * W + w) * 3 + c;
}

// =======================================================
// Bilinear sampling utilities
// =======================================================

// Simple bilinear sample (for forward pass)
__device__ __forceinline__ float bilinear_sample(const float* data, int height, int width, float y, float x) {
    // 완전 OOB는 즉시 0 반환 (int 캐스팅 전에 컷)
    if (x < 0.f || x > (float)(width-1) || y < 0.f || y > (float)(height-1))
        return 0.f;

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    x0 = max(0, min(x0, width - 1));
    y0 = max(0, min(y0, height - 1));

    float wx = x - x0;
    float wy = y - y0;
    
    auto in = [&](int yy, int xx){ return (0 <= xx && xx < width && 0 <= yy && yy < height); };
    auto at = [&](int yy, int xx){ return in(yy,xx) ? data[yy*width + xx] : 0.f; };

    float v00 = at(y0,   x0   );
    float v01 = at(y0,   x0+1 );
    float v10 = at(y0+1, x0   );
    float v11 = at(y0+1, x0+1 );

    return v00 * (1.f - wx) * (1.f - wy) + v01 * wx * (1.f - wy) +
           v10 * (1.f - wx) * wy + v11 * wx * wy;
}

// Bilinear sample + gradients w.r.t. x (u) and y (v) in template space
__device__ __forceinline__ float bilinear_value_and_grad_xy(const float* data, int height, int width, float y, float x, float& grad_x, float& grad_y) {
    // 완전 OOB는 값/그라드 0
    if (x < 0.f || x > (float)(width-1) || y < 0.f || y > (float)(height-1)) {
        grad_x = 0.f; grad_y = 0.f;
        return 0.f;
    }

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    x0 = max(0, min(x0, width - 1));
    y0 = max(0, min(y0, height - 1));

    float wx = x - x0;
    float wy = y - y0;

    auto in = [&](int yy, int xx){ return (0 <= xx && xx < width && 0 <= yy && yy < height); };
    auto at = [&](int yy, int xx){ return in(yy,xx) ? data[yy*width + xx] : 0.f; };

    float v00 = at(y0,   x0   );
    float v01 = at(y0,   x0+1 );
    float v10 = at(y0+1, x0   );
    float v11 = at(y0+1, x0+1 );

    // value
    float f = v00 * (1.f-wx)*(1.f-wy) +
              v01 * wx     *(1.f-wy) +
              v10 * (1.f-wx)*wy      +
              v11 * wx     *wy;

    // grads: OOB 이웃은 값이 0이므로 그대로 선형식 사용하면 됨
    grad_x = (v01 - v00) * (1.f - wy) + (v11 - v10) * wy;
    grad_y = (v10 - v00) * (1.f - wx) + (v11 - v01) * wx;

    // 완전히 OOB인 경우(네 모서리 모두 OOB) PyTorch는 출력/그래드 0
    if (!in(y0, x0) && !in(y0, x0+1) && !in(y0+1, x0) && !in(y0+1, x0+1)) {
        grad_x = 0.f; grad_y = 0.f;
    }
    return f;
}


#endif // TILE_COMMON_H

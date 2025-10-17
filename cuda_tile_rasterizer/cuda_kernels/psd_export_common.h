#ifndef PSD_EXPORT_COMMON_H
#define PSD_EXPORT_COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cstdint>

// =======================================================
// PSD Export specific structures and utilities
// =======================================================

// Configuration for PSD export kernels
class PSDExportConfig {
public:
    int N;                    // Number of primitives
    int H, W;                 // Canvas dimensions
    int template_height;      // Template height
    int template_width;       // Template width
    float scale_factor;       // Export scaling factor
    float alpha_upper_bound;  // Maximum alpha value (from psd_exporter.py)
    float c_blend;            // Color blending factor
    const float* colors_orig; // Original primitive colors (c_o) for blending
    
    __host__ __device__ PSDExportConfig(int N, int H, int W, int template_height, int template_width, 
                                       float scale_factor = 1.0f, float alpha_upper_bound = 1.0f,
                                       float c_blend = 0.0f, const float* colors_orig = nullptr)
        : N(N), H(H), W(W), template_height(template_height), template_width(template_width),
          scale_factor(scale_factor), alpha_upper_bound(alpha_upper_bound), c_blend(c_blend), colors_orig(colors_orig) {}
};

// Input tensors for PSD export
class PSDInputTensors {
public:
    const float* means2D;           // [N, 2] - primitive positions
    const float* radii;             // [N] - primitive radii
    const float* rotations;         // [N] - primitive rotations
    const float* colors;            // [N, 3] - RGB colors (sigmoid applied)
    const float* visibility;        // [N] - visibility (sigmoid applied)
    const float* primitive_templates; // [P, H_t, W_t] - template masks
    const int* global_bmp_sel;      // [N] - template selection indices
    
    __host__ __device__ PSDInputTensors(
        const float* means2D, const float* radii, const float* rotations,
        const float* colors, const float* visibility, const float* primitive_templates,
        const int* global_bmp_sel)
        : means2D(means2D), radii(radii), rotations(rotations), colors(colors),
          visibility(visibility), primitive_templates(primitive_templates), global_bmp_sel(global_bmp_sel) {}
};

// =======================================================
// Grid sampling utilities (matching PyTorch F.grid_sample)
// =======================================================

__device__ __forceinline__ float psd_grid_sample_bilinear(
    const float* template_data, int template_idx,
    int px, int py, float x, float y, float r, float theta,
    const PSDExportConfig& config
) {
    // Normalize coordinates to [-1, 1] (same as PyTorch grid_sample)
    float u_norm = ((float)px - x) / (r );
    float v_norm = ((float)py - y) / (r );
    
    // Apply inverse rotation
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float u_rot = cos_t * u_norm + sin_t * v_norm;
    float v_rot = -sin_t * u_norm + cos_t * v_norm;
    
    // Convert to template coordinates [0, W_t-1] x [0, H_t-1]
    float u_template = (u_rot + 1.0f) * 0.5f * (config.template_width - 1);
    float v_template = (v_rot + 1.0f) * 0.5f * (config.template_height - 1);
    
    // Boundary check
    if (u_template < 0.0f || u_template >= config.template_width - 1 ||
        v_template < 0.0f || v_template >= config.template_height - 1) {
        return 0.0f;
    }
    
    // Bilinear interpolation
    int u0 = (int)floorf(u_template);
    int v0 = (int)floorf(v_template);
    u0 = max(0, min(u0, config.template_width - 1));
    v0 = max(0, min(v0, config.template_height - 1));
    int u1 = u0 + 1;
    int v1 = v0 + 1;
    
    float wu = u_template - u0;
    float wv = v_template - v0;
    float wu_inv = 1.0f - wu;
    float wv_inv = 1.0f - wv;
    
    // Template data indexing
    int base_idx = template_idx * config.template_height * config.template_width;
    float val00 = template_data[base_idx + v0 * config.template_width + u0];
    float val01 = template_data[base_idx + v0 * config.template_width + u1];
    float val10 = template_data[base_idx + v1 * config.template_width + u0];
    float val11 = template_data[base_idx + v1 * config.template_width + u1];
    
    // Bilinear interpolation result
    return wu_inv * wv_inv * val00 + wu * wv_inv * val01 +
           wu_inv * wv * val10 + wu * wv * val11;
}

// =======================================================
// Utility functions
// =======================================================

__device__ __forceinline__ float safe_sigmoid(float x) {
    if (x >= 0) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

__device__ __forceinline__ uint8_t float_to_uint8(float val) {
    return (uint8_t)fmaxf(0.0f, fminf(255.0f, val * 255.0f));
}

// Color map sampling for PSD export (same as tile_forward.cu)
__device__ __forceinline__ float3 psd_sample_color_map(
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

#endif // PSD_EXPORT_COMMON_H

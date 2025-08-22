#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "tile_common.h"

// Configuration struct constructors
__host__ __device__ TileConfig::TileConfig(int image_height, int image_width, int tile_size, float sigma, float alpha_upper_bound)
    : image_height(image_height), image_width(image_width), tile_size(tile_size), sigma(sigma), alpha_upper_bound(alpha_upper_bound) {
    total_tiles = ((image_width + tile_size - 1) / tile_size) * ((image_height + tile_size - 1) / tile_size);
    tile_width = (image_width + tile_size - 1) / tile_size;
    tile_height = (image_height + tile_size - 1) / tile_size;
}

__host__ __device__ PrimitiveConfig::PrimitiveConfig(int num_primitives, int num_templates, int template_height, int template_width, int max_prims_per_pixel)
    : num_primitives(num_primitives), num_templates(num_templates), template_height(template_height), template_width(template_width), max_prims_per_pixel(max_prims_per_pixel) {}

// Input tensor group constructors
__host__ __device__ InputTensors::InputTensors(
    const float* means2D, const float* radii, const float* rotations,
    const float* opacities, const float* colors, const float* primitive_templates,
    const int* global_bmp_sel)
    : means2D(means2D), radii(radii), rotations(rotations),
      opacities(opacities), colors(colors), primitive_templates(primitive_templates),
      global_bmp_sel(global_bmp_sel) {}

// Output tensor group constructors
__host__ __device__ OutputTensors::OutputTensors(
    float* out_color, float* out_alpha,
    float* grad_means2D, float* grad_radii, float* grad_rotations,
    float* grad_opacities, float* grad_colors)
    : out_color(out_color), out_alpha(out_alpha),
      grad_means2D(grad_means2D), grad_radii(grad_radii), grad_rotations(grad_rotations),
      grad_opacities(grad_opacities), grad_colors(grad_colors) {}

// Global buffers constructors
__host__ __device__ GlobalBuffers::GlobalBuffers(
    float* pixel_alphas, float* pixel_colors_r, float* pixel_colors_g, float* pixel_colors_b,
    float* pixel_T_values, int* pixel_prim_counts, float* sigma_inv, float* grad_sigma,
    int* d_tile_offsets, int* d_tile_indices, int tile_offsets_size, int tile_indices_size)
    : pixel_alphas(pixel_alphas), pixel_colors_r(pixel_colors_r), pixel_colors_g(pixel_colors_g),
      pixel_colors_b(pixel_colors_b), pixel_T_values(pixel_T_values), pixel_prim_counts(pixel_prim_counts),
      sigma_inv(sigma_inv), grad_sigma(grad_sigma),
      d_tile_offsets(d_tile_offsets), d_tile_indices(d_tile_indices), tile_offsets_size(tile_offsets_size), tile_indices_size(tile_indices_size) {}

__host__ __device__ LearningRateConfig::LearningRateConfig(float lr, float gx, float gy, float gr, float gv, float gt, float gc)
    : default_lr(lr), gain_x(gx), gain_y(gy), gain_r(gr), gain_v(gv), gain_theta(gt), gain_c(gc) {}

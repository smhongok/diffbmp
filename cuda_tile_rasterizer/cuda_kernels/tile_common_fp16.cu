#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "tile_common_fp16.h"

// Configuration struct constructors
__host__ __device__ TileConfigFP16::TileConfigFP16(int image_height, int image_width, int tile_size, __half sigma, __half alpha_upper_bound)
    : image_height(image_height), image_width(image_width), tile_size(tile_size), sigma(sigma), alpha_upper_bound(alpha_upper_bound) {
    total_tiles = ((image_width + tile_size - 1) / tile_size) * ((image_height + tile_size - 1) / tile_size);
    tile_width = (image_width + tile_size - 1) / tile_size;
    tile_height = (image_height + tile_size - 1) / tile_size;
}

__host__ __device__ PrimitiveConfigFP16::PrimitiveConfigFP16(int num_primitives, int num_templates, int template_height, int template_width, int max_prims_per_pixel)
    : num_primitives(num_primitives), num_templates(num_templates), template_height(template_height), template_width(template_width), max_prims_per_pixel(max_prims_per_pixel) {}

// Input tensor group constructors
__host__ __device__ InputTensorsFP16::InputTensorsFP16(
    const __half* means2D, const __half* radii, const __half* rotations,
    const __half* opacities, const __half* colors, const __half* primitive_templates,
    const int* global_bmp_sel)
    : means2D(means2D), radii(radii), rotations(rotations),
      opacities(opacities), colors(colors), primitive_templates(primitive_templates),
      global_bmp_sel(global_bmp_sel) {}

// Output tensor group constructors
__host__ __device__ OutputTensorsFP16::OutputTensorsFP16(
    __half* out_color, __half* out_alpha,
    __half* grad_means2D, __half* grad_radii, __half* grad_rotations,
    __half* grad_opacities, __half* grad_colors)
    : out_color(out_color), out_alpha(out_alpha),
      grad_means2D(grad_means2D), grad_radii(grad_radii), grad_rotations(grad_rotations),
      grad_opacities(grad_opacities), grad_colors(grad_colors) {}

// Global buffers constructors
__host__ __device__ GlobalBuffersFP16::GlobalBuffersFP16(
    __half* pixel_alphas, __half* pixel_colors_r, __half* pixel_colors_g, __half* pixel_colors_b,
    __half* pixel_T_values, int* pixel_prim_counts, __half* sigma_inv, __half* grad_sigma,
    int* d_tile_offsets, int* d_tile_indices, int tile_offsets_size, int tile_indices_size)
    : pixel_alphas(pixel_alphas), pixel_colors_r(pixel_colors_r), pixel_colors_g(pixel_colors_g),
      pixel_colors_b(pixel_colors_b), pixel_T_values(pixel_T_values), pixel_prim_counts(pixel_prim_counts),
      sigma_inv(sigma_inv), grad_sigma(grad_sigma),
      d_tile_offsets(d_tile_offsets), d_tile_indices(d_tile_indices), tile_offsets_size(tile_offsets_size), tile_indices_size(tile_indices_size) {}

__host__ __device__ LearningRateConfigFP16::LearningRateConfigFP16(__half lr, __half gx, __half gy, __half gr, __half gv, __half gt, __half gc)
    : default_lr(lr), gain_x(gx), gain_y(gy), gain_r(gr), gain_v(gv), gain_theta(gt), gain_c(gc) {}

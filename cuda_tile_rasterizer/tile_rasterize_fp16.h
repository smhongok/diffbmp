#ifndef TILE_RASTERIZE_FP16_H
#define TILE_RASTERIZE_FP16_H

#include <torch/extension.h>
#include <tuple>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// TileRasterizerFP16 class for managing global memory
class TileRasterizerFP16 {
private:
    // Global memory arrays (shared between forward and backward)
    __half* pixel_alphas;
    __half* pixel_colors_r;
    __half* pixel_colors_g;
    __half* pixel_colors_b;
    __half* pixel_T_values;    // Transmittance values for accurate gradient computation
    int* pixel_prim_counts;
    __half* sigma_inv;
    __half* grad_sigma;
    
    // Tile-related arrays for backward pass (allocated once and reused)
    int* d_tile_offsets;      // [num_tiles + 1] - device tile offsets
    int* d_tile_indices;      // [total_prims] - device tile indices
    int tile_offsets_size;    // Size of tile_offsets array
    int tile_indices_size;    // Size of tile_indices array
    
    // Gradient tensors and output tensors (allocated once and reused)
    __half* out_color;
    __half* out_alpha;
    __half* grad_means2D;
    __half* grad_radii;
    __half* grad_rotations;
    __half* grad_opacities;
    __half* grad_colors;
    
    // Track if memory is allocated
    bool memory_allocated;
    
public:
    // Configuration (public for access in wrapper functions)
    int max_prims_per_pixel;
    int image_height;
    int image_width;
    int tile_size;
    __half sigma;
    __half alpha_upper_bound;
    int num_primitives;
    
    // Constructor
    TileRasterizerFP16(int image_h, int image_w, int tile_sz, __half sig = __float2half(0.0f), __half alpha_ub = __float2half(1.0f), int max_prims = 16, int num_prims = 1024);
    
    // Destructor
    ~TileRasterizerFP16();
    
    // Forward pass
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor means2D,
        torch::Tensor radii,
        torch::Tensor rotations,
        torch::Tensor opacities,
        torch::Tensor colors,
        torch::Tensor colors_orig,
        torch::Tensor primitive_templates,
        torch::Tensor global_bmp_sel,
        float c_blend,
        torch::Tensor tile_primitive_mapping);
    
    // Backward pass
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> backward(
        torch::Tensor grad_out_color,
        torch::Tensor grad_out_alpha,
        torch::Tensor means2D,
        torch::Tensor radii,
        torch::Tensor rotations,
        torch::Tensor opacities,
        torch::Tensor colors,
        torch::Tensor colors_orig,
        torch::Tensor primitive_templates,
        torch::Tensor global_bmp_sel,
        float c_blend,
        torch::Tensor lr_config
    );
    
    // Allocate memory if not already allocated
    void allocateMemory();
    
    // Free memory
    void freeMemory();
};

// Global instance for Python binding (will be managed by Python)
extern std::shared_ptr<TileRasterizerFP16> global_tile_rasterizer_fp16;

// Forward declarations for CUDA functions (for backward compatibility)
std::tuple<torch::Tensor, torch::Tensor> CudaRasterizeTilesForwardFP16(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor colors_orig,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    float c_blend,
    torch::Tensor tile_primitive_mapping,
    int image_height,
    int image_width,
    int tile_size,
    float sigma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CudaRasterizeTilesBackwardFP16(
    torch::Tensor grad_out_color,
    torch::Tensor grad_out_alpha,
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor colors_orig,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,  // [num_primitives] - template selection indices
    float c_blend,
    torch::Tensor lr_config_tensor,
    int image_height,
    int image_width,
    int tile_size,
    float sigma);

#endif // TILE_RASTERIZE_FP16_H

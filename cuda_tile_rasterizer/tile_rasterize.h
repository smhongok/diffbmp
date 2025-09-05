#ifndef TILE_RASTERIZE_H
#define TILE_RASTERIZE_H

#include <torch/extension.h>
#include <tuple>
#include <memory>
#include <cuda_runtime.h>

// TileRasterizer class for managing global memory
class TileRasterizer {
private:
    // Global memory arrays (shared between forward and backward)
    float* pixel_alphas;
    float* pixel_colors_r;
    float* pixel_colors_g;
    float* pixel_colors_b;
    float* pixel_T_values;    // Transmittance values for accurate gradient computation
    int* pixel_prim_counts;
    float* sigma_inv;
    float* grad_sigma;
    
    // Tile-related arrays for backward pass (allocated once and reused)
    int* d_tile_offsets;      // [num_tiles + 1] - device tile offsets
    int* d_tile_indices;      // [total_prims] - device tile indices
    int tile_offsets_size;    // Size of tile_offsets array
    int tile_indices_size;    // Size of tile_indices array
    
    // Gradient tensors and output tensors (allocated once and reused)
    float* out_color;
    float* out_alpha;
    float* grad_means2D;
    float* grad_radii;
    float* grad_rotations;
    float* grad_opacities;
    float* grad_colors;
    
    // Track if memory is allocated
    bool memory_allocated;
    
public:
    // Configuration (public for access in wrapper functions)
    int max_prims_per_pixel;
    int image_height;
    int image_width;
    int tile_size;
    float sigma;
    float alpha_upper_bound;
    int num_primitives;
    
    // Constructor
    TileRasterizer(int image_h, int image_w, int tile_sz, float sig = 0.0f, float alpha_ub = 1.0f, int max_prims = 16, int num_prims = 1024);
    
    // Destructor
    ~TileRasterizer();
    
    // Forward pass
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor means2D,
        torch::Tensor radii,
        torch::Tensor rotations,
        torch::Tensor opacities,
        torch::Tensor colors,
        torch::Tensor primitive_templates,
        torch::Tensor global_bmp_sel,
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
        torch::Tensor primitive_templates,
        torch::Tensor global_bmp_sel,
        torch::Tensor lr_config
    );

    // Add this to the public section of TileRasterizer class
    torch::Tensor compute_per_pixel_gradients(
        torch::Tensor means2D,
        torch::Tensor radii,
        torch::Tensor rotations,
        torch::Tensor opacities,
        torch::Tensor colors,
        torch::Tensor primitive_templates,
        torch::Tensor global_bmp_sel,
        torch::Tensor target_image,
        int pixels_per_tile
    );

    
    // Allocate memory if not already allocated
    void allocateMemory();
    
    // Free memory
    void freeMemory();
};

// Global instance for Python binding (will be managed by Python)
extern std::shared_ptr<TileRasterizer> global_tile_rasterizer;

// Forward declarations for CUDA functions (for backward compatibility)
std::tuple<torch::Tensor, torch::Tensor> CudaRasterizeTilesForward(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    torch::Tensor tile_primitive_mapping,
    int image_height,
    int image_width,
    int tile_size,
    float sigma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CudaRasterizeTilesBackward(
    torch::Tensor grad_out_color,
    torch::Tensor grad_out_alpha,
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,  // [num_primitives] - template selection indices
    torch::Tensor lr_config_tensor,
    int image_height,
    int image_width,
    int tile_size,
    float sigma);

#endif // TILE_RASTERIZE_H
#include <torch/extension.h>
#include "tile_rasterize.h"

// Function to initialize global tile rasterizer
void init_tile_rasterizer(int image_height, int image_width, int tile_size, float sigma, float alpha_upper_bound, int max_prims_per_pixel, int num_primitives) {
    global_tile_rasterizer = std::make_shared<TileRasterizer>(
        image_height, image_width, tile_size, sigma, alpha_upper_bound, max_prims_per_pixel, num_primitives);
}

// Class-based forward function
std::tuple<torch::Tensor, torch::Tensor> rasterize_tiles_class(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    torch::Tensor tile_primitive_mapping) {
    
    if (!global_tile_rasterizer) {
        throw std::runtime_error("TileRasterizer not initialized. Call init_tile_rasterizer first.");
    }
    
    return global_tile_rasterizer->forward(means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, tile_primitive_mapping);
}

// Class-based backward function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_tiles_backward_class(
    torch::Tensor grad_out_color,
    torch::Tensor grad_out_alpha,
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    torch::Tensor lr_config_tensor) {
    
    if (!global_tile_rasterizer) {
        throw std::runtime_error("TileRasterizer not initialized or forward pass not called.");
    }
        
    return global_tile_rasterizer->backward(grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config_tensor);
}

std::tuple<torch::Tensor, torch::Tensor> rasterize_tiles(
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
    float sigma) {
    
    return CudaRasterizeTilesForward(means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, tile_primitive_mapping, image_height, image_width, tile_size, sigma);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_tiles_backward(
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
    float sigma) {
    
    return CudaRasterizeTilesBackward(grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates, global_bmp_sel, lr_config_tensor, image_height, image_width, tile_size, sigma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_tiles", &rasterize_tiles, "CUDA tile rasterization forward");
    m.def("rasterize_tiles_backward", &rasterize_tiles_backward, "CUDA tile rasterization backward");
    
    // Class-based functions
    m.def("init_tile_rasterizer", &init_tile_rasterizer, "Initialize global tile rasterizer");
    m.def("rasterize_tiles_class", &rasterize_tiles_class, "CUDA tile rasterization forward (class-based)");
    m.def("rasterize_tiles_backward_class", &rasterize_tiles_backward_class, "CUDA tile rasterization backward (class-based)");
    
    // Per-pixel gradient computation
    m.def("compute_per_pixel_gradients", [](
        torch::Tensor means2D,
        torch::Tensor radii,
        torch::Tensor rotations,
        torch::Tensor opacities,
        torch::Tensor colors,
        torch::Tensor primitive_templates,
        torch::Tensor global_bmp_sel,
        torch::Tensor target_image,
        int pixels_per_tile
    ) {
        if (!global_tile_rasterizer) {
            throw std::runtime_error("TileRasterizer not initialized. Call init_tile_rasterizer first.");
        }
        return global_tile_rasterizer->compute_per_pixel_gradients(
            means2D, radii, rotations, opacities, colors,
            primitive_templates, global_bmp_sel, target_image, pixels_per_tile
        );
    }, "Compute per-pixel gradient magnitudes using CUDA");
}
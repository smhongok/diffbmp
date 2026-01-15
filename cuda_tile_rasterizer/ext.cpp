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
    torch::Tensor colors_orig,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    float c_blend,
    torch::Tensor tile_primitive_mapping) {
    
    if (!global_tile_rasterizer) {
        throw std::runtime_error("TileRasterizer not initialized. Call init_tile_rasterizer first.");
    }
    
    return global_tile_rasterizer->forward(means2D, radii, rotations, opacities, colors, colors_orig, primitive_templates, global_bmp_sel, c_blend, tile_primitive_mapping);
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
    torch::Tensor colors_orig,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    float c_blend,
    torch::Tensor lr_config_tensor) {
    
    if (!global_tile_rasterizer) {
        throw std::runtime_error("TileRasterizer not initialized or forward pass not called.");
    }
        
    return global_tile_rasterizer->backward(grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, colors_orig, primitive_templates, global_bmp_sel, c_blend, lr_config_tensor);
}

// Python binding for forward_batch (class-based, globally managed memory)
std::tuple<torch::Tensor, torch::Tensor> rasterize_tiles_batch_class(
    torch::Tensor means2D,              // (B, N, 2)
    torch::Tensor radii,                // (B, N)
    torch::Tensor rotations,            // (B, N)
    torch::Tensor opacities,            // (B, N)
    torch::Tensor colors,               // (B, N, 3)
    torch::Tensor colors_orig,          // (B, N, H, W, 3)
    torch::Tensor primitive_templates,  // (P, H, W)
    torch::Tensor global_bmp_sel,       // (N,) - shared across batch
    float c_blend,
    py::object tile_primitive_mappings_obj) {  // Python list or None
    
    if (!global_tile_rasterizer) {
        throw std::runtime_error("TileRasterizer not initialized. Call init_tile_rasterizer first.");
    }
    
    // Convert Python list to C++ vector
    std::vector<torch::Tensor> tile_primitive_mappings;
    
    if (!tile_primitive_mappings_obj.is_none()) {
        if (py::isinstance<py::list>(tile_primitive_mappings_obj)) {
            py::list mapping_list = tile_primitive_mappings_obj.cast<py::list>();
            for (size_t i = 0; i < mapping_list.size(); i++) {
                tile_primitive_mappings.push_back(mapping_list[i].cast<torch::Tensor>());
            }
        } else {
            throw std::runtime_error("tile_primitive_mappings must be a list of tensors or None");
        }
    }
    
    return global_tile_rasterizer->forward_batch(means2D, radii, rotations, opacities, colors, colors_orig,
                                                  primitive_templates, global_bmp_sel, c_blend, tile_primitive_mappings);
}

// Python binding for backward_batch (class-based, globally managed memory)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_tiles_backward_batch_class(
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
) {
    if (!global_tile_rasterizer) {
        throw std::runtime_error("TileRasterizer not initialized. Call init_tile_rasterizer first.");
    }
    return global_tile_rasterizer->backward_batch(grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities,
                                 colors, colors_orig, primitive_templates, global_bmp_sel, c_blend, lr_config);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Class-based functions
    m.def("init_tile_rasterizer", &init_tile_rasterizer, "Initialize global tile rasterizer");
    m.def("rasterize_tiles_class", &rasterize_tiles_class, "CUDA tile rasterization forward (class-based)");
    m.def("rasterize_tiles_batch_class", &rasterize_tiles_batch_class, "CUDA tile rasterization forward batch (class-based)");
    m.def("rasterize_tiles_backward_class", &rasterize_tiles_backward_class, "CUDA tile rasterization backward (class-based)");
    m.def("rasterize_tiles_backward_batch_class", &rasterize_tiles_backward_batch_class, "CUDA tile rasterization backward batch (class-based)");
    
    // Timing functions
    m.def("print_cuda_timing_stats", &printCudaTimingStats, "Print CUDA timing statistics");
    m.def("reset_cuda_timing_stats", &resetCudaTimingStats, "Reset CUDA timing statistics");
}
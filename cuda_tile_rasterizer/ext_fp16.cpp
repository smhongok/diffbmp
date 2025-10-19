#include <torch/extension.h>
#include "tile_rasterize_fp16.h"

// Function to initialize global tile rasterizer
void init_tile_rasterizer_fp16(int image_height, int image_width, int tile_size, float sigma, float alpha_upper_bound, int max_prims_per_pixel, int num_primitives) {
    global_tile_rasterizer_fp16 = std::make_shared<TileRasterizerFP16>(
        image_height, image_width, tile_size, __float2half(sigma), __float2half(alpha_upper_bound), max_prims_per_pixel, num_primitives);
}

// Class-based forward function
std::tuple<torch::Tensor, torch::Tensor> rasterize_tiles_class_fp16(
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
    
    if (!global_tile_rasterizer_fp16) {
        throw std::runtime_error("TileRasterizerFP16 not initialized. Call init_tile_rasterizer_fp16 first.");
    }
    
    return global_tile_rasterizer_fp16->forward(means2D, radii, rotations, opacities, colors, colors_orig, primitive_templates, global_bmp_sel, c_blend, tile_primitive_mapping);
}

// Class-based backward function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_tiles_backward_class_fp16(
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
    
    if (!global_tile_rasterizer_fp16) {
        throw std::runtime_error("TileRasterizerFP16 not initialized or forward pass not called.");
    }
        
    return global_tile_rasterizer_fp16->backward(grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, colors_orig, primitive_templates, global_bmp_sel, c_blend, lr_config_tensor);
}

std::tuple<torch::Tensor, torch::Tensor> rasterize_tiles_fp16(
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
    float sigma) {
    
    return CudaRasterizeTilesForwardFP16(means2D, radii, rotations, opacities, colors, colors_orig, primitive_templates, global_bmp_sel, c_blend, tile_primitive_mapping, image_height, image_width, tile_size, sigma);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_tiles_backward_fp16(
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
    float sigma) {
    
    return CudaRasterizeTilesBackwardFP16(grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, colors_orig, primitive_templates, global_bmp_sel, c_blend, lr_config_tensor, image_height, image_width, tile_size, sigma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_tiles_fp16", &rasterize_tiles_fp16, "CUDA tile rasterization forward FP16");
    m.def("rasterize_tiles_backward_fp16", &rasterize_tiles_backward_fp16, "CUDA tile rasterization backward FP16");
    
    // Class-based functions
    m.def("init_tile_rasterizer_fp16", &init_tile_rasterizer_fp16, "Initialize global tile rasterizer FP16");
    m.def("rasterize_tiles_class_fp16", &rasterize_tiles_class_fp16, "CUDA tile rasterization forward FP16 (class-based)");
    m.def("rasterize_tiles_backward_class_fp16", &rasterize_tiles_backward_class_fp16, "CUDA tile rasterization backward FP16 (class-based)");
}

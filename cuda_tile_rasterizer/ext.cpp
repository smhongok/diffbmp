#include <torch/extension.h>
#include "tile_rasterize.h"

std::tuple<torch::Tensor, torch::Tensor> rasterize_tiles(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    int image_height,
    int image_width,
    int tile_size,
    float sigma) {
    
    return CudaRasterizeTilesForward(
        means2D, radii, rotations, opacities, colors, primitive_templates,
        image_height, image_width, tile_size, sigma);
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
    int image_height,
    int image_width,
    int tile_size,
    float sigma) {
    
    return CudaRasterizeTilesBackward(
        grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates,
        image_height, image_width, tile_size, sigma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_tiles", &rasterize_tiles, "CUDA tile rasterization forward");
    m.def("rasterize_tiles_backward", &rasterize_tiles_backward, "CUDA tile rasterization backward");
}
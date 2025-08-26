#include <torch/extension.h>
#include "tile_rasterize_fp16.h"

std::tuple<torch::Tensor, torch::Tensor> rasterize_tiles_fp16(
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
    
    // Convert to FP16 if not already
    if (means2D.dtype() != torch::kFloat16) {
        means2D = means2D.to(torch::kFloat16);
    }
    if (radii.dtype() != torch::kFloat16) {
        radii = radii.to(torch::kFloat16);
    }
    if (rotations.dtype() != torch::kFloat16) {
        rotations = rotations.to(torch::kFloat16);
    }
    if (opacities.dtype() != torch::kFloat16) {
        opacities = opacities.to(torch::kFloat16);
    }
    if (colors.dtype() != torch::kFloat16) {
        colors = colors.to(torch::kFloat16);
    }
    if (primitive_templates.dtype() != torch::kFloat16) {
        primitive_templates = primitive_templates.to(torch::kFloat16);
    }
    
    // Call the main FP16 function
    return CudaRasterizeTilesForwardFP16(
        means2D, radii, rotations, opacities, colors, primitive_templates,
        image_height, image_width, tile_size, sigma
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_tiles_backward_fp16(
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
    
    
    // Convert to FP16 if not already
    if (grad_out_color.dtype() != torch::kFloat16) {
        grad_out_color = grad_out_color.to(torch::kFloat16);
    }
    if (grad_out_alpha.dtype() != torch::kFloat16) {
        grad_out_alpha = grad_out_alpha.to(torch::kFloat16);
    }
    if (means2D.dtype() != torch::kFloat16) {
        means2D = means2D.to(torch::kFloat16);
    }
    if (radii.dtype() != torch::kFloat16) {
        radii = radii.to(torch::kFloat16);
    }
    if (rotations.dtype() != torch::kFloat16) {
        rotations = rotations.to(torch::kFloat16);
    }
    if (opacities.dtype() != torch::kFloat16) {
        opacities = opacities.to(torch::kFloat16);
    }
    if (colors.dtype() != torch::kFloat16) {
        colors = colors.to(torch::kFloat16);
    }
    if (primitive_templates.dtype() != torch::kFloat16) {
        primitive_templates = primitive_templates.to(torch::kFloat16);
    }
    
    // Call the main FP16 backward function
    return CudaRasterizeTilesBackwardFP16(
        grad_out_color, grad_out_alpha, means2D, radii, rotations, opacities, colors, primitive_templates,
        image_height, image_width, tile_size, sigma
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_tiles_fp16", &rasterize_tiles_fp16, "CUDA tile rasterization forward FP16");
    m.def("rasterize_tiles_backward_fp16", &rasterize_tiles_backward_fp16, "CUDA tile rasterization backward FP16");
}

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA kernel launchers
extern "C" {
    void launch_compute_bounding_boxes(
        const float* means2D,
        const float* radii,
        const float* rotations,
        const float* primitive_templates,
        const int* global_bmp_sel,
        int* output_bounding_boxes,
        int* cropped_sizes,
        int N, int H, int W, int template_height, int template_width,
        float scale_factor, float alpha_upper_bound
    );

    void launch_generate_cropped_layers(
        const float* means2D,
        const float* radii,
        const float* rotations,
        const float* colors,
        const float* visibility,
        const float* primitive_templates,
        const int* global_bmp_sel,
        const int* bounding_boxes,
        const int* layer_offsets,
        uint8_t* cropped_output_buffer,
        int N, int H, int W, int template_height, int template_width,
        float scale_factor, float alpha_upper_bound, float c_blend, const float* colors_orig,
        int max_bbox_w, int max_bbox_h
    );

}

// Python binding for bounding box computation
void compute_bounding_boxes(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    torch::Tensor output_bounding_boxes,
    torch::Tensor cropped_sizes,
    int H, int W,
    float scale_factor,
    float alpha_upper_bound
) {
    // Check tensor properties
    TORCH_CHECK(means2D.is_cuda(), "means2D must be a CUDA tensor");
    TORCH_CHECK(means2D.dtype() == torch::kFloat32, "means2D must be float32");
    
    int N = means2D.size(0);
    int template_height = primitive_templates.size(1);
    int template_width = primitive_templates.size(2);
    
    launch_compute_bounding_boxes(
        means2D.data_ptr<float>(),
        radii.data_ptr<float>(),
        rotations.data_ptr<float>(),
        primitive_templates.data_ptr<float>(),
        global_bmp_sel.data_ptr<int>(),
        output_bounding_boxes.data_ptr<int>(),
        cropped_sizes.data_ptr<int>(),
        N, H, W, template_height, template_width,
        scale_factor, alpha_upper_bound
    );
}


// Python binding for cropped layer generation
void generate_cropped_layers(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor colors,
    torch::Tensor visibility,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    torch::Tensor bounding_boxes,
    torch::Tensor layer_offsets,
    torch::Tensor cropped_output_buffer,
    int H, int W, float scale_factor, float alpha_upper_bound,
    float c_blend, torch::Tensor colors_orig
) {
    // Check tensor properties
    TORCH_CHECK(means2D.is_cuda(), "means2D must be a CUDA tensor");
    TORCH_CHECK(means2D.dtype() == torch::kFloat32, "means2D must be float32");
    
    int N = means2D.size(0);
    int template_height = primitive_templates.size(1);
    int template_width = primitive_templates.size(2);
    
    // Handle colors_orig tensor (can be null)
    const float* colors_orig_ptr = nullptr;
    if (colors_orig.defined() && colors_orig.numel() > 0) {
        TORCH_CHECK(colors_orig.is_cuda(), "colors_orig must be a CUDA tensor");
        TORCH_CHECK(colors_orig.dtype() == torch::kFloat32, "colors_orig must be float32");
        colors_orig_ptr = colors_orig.data_ptr<float>();
    }
    
    // Calculate maximum bounding box dimensions for grid sizing
    // This is a simple approach - could be optimized by calculating per-primitive
    int max_bbox_w = W;  // Conservative estimate
    int max_bbox_h = H;  // Conservative estimate
    
    launch_generate_cropped_layers(
        means2D.data_ptr<float>(),
        radii.data_ptr<float>(),
        rotations.data_ptr<float>(),
        colors.data_ptr<float>(),
        visibility.data_ptr<float>(),
        primitive_templates.data_ptr<float>(),
        global_bmp_sel.data_ptr<int>(),
        bounding_boxes.data_ptr<int>(),
        layer_offsets.data_ptr<int>(),
        cropped_output_buffer.data_ptr<uint8_t>(),
        N, H, W, template_height, template_width,
        scale_factor, alpha_upper_bound, c_blend, colors_orig_ptr,
        max_bbox_w, max_bbox_h
    );
}

// Module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_bounding_boxes", &compute_bounding_boxes, "Compute bounding boxes for primitives");
    m.def("generate_cropped_layers", &generate_cropped_layers, "Generate cropped layers for PSD export");
}

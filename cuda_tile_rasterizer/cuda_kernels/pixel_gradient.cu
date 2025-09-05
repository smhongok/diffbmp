#include "pixel_gradient.h"
#include "tile_forward.h"
#include "tile_backward.h"
#include "tile_common.h"

#define BLOCK_SIZE 256
#define MAX_PRIMITIVES_PER_TILE 512
#define EPS 1e-6f

// Analytic gradient computation for per-pixel gradients
// Based on tile_backward.cu logic but adapted for gradient magnitude calculation
__device__ void compute_per_pixel_analytic_gradient(
    int px, int py,
    const float* means2D,
    const float* radii,
    const float* rotations,
    const float* opacities,
    const float* colors,
    const float* primitive_templates,
    const int* global_bmp_sel,
    const float* target_pixel,  // [3] - target RGB at this pixel
    const int* tile_primitives, // List of primitive indices for this tile
    int num_tile_primitives,
    const TileConfig& tile_config,
    const PrimitiveConfig& prim_config,
    float* gradient_magnitudes  // Output: gradient magnitude per primitive
) {
    // Forward pass: render pixel and store intermediate values
    float final_color[3] = {0.0f, 0.0f, 0.0f};
    float T = 1.0f; // Transmittance
    
    // Store intermediate values for backward pass
    float alphas[MAX_PRIMITIVES_PER_TILE];
    float T_values[MAX_PRIMITIVES_PER_TILE];
    float colors_contrib[MAX_PRIMITIVES_PER_TILE * 3];
    int valid_primitives[MAX_PRIMITIVES_PER_TILE];
    int num_valid = 0;
    
    // Forward pass: compute intermediate values (same logic as tile_backward.cu)
    for (int k = 0; k < num_tile_primitives && k < MAX_PRIMITIVES_PER_TILE; k++) {
        int n = tile_primitives[k];
        if (n < 0 || n >= prim_config.num_primitives) continue;
        
        // Get primitive parameters
        const float mx = means2D[2*n + 0];
        const float my = means2D[2*n + 1];
        const float r = radii[n];
        const float phi = rotations[n];
        const float v_op = opacities[n];
        const float cr = colors[3*n + 0];
        const float cg = colors[3*n + 1];
        const float cb = colors[3*n + 2];
        
        // Compute relative position
        const float dx = (float)px - mx;
        const float dy = (float)py - my;
        
        // Template-aware bounding box check (same as tile_backward.cu)
        const int template_idx = global_bmp_sel[n];
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const float template_width = (float)prim_config.template_width;
            const float template_height = (float)prim_config.template_height;
            const float max_dim = fmaxf(template_width, template_height);
            const float scale_factor = r / (max_dim * 0.5f);
            
            const float half_width = template_width * 0.5f * scale_factor;
            const float half_height = template_height * 0.5f * scale_factor;
            
            if (fabsf(dx) > half_width || fabsf(dy) > half_height) {
                continue;  // pixel outside template bounds
            }
        } else {
            // Fallback to circular check for invalid template index
            if (dx*dx + dy*dy > r*r) {
                continue;
            }
        }
        
        // Transform to template coordinates (u, v)
        const float inv_r = (r > 1e-6f) ? (1.0f / r) : 1e6f;
        const float c = __cosf(phi);
        const float s = __sinf(phi);
        const float ndx = dx * inv_r;
        const float ndy = dy * inv_r;
        const float u = c * ndx + s * ndy;
        const float v = -s * ndx + c * ndy;
        
        // Sample mask from template
        float mask_val = 0.0f;
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const float tex_x = (u + 1.0f) * 0.5f * (prim_config.template_width - 1);
            const float tex_y = (v + 1.0f) * 0.5f * (prim_config.template_height - 1);
            
            const float* tex = &primitive_templates[
                template_idx * prim_config.template_height * prim_config.template_width];
            mask_val = bilinear_sample(tex, prim_config.template_height, prim_config.template_width, tex_y, tex_x);
            mask_val = fmaxf(0.0f, fminf(1.0f, mask_val));
        }
        
        if (mask_val <= 1e-6f) continue;
        
        // Compute alpha (same as tile_backward.cu)
        const float s_op = sigmoidf_safe(v_op);
        const float alpha = tile_config.alpha_upper_bound * s_op * mask_val;
        
        if (alpha <= 1e-6f) continue;
        
        // Store intermediate values
        alphas[num_valid] = alpha;
        T_values[num_valid] = T;
        colors_contrib[num_valid * 3 + 0] = sigmoidf_safe(cr);
        colors_contrib[num_valid * 3 + 1] = sigmoidf_safe(cg);
        colors_contrib[num_valid * 3 + 2] = sigmoidf_safe(cb);
        valid_primitives[num_valid] = n;
        
        // Update final color and transmittance (OVER compositing)
        final_color[0] += T * alpha * colors_contrib[num_valid * 3 + 0];
        final_color[1] += T * alpha * colors_contrib[num_valid * 3 + 1];
        final_color[2] += T * alpha * colors_contrib[num_valid * 3 + 2];
        T *= (1.0f - alpha);
        
        num_valid++;
        if (T < 1e-6f) break; // Early termination
    }
    
    // Compute loss gradient w.r.t. final color (MSE loss)
    const float gCx = 2.0f * (final_color[0] - target_pixel[0]);
    const float gCy = 2.0f * (final_color[1] - target_pixel[1]);
    const float gCz = 2.0f * (final_color[2] - target_pixel[2]);
    
    // Backward pass: compute gradients w.r.t. primitive positions (same logic as tile_backward.cu)
    for (int k = 0; k < num_valid; k++) {
        const int n = valid_primitives[k];
        const float a_k = alphas[k];
        const float T_k = T_values[k];
        const float cr = colors_contrib[k * 3 + 0];
        const float cg = colors_contrib[k * 3 + 1];
        const float cb = colors_contrib[k * 3 + 2];
        
        // Get primitive parameters
        const float mx = means2D[2*n + 0];
        const float my = means2D[2*n + 1];
        const float r = radii[n];
        const float phi = rotations[n];
        const float v_op = opacities[n];
        
        // Compute relative position
        const float dx = (float)px - mx;
        const float dy = (float)py - my;
        
        // Build suffix S and back-product B (exact same logic as tile_backward.cu)
        float Sx = 0.0f, Sy = 0.0f, Sz = 0.0f, B = 1.0f;
        for (int m = k + 1; m < num_valid; m++) {
            const float am = alphas[m];
            const float Tm = T_values[m];
            const float crm = colors_contrib[m * 3 + 0];
            const float cgm = colors_contrib[m * 3 + 1];
            const float cbm = colors_contrib[m * 3 + 2];
            Sx += crm * am * Tm;
            Sy += cgm * am * Tm;
            Sz += cbm * am * Tm;
            B *= fmaxf(1.0f - am, 0.0f);
        }
        
        // Compute dL/dalpha_k using OVER compositing formula (exact same as tile_backward.cu)
        const float inv1m = 1.0f / fmaxf(1.0f - a_k, EPS1);
        const float dCda_x = T_k * cr - Sx * inv1m;
        const float dCda_y = T_k * cg - Sy * inv1m;
        const float dCda_z = T_k * cb - Sz * inv1m;
        const float dLdalpha = gCx * dCda_x + gCy * dCda_y + gCz * dCda_z;
        
        // Transform coordinates for gradient computation (exact same as tile_backward.cu)
        const float inv_r = (r > 1e-6f) ? (1.0f / r) : 1e6f;
        const float c = __cosf(phi);
        const float s = __sinf(phi);
        const float ndx = dx * inv_r;
        const float ndy = dy * inv_r;
        const float u = c * ndx + s * ndy;
        const float v = -s * ndx + c * ndy;
        
        // Compute texture coordinates
        const float tex_x = (u + 1.0f) * 0.5f * (prim_config.template_width - 1);
        const float tex_y = (v + 1.0f) * 0.5f * (prim_config.template_height - 1);
        
        // CRITICAL: Get mask value and texture gradients using bilinear_value_and_grad_xy
        // This is the key difference from the simplified version - we compute EXACT texture gradients
        float mask_val = 0.0f;
        float dmask_dx_tex = 0.0f, dmask_dy_tex = 0.0f;
        const int template_idx = global_bmp_sel[n];
        if (template_idx >= 0 && template_idx < prim_config.num_templates) {
            const float* tex = &primitive_templates[
                template_idx * prim_config.template_height * prim_config.template_width];
            mask_val = bilinear_value_and_grad_xy(
                tex, prim_config.template_height, prim_config.template_width,
                tex_y, tex_x, dmask_dx_tex, dmask_dy_tex);
            mask_val = fmaxf(0.0f, fminf(1.0f, mask_val));
        } else {
            mask_val = 0.0f;
            dmask_dx_tex = 0.0f;
            dmask_dy_tex = 0.0f;
        }
        
        if (mask_val <= 1e-6f) continue;
        
        // Chain rule: dalpha/dmask * dmask/d(tex_coords) * d(tex_coords)/d(u,v) * d(u,v)/d(mx,my)
        // (exact same logic as tile_backward.cu)
        const float s_op = sigmoidf_safe(v_op);
        const float dalpha_dmask = tile_config.alpha_upper_bound * s_op;
        
        // Texture coordinate derivatives: tex_x = 0.5*(u+1)*(W-1), tex_y = 0.5*(v+1)*(H-1)
        const float du2x = 0.5f * (prim_config.template_width - 1);
        const float dv2y = 0.5f * (prim_config.template_height - 1);
        const float dmask_du = dmask_dx_tex * du2x;
        const float dmask_dv = dmask_dy_tex * dv2y;
        
        const float dL_dmask = dLdalpha * dalpha_dmask;
        const float dL_du = dL_dmask * dmask_du;
        const float dL_dv = dL_dmask * dmask_dv;
        
        // Transform derivatives: u = c*ndx + s*ndy, v = -s*ndx + c*ndy
        // where ndx = dx/r, ndy = dy/r, dx = px - mx, dy = py - my
        // (exact same logic as tile_backward.cu)
        const float du_dmx = -c * inv_r;  // du/d(mx) = -du/d(dx) = -c/r
        const float du_dmy = -s * inv_r;  // du/d(my) = -du/d(dy) = -s/r
        const float dv_dmx = s * inv_r;   // dv/d(mx) = -dv/d(dx) = s/r
        const float dv_dmy = -c * inv_r;  // dv/d(my) = -dv/d(dy) = -c/r
        
        // Final position gradients (exact same as tile_backward.cu)
        const float grad_mx = dL_du * du_dmx + dL_dv * dv_dmx;
        const float grad_my = dL_du * du_dmy + dL_dv * dv_dmy;
        
        // Store gradient magnitude for this primitive
        const float grad_magnitude = sqrtf(grad_mx * grad_mx + grad_my * grad_my);
        atomicAdd(&gradient_magnitudes[n], grad_magnitude);
    }
}

// CUDA kernel for computing per-pixel gradients
__global__ void compute_per_pixel_gradient_kernel(
    const float* means2D,
    const float* radii,
    const float* rotations,
    const float* opacities,
    const float* colors,
    const float* primitive_templates,
    const int* global_bmp_sel,
    const float* target_image,
    const int* tile_offsets,
    const int* tile_indices,
    TileConfig tile_config,
    PrimitiveConfig prim_config,
    int pixels_per_tile,
    float* gradient_magnitudes
) {
    // Calculate tile indices
    const int tile_rows = (tile_config.image_height + tile_config.tile_size - 1) / tile_config.tile_size;
    const int tile_cols = (tile_config.image_width + tile_config.tile_size - 1) / tile_config.tile_size;
    const int total_tiles = tile_rows * tile_cols;
    
    // Each block processes one tile
    const int tile_idx = blockIdx.x;
    if (tile_idx >= total_tiles) return;
    
    // Calculate tile boundaries
    const int tile_row = tile_idx / tile_cols;
    const int tile_col = tile_idx % tile_cols;
    const int y_min = tile_row * tile_config.tile_size;
    const int y_max = min(y_min + tile_config.tile_size, tile_config.image_height);
    const int x_min = tile_col * tile_config.tile_size;
    const int x_max = min(x_min + tile_config.tile_size, tile_config.image_width);
    
    // Get primitives for this tile
    const int prim_start = tile_offsets[tile_idx];
    const int prim_end = tile_offsets[tile_idx + 1];
    const int num_tile_primitives = prim_end - prim_start;
    
    if (num_tile_primitives <= 0) return;
    
    // Shared memory for tile primitives
    __shared__ int tile_primitives[MAX_PRIMITIVES_PER_TILE];
    
    // Load tile primitives into shared memory
    for (int i = threadIdx.x; i < min(num_tile_primitives, MAX_PRIMITIVES_PER_TILE); i += blockDim.x) {
        tile_primitives[i] = tile_indices[prim_start + i];
    }
    __syncthreads();
    
    // Each thread processes multiple pixels within the tile
    const int thread_id = threadIdx.x;
    const int total_threads = blockDim.x;
    
    // Calculate total pixels in this tile
    const int tile_width = x_max - x_min;
    const int tile_height = y_max - y_min;
    const int total_tile_pixels = tile_width * tile_height;
    
    // Sample pixels within this tile
    const int actual_pixels_per_tile = min(pixels_per_tile, total_tile_pixels);
    
    // Initialize random state for this thread
    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state;
    curand_init(global_thread_id, 0, 0, &local_state);
    
    // Process sampled pixels
    for (int pixel_idx = thread_id; pixel_idx < actual_pixels_per_tile; pixel_idx += total_threads) {
        int px, py;
        
        if (actual_pixels_per_tile < total_tile_pixels) {
            // Random sampling within tile bounds
            const int random_offset = curand(&local_state) % total_tile_pixels;
            const int local_y = random_offset / tile_width;
            const int local_x = random_offset % tile_width;
            py = y_min + local_y;
            px = x_min + local_x;
        } else {
            // Use all pixels within actual tile bounds
            const int local_y = pixel_idx / tile_width;
            const int local_x = pixel_idx % tile_width;
            py = y_min + local_y;
            px = x_min + local_x;
        }
        
        // Final bounds check
        if (px >= tile_config.image_width || py >= tile_config.image_height) continue;
        
        // Get target pixel value
        const int pixel_offset = (py * tile_config.image_width + px) * 3;
        const float target_pixel[3] = {
            target_image[pixel_offset + 0],
            target_image[pixel_offset + 1], 
            target_image[pixel_offset + 2]
        };
        
        // Compute analytic gradients for this pixel
        compute_per_pixel_analytic_gradient(
            px, py, means2D, radii, rotations, opacities, colors,
            primitive_templates, global_bmp_sel, target_pixel, tile_primitives,
            min(num_tile_primitives, MAX_PRIMITIVES_PER_TILE), tile_config, prim_config,
            gradient_magnitudes
        );
    }
}

// Host function to launch the CUDA kernel
torch::Tensor compute_per_pixel_gradient_cuda(
    torch::Tensor means2D,
    torch::Tensor radii,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor primitive_templates,
    torch::Tensor global_bmp_sel,
    torch::Tensor target_image,
    torch::Tensor tile_offsets,
    torch::Tensor tile_indices,
    int tile_size,
    float sigma,
    float alpha_upper_bound,
    int max_prims_per_pixel,
    int pixels_per_tile
) {
    const int N = means2D.size(0);
    const int H = target_image.size(0);
    const int W = target_image.size(1);
    
    // Create output tensor for gradient magnitudes
    torch::Tensor gradient_magnitudes = torch::zeros({N}, means2D.options());
    
    // Setup configuration structs
    TileConfig tile_config;
    tile_config.image_height = H;
    tile_config.image_width = W;
    tile_config.tile_size = tile_size;
    tile_config.sigma = sigma;
    tile_config.alpha_upper_bound = alpha_upper_bound;
    
    PrimitiveConfig prim_config;
    prim_config.num_primitives = N;
    prim_config.max_prims_per_pixel = max_prims_per_pixel;
    prim_config.num_templates = primitive_templates.size(0);
    prim_config.template_height = primitive_templates.size(1);
    prim_config.template_width = primitive_templates.size(2);
    
    // Calculate grid and block dimensions
    const int tile_rows = (H + tile_size - 1) / tile_size;
    const int tile_cols = (W + tile_size - 1) / tile_size;
    const int total_tiles = tile_rows * tile_cols;
    
    dim3 grid(total_tiles);
    dim3 block(BLOCK_SIZE);
    
    // Launch kernel
    compute_per_pixel_gradient_kernel<<<grid, block>>>(
        means2D.data_ptr<float>(),
        radii.data_ptr<float>(),
        rotations.data_ptr<float>(),
        opacities.data_ptr<float>(),
        colors.data_ptr<float>(),
        primitive_templates.data_ptr<float>(),
        global_bmp_sel.data_ptr<int>(),
        target_image.data_ptr<float>(),
        tile_offsets.data_ptr<int>(),
        tile_indices.data_ptr<int>(),
        tile_config,
        prim_config,
        pixels_per_tile,
        gradient_magnitudes.data_ptr<float>()
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in compute_per_pixel_gradient_cuda: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA kernel launch failed");
    }
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    return gradient_magnitudes;
}

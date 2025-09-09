# PSD Export CUDA Kernel Implementation Plan

## 📋 Overview

This document outlines the implementation plan for a new FP16 CUDA kernel specifically designed for efficient PSD layer export in the circle_art project. The goal is to replace the current inefficient PyTorch-based approach with a highly optimized primitive-parallel CUDA implementation.

## 🎯 Motivation and Goals

### Current Problem
- **Performance Bottleneck**: Current PSD export uses PyTorch `F.grid_sample()` in Python loops, causing significant performance degradation
- **Memory Inefficiency**: Individual primitive rendering creates unnecessary intermediate tensors
- **Type Mismatch Issues**: Float16/Float32 compatibility problems between CUDA kernels and PyTorch operations
- **Sequential Processing**: Current approach processes primitives one by one instead of leveraging GPU parallelism

### Target Objectives
1. **Performance**: 10-50x speedup for PSD export operations
2. **Memory Efficiency**: 50% memory reduction using FP16 precision
3. **Scalability**: Support for 2000+ primitives on high-resolution canvases (2048x2048+)
4. **Accuracy**: Maintain pixel-perfect consistency with existing renderer output
5. **Integration**: Seamless integration with existing codebase without breaking changes

## 🏗️ Architecture Design

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PSD Export    │    │  New CUDA Kernel │    │  Final PNG      │
│   (New Kernel)  │    │  (Primitive-     │    │  (Existing      │
│                 │    │   Parallel)      │    │   Renderer)     │
│  Individual     │    │                  │    │  Porter-Duff    │
│  Layers         │    │  No Compositing  │    │  Compositing    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Kernel Design Philosophy
- **Primitive-Parallel**: Each primitive processed by independent thread blocks
- **2D Thread Grid**: Pixel-level parallelism within each primitive's bounding box
- **Forward-Only**: No backward pass needed (export-only operation)
- **Direct Output**: Generate [H,W,4] RGBA tensors directly

## 🔧 Technical Implementation

### 1. Kernel Architecture

#### Main Kernel Structure
```cpp
__global__ void psd_export_kernel_fp16(
    // Input parameters
    const __half* means2D,           // [N, 2] - primitive positions
    const __half* radii,             // [N] - primitive scales
    const __half* rotations,         // [N] - primitive rotations  
    const __half* primitive_templates, // [P, H_t, W_t] - unblurred templates
    const int* global_bmp_sel,       // [N] - template selection indices
    const int* bounding_boxes,       // [N, 4] - precomputed bounding boxes
    
    // Output
    __half* output_layers,           // [N, H, W, 4] - RGBA layer data
    
    // Configuration
    PSDExportConfigFP16 config
);
```

#### Launch Configuration
```cpp
// Per-primitive processing
for (int prim_id = 0; prim_id < N_primitives; prim_id++) {
    // Get bounding box for this primitive
    int bbox_w = bounding_boxes[prim_id * 4 + 2] - bounding_boxes[prim_id * 4 + 0];
    int bbox_h = bounding_boxes[prim_id * 4 + 3] - bounding_boxes[prim_id * 4 + 1];
    
    // Calculate grid dimensions
    dim3 grid((bbox_w + 15) / 16, (bbox_h + 15) / 16);
    dim3 block(16, 16);  // 256 threads per block
    
    // Launch kernel for this primitive
    psd_export_kernel_fp16<<<grid, block>>>(
        prim_id, means2D, radii, rotations, primitive_templates,
        global_bmp_sel, bounding_boxes, output_layers, config
    );
}
```

### 2. Core Components

#### A. Bounding Box Computation
```cpp
__global__ void compute_bounding_boxes_fp16(
    const __half* means2D, const __half* radii,
    int* bounding_boxes, PSDExportConfigFP16 config
) {
    int prim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (prim_id >= config.N) return;
    
    __half x = __hmul(means2D[prim_id * 2 + 0], config.scale_factor);
    __half y = __hmul(means2D[prim_id * 2 + 1], config.scale_factor);
    __half r = __hmul(radii[prim_id], config.scale_factor);
    
    // Conservative bounding box (can be optimized with rotation)
    int min_x = max(0, __half2int_rd(__hsub(x, r)));
    int min_y = max(0, __half2int_rd(__hsub(y, r)));
    int max_x = min(config.W, __half2int_ru(__hadd(x, r)) + 1);
    int max_y = min(config.H, __half2int_ru(__hadd(y, r)) + 1);
    
    bounding_boxes[prim_id * 4 + 0] = min_x;
    bounding_boxes[prim_id * 4 + 1] = min_y;
    bounding_boxes[prim_id * 4 + 2] = max_x;
    bounding_boxes[prim_id * 4 + 3] = max_y;
}
```

#### B. FP16 Grid Sampling
```cpp
__device__ __half4 cuda_grid_sample_rgba_fp16(
    const __half* template_data, int H_t, int W_t,
    int px, int py, __half x, __half y, __half r, __half theta
) {
    // Normalize coordinates to [-1, 1]
    __half u_norm = __hdiv(__hsub(__int2half_rn(px), x), r);
    __half v_norm = __hdiv(__hsub(__int2half_rn(py), y), r);
    
    // Apply inverse rotation
    __half cos_t = hcos(theta);
    __half sin_t = hsin(theta);
    __half u_rot = __hadd(__hmul(cos_t, u_norm), __hmul(sin_t, v_norm));
    __half v_rot = __hsub(__hmul(__hneg(sin_t), u_norm), __hmul(cos_t, v_norm));
    
    // Convert to template coordinates [0, W_t-1] x [0, H_t-1]
    __half u_template = __hmul(__hadd(u_rot, __float2half(1.0f)), 
                              __hmul(__float2half(0.5f), __int2half_rn(W_t - 1)));
    __half v_template = __hmul(__hadd(v_rot, __float2half(1.0f)), 
                              __hmul(__float2half(0.5f), __int2half_rn(H_t - 1)));
    
    // Bilinear interpolation
    return bilinear_sample_fp16(template_data, H_t, W_t, u_template, v_template);
}

__device__ __half4 bilinear_sample_fp16(
    const __half* data, int H, int W, __half u, __half v
) {
    // Bounds check
    if (__hlt(u, __float2half(0.0f)) || __hge(u, __int2half_rn(W-1)) ||
        __hlt(v, __float2half(0.0f)) || __hge(v, __int2half_rn(H-1))) {
        return make_half4(__float2half(0.0f), __float2half(0.0f), 
                         __float2half(0.0f), __float2half(0.0f));
    }
    
    // Integer coordinates
    int u0 = __half2int_rd(u), u1 = u0 + 1;
    int v0 = __half2int_rd(v), v1 = v0 + 1;
    
    // Interpolation weights
    __half wu = __hsub(u, __int2half_rn(u0));
    __half wv = __hsub(v, __int2half_rn(v0));
    __half wu_inv = __hsub(__float2half(1.0f), wu);
    __half wv_inv = __hsub(__float2half(1.0f), wv);
    
    // Sample values (assuming grayscale template -> RGBA)
    __half val00 = data[v0 * W + u0];
    __half val01 = data[v0 * W + u1];
    __half val10 = data[v1 * W + u0];
    __half val11 = data[v1 * W + u1];
    
    // Bilinear interpolation
    __half result = __hadd(__hadd(__hmul(__hmul(wu_inv, wv_inv), val00),
                                 __hmul(__hmul(wu, wv_inv), val01)),
                          __hadd(__hmul(__hmul(wu_inv, wv), val10),
                                __hmul(__hmul(wu, wv), val11)));
    
    // Convert grayscale to RGBA (R=G=B=alpha, A=alpha)
    return make_half4(result, result, result, result);
}
```

### 3. File Structure

```
cuda_tile_rasterizer/
├── cuda_psd_export_fp16/
│   ├── psd_export_fp16.h           # Header definitions
│   ├── psd_export_fp16.cu          # Main kernel implementation
│   ├── psd_kernels_fp16.cu         # Helper kernels (bbox, grid_sample)
│   └── psd_common_fp16.h           # Common structures and utilities
├── ext_psd_fp16.cpp                # Python binding
├── setup_psd_fp16.py               # Build configuration
└── __init__.py                     # Python interface
```

### 4. Python Integration

#### Python Wrapper
```python
# cuda_tile_rasterizer/psd_export_fp16.py
import torch
from . import _C_psd_fp16

def export_psd_layers_cuda_fp16(
    renderer_S: torch.Tensor,        # [P, H_t, W_t] primitive templates
    x: torch.Tensor,                 # [N] x positions
    y: torch.Tensor,                 # [N] y positions  
    r: torch.Tensor,                 # [N] scales
    theta: torch.Tensor,             # [N] rotations
    v: torch.Tensor,                 # [N] visibility (unused in export)
    c: torch.Tensor,                 # [N, 3] colors (unused in export)
    H: int, W: int,                  # Canvas dimensions
    scale_factor: float = 1.0        # Export scaling
) -> List[torch.Tensor]:
    """
    Export individual primitive layers using FP16 CUDA kernel.
    
    Returns:
        List of [H, W, 4] RGBA tensors, one per primitive
    """
    N = len(x)
    device = x.device
    
    # Convert inputs to FP16
    x_fp16 = x.half()
    y_fp16 = y.half()
    r_fp16 = r.half()
    theta_fp16 = theta.half()
    renderer_S_fp16 = renderer_S.half()
    
    # Allocate output tensor [N, H, W, 4]
    output_layers = torch.zeros(N, H, W, 4, dtype=torch.float16, device=device)
    
    # Call CUDA kernel
    _C_psd_fp16.export_layers(
        x_fp16, y_fp16, r_fp16, theta_fp16, renderer_S_fp16,
        output_layers, H, W, scale_factor
    )
    
    # Convert to list of individual layers
    return [output_layers[i] for i in range(N)]
```

#### Integration with PSDExporter
```python
# util/psd_exporter.py
def add_layers_batch_optimized(self, primitive_templates, x_tensor, y_tensor, 
                              r_tensor, theta_tensor, v_tensor, c_tensor):
    """Replace current implementation with CUDA kernel call."""
    
    # Use new FP16 CUDA kernel
    from cuda_tile_rasterizer.psd_export_fp16 import export_psd_layers_cuda_fp16
    
    transformed_layers = export_psd_layers_cuda_fp16(
        primitive_templates, x_tensor, y_tensor, r_tensor, theta_tensor,
        v_tensor, c_tensor, self.export_height, self.export_width, 
        self.scale_factor
    )
    
    # Convert to PIL images and add to PSD
    for i, layer_tensor in enumerate(transformed_layers):
        # Convert [H, W, 4] tensor to PIL Image
        layer_np = layer_tensor.detach().cpu().numpy()
        layer_np = (layer_np * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(layer_np, 'RGBA')
        self.add_layer(pil_image, f"primitive_{i}")
```

## 📊 Performance Analysis

### Expected Performance Gains

#### Memory Usage Comparison
```
Current (PyTorch F.grid_sample):
- Per-primitive processing: O(N × H × W × 4 × 4 bytes) = 16N×H×W bytes
- Intermediate tensors: Additional 2-3x overhead

New (FP16 CUDA):
- Direct output: O(N × H × W × 4 × 2 bytes) = 8N×H×W bytes  
- No intermediate tensors: 50% base reduction + no overhead
```

#### Speed Comparison
```
Test Case: 2000 primitives, 2048×2048 canvas

Current Implementation:
- PyTorch grid_sample: ~2000 individual calls
- CPU-GPU transfers: Multiple round trips
- Estimated time: 30-60 seconds

New Implementation:  
- Single CUDA kernel launch per primitive
- Pure GPU processing: No CPU-GPU transfers
- Estimated time: 1-3 seconds (10-50x speedup)
```

### Scalability Analysis
```
Memory Requirements (FP16):
- 1000 primitives × 1024² × 4 channels × 2 bytes = ~8.6 GB
- 2000 primitives × 2048² × 4 channels × 2 bytes = ~67 GB
- 4000 primitives × 2048² × 4 channels × 2 bytes = ~134 GB

GPU Memory Limits:
- RTX 3090: 24 GB → ~700 primitives @ 2048²
- RTX 4090: 24 GB → ~700 primitives @ 2048²  
- A100: 80 GB → ~2400 primitives @ 2048²
```

## 🛠️ Implementation Timeline

### Phase 1: Core Kernel Development (Week 1-2)
- [ ] Implement basic FP16 kernel structure
- [ ] Create bounding box computation kernel
- [ ] Implement FP16 bilinear interpolation
- [ ] Basic grid sampling functionality
- [ ] Unit tests for individual components

### Phase 2: Integration & Optimization (Week 3)
- [ ] Python binding implementation
- [ ] Integration with existing PSDExporter
- [ ] Performance benchmarking
- [ ] Memory usage optimization
- [ ] Error handling and edge cases

### Phase 3: Testing & Validation (Week 4)
- [ ] Pixel-perfect accuracy validation
- [ ] Large-scale performance testing
- [ ] Memory leak detection
- [ ] Cross-platform compatibility
- [ ] Documentation and code review

### Phase 4: Deployment (Week 5)
- [ ] Final integration testing
- [ ] Performance regression testing
- [ ] User acceptance testing
- [ ] Production deployment
- [ ] Monitoring and optimization

## 🧪 Testing Strategy

### Unit Tests
1. **Bounding Box Accuracy**: Verify correct bbox computation for various primitive configurations
2. **Grid Sampling Precision**: Compare FP16 results with FP32 reference implementation
3. **Memory Management**: Test for memory leaks and proper cleanup
4. **Edge Cases**: Handle out-of-bounds coordinates, zero radii, extreme rotations

### Integration Tests  
1. **PSD Compatibility**: Ensure exported PSD files are valid and loadable
2. **Visual Accuracy**: Pixel-perfect comparison with current implementation
3. **Performance Benchmarks**: Measure speedup across different configurations
4. **Memory Usage**: Validate 50% memory reduction claims

### Stress Tests
1. **Large Scale**: Test with 4000+ primitives on maximum resolution
2. **Memory Limits**: Test behavior near GPU memory limits
3. **Concurrent Access**: Multiple simultaneous export operations
4. **Error Recovery**: Graceful handling of CUDA errors and fallbacks

## 🔄 Backward Compatibility

### Fallback Strategy
```python
def add_layers_batch_optimized(self, ...):
    try:
        # Try new FP16 CUDA kernel
        return self._cuda_fp16_export(...)
    except (ImportError, RuntimeError) as e:
        # Fallback to current PyTorch implementation
        logger.warning(f"CUDA FP16 export failed: {e}, using fallback")
        return self._pytorch_export(...)
```

### Configuration Options
```python
# config.json
{
    "postprocessing": {
        "export_psd": true,
        "psd_export_method": "cuda_fp16",  # "cuda_fp16" | "pytorch" | "auto"
        "psd_scale_factor": 1.0,
        "psd_memory_limit_gb": 20.0
    }
}
```

## 📈 Success Metrics

### Performance Targets
- [ ] **Speed**: 10x minimum speedup for typical workloads (1000 primitives, 1024²)
- [ ] **Memory**: 50% memory reduction compared to current implementation
- [ ] **Scalability**: Support 2000+ primitives on 2048² canvas with 24GB GPU
- [ ] **Accuracy**: Pixel-perfect match with current renderer output

### Quality Targets
- [ ] **Reliability**: Zero memory leaks, proper error handling
- [ ] **Compatibility**: Works across CUDA 11.0+ and major GPU architectures
- [ ] **Maintainability**: Clean, documented code following project conventions
- [ ] **Usability**: Seamless integration, no breaking changes to existing API

## 🚀 Future Enhancements

### Potential Optimizations
1. **Rotation-Aware Bounding Boxes**: Tighter bounding boxes considering rotation
2. **Template Caching**: GPU-side template caching for repeated primitives
3. **Multi-GPU Support**: Distribute primitives across multiple GPUs
4. **Streaming**: Process primitives in batches to handle memory-limited scenarios
5. **Compression**: On-the-fly compression for large layer outputs

### Extended Features
1. **Real-time Preview**: Live PSD layer updates during optimization
2. **Selective Export**: Export only modified layers for incremental updates
3. **Format Support**: Additional export formats (TIFF, OpenEXR, etc.)
4. **Quality Modes**: Speed vs. quality trade-offs for different use cases

---

## 📝 Notes

This implementation plan prioritizes performance, memory efficiency, and maintainability while ensuring seamless integration with the existing codebase. The FP16 approach leverages existing infrastructure and provides significant benefits for large-scale primitive processing.

The modular design allows for incremental development and testing, with clear fallback mechanisms to ensure system stability during the transition period.

#ifndef TILE_BACKWARD_FP16_H
#define TILE_BACKWARD_FP16_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tile_common_fp16.h"

// FP16 version of the backward CUDA kernel
void CudaRasterizeTilesBackwardKernelFP16(
    const __half* grad_out_color,
    const __half* grad_out_alpha,
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config,
    const LearningRateConfigFP16 lr_config);

#endif // TILE_BACKWARD_FP16_H

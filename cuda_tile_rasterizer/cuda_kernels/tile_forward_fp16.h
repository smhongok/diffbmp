#ifndef TILE_FORWARD_FP16_H
#define TILE_FORWARD_FP16_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tile_common_fp16.h"

// FP16 version of the CUDA kernel
void CudaRasterizeTilesForwardKernelFP16(
    const InputTensorsFP16 inputs,
    const OutputTensorsFP16 outputs,
    const GlobalBuffersFP16 buffers,
    const TileConfigFP16 tile_config,
    const PrimitiveConfigFP16 prim_config);

#endif // TILE_FORWARD_FP16_H

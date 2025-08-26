#ifndef TILE_BACKWARD_H
#define TILE_BACKWARD_H

#include <cuda_runtime.h>
#include "tile_common.h"

void CudaRasterizeTilesBackwardKernel(
    const float* grad_out_color,
    const float* grad_out_alpha,
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const TileConfig tile_config,
    const PrimitiveConfig prim_config,
    const LearningRateConfig lr_config);

#endif // TILE_BACKWARD_H
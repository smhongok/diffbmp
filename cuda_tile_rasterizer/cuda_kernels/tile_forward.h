#ifndef TILE_FORWARD_H
#define TILE_FORWARD_H

#include <cuda_runtime.h>
#include "tile_common.h"

void CudaRasterizeTilesForwardKernel(
    const InputTensors inputs,
    const OutputTensors outputs,
    const GlobalBuffers buffers,
    const TileConfig tile_config,
    const PrimitiveConfig prim_config);

#endif // TILE_FORWARD_H
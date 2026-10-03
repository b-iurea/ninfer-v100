#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Converts a contiguous BF16 activation to FP16 once, ahead of a Volta QPN projection. The QPN
// kernels otherwise convert inside every CTA and warp, repeating the same per-element conversion
// for each output tile. The conversion is the one those kernels perform in place, so routing
// through this staging copy does not change results.
void stage_bf16_as_fp16_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream);

} // namespace ninfer::ops::detail

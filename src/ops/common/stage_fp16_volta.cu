#include "ops/common/stage_fp16_volta.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__global__ void stage_bf16_as_fp16_kernel(const __nv_bfloat16* __restrict__ input,
                                          half* __restrict__ output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = __float2half(__bfloat162float(input[i])); }
}

} // namespace

void stage_bf16_as_fp16_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream) {
    const std::int64_t count = x.numel();
    stage_bf16_as_fp16_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<half*>(fp16_out), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

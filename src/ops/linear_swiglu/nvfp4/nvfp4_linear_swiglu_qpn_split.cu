#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/stage_fp16_volta.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_qpn_split.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
constexpr std::int32_t kIntermediate = 17408; // Nvfp4MlpGateUpGeometry::kOutputRows / 2
constexpr std::int32_t kMTileOffset  = kIntermediate / 128; // 136, exact
} // namespace

bool nvfp4_linear_swiglu_qpn_split_supported(std::int32_t k, std::int32_t t) noexcept {
    return nvfp4_volta_qpn_supported(kIntermediate, k, t);
}

void nvfp4_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          float* gate_scratch, float* up_scratch,
                                          void* activation_scratch,
                                          cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    auto* x_fp16 = static_cast<half*>(activation_scratch);
    stage_bf16_as_fp16_sm70(x, x_fp16, stream);

    Weight gate_weight = weight;
    gate_weight.n      = kIntermediate;

    Weight up_weight = weight;
    up_weight.n       = kIntermediate;
    up_weight.qdata   = static_cast<const std::uint8_t*>(weight.qdata) +
                       static_cast<std::int64_t>(kIntermediate) * (k / 2);
    up_weight.scales = static_cast<const std::uint8_t*>(weight.scales) +
                       static_cast<std::int64_t>(kMTileOffset) * (k / 64) * 512;

    launch_nvfp4_volta_qpn_with_fp16_activation(
        x, gate_weight, x_fp16, Nvfp4Fp32ContiguousOutput{gate_scratch, kIntermediate},
        kIntermediate, inverse_weight_divisor, stream);
    launch_nvfp4_volta_qpn_with_fp16_activation(
        x, up_weight, x_fp16, Nvfp4Fp32ContiguousOutput{up_scratch, kIntermediate},
        kIntermediate, inverse_weight_divisor, stream);

    const std::int64_t elements = static_cast<std::int64_t>(kIntermediate) * t;
    const int threads           = 256;
    const int blocks = static_cast<int>(std::min<std::int64_t>((elements + threads - 1) / threads, 4096));
    nvfp4_swiglu_fp32_combine_kernel<<<blocks, threads, 0, stream>>>(
        gate_scratch, up_scratch, static_cast<__nv_bfloat16*>(out.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail

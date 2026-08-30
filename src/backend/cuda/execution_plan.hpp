#pragma once

#include "celeg/backend/cuda/runtime_types.hpp"

#include <string>

namespace celeg {

enum class LinearKernelKind : uint8_t {
    Bf16Cublas,
    Bf16CublasLt,
    W8A16,
    W4A16,
    Q4kMmq,
    Q6kMmq,
    MixedBf16AndGgufMmq,
    Fp8W8A8,
    Nvfp4W4A4,
};

class CudaExecutionPlan {
public:
    static CudaExecutionPlan compile(
        CudaModelOptions requested, int max_context,
        CudaDeviceCapabilities device = {});
    static uint64_t compile_count();

    const CudaModelOptions& options() const { return options_; }
    LinearKernelKind linear_kernel() const { return linear_kernel_; }
    bool segmented_attention(int position) const;
    bool segmented_capable() const {
        return options_.attention_mode != AttentionMode::Single;
    }
    int attention_chunks() const { return attention_chunks_; }
    int max_context() const { return max_context_; }
    std::string description() const;
    const CudaDeviceCapabilities& device() const { return device_; }
    bool mmq_tensor_cores_enabled() const {
        return device_.mmq_tensor_core_enabled;
    }
    uint64_t fingerprint() const { return fingerprint_; }

private:
    CudaModelOptions options_{};
    CudaDeviceCapabilities device_{};
    LinearKernelKind linear_kernel_ = LinearKernelKind::Bf16Cublas;
    int attention_chunks_ = 0;
    int max_context_ = 0;
    uint64_t fingerprint_ = 0;
};

}

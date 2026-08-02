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
    // Phase 1.4: when --weight-mode is native, checkpoint tensors stay in their
    // on-disk Q4_K / Q6_K formats and the dispatch path dequantizes on the
    // fly via MMQ. Not every linear weight is GGUF-quantized in a real
    // checkpoint (conv / norm are BF16), so the plan truthfully reports
    // "mixed BF16 + native GGUF MMQ" rather than falsely labeling the whole
    // model as BF16 / cuBLASLt. The per-tensor decision happens in
    // GemmDispatcher at run time from LinearWeight::kind.
    MixedBf16AndGgufMmq,
};

// Immutable, validated runtime plan. User-facing options are compiled once at
// construction so hot paths do not repeatedly reinterpret configuration.
class CudaExecutionPlan {
public:
    static CudaExecutionPlan compile(CudaModelOptions requested, int max_context);

    const CudaModelOptions& options() const { return options_; }
    LinearKernelKind linear_kernel() const { return linear_kernel_; }
    bool segmented_attention(int position) const;
    bool segmented_capable() const {
        return options_.attention_mode != AttentionMode::Single;
    }
    int attention_chunks() const { return attention_chunks_; }
    std::string description() const;

private:
    CudaModelOptions options_{};
    LinearKernelKind linear_kernel_ = LinearKernelKind::Bf16Cublas;
    int attention_chunks_ = 0;
};

} // namespace celeg

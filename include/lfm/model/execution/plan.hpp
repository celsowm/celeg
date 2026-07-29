#pragma once

#include "lfm/model/execution/runtime_types.hpp"

#include <string>

namespace lfm {

enum class LinearKernelKind : uint8_t {
    Bf16Cublas,
    Bf16CublasLt,
    W8A16,
    W4A16,
};

enum class SamplingKernelKind : uint8_t {
    Legacy,
    Fused,
};

// Immutable, validated runtime plan. User-facing options are compiled once at
// construction so hot paths do not repeatedly reinterpret configuration.
class ExecutionPlan {
public:
    static ExecutionPlan compile(ModelOptions requested, int max_context);

    const ModelOptions& options() const { return options_; }
    LinearKernelKind linear_kernel() const { return linear_kernel_; }
    SamplingKernelKind sampling_kernel() const { return sampling_kernel_; }
    bool segmented_attention(int position) const;
    bool segmented_capable() const {
        return options_.attention_mode != AttentionMode::Single;
    }
    int attention_chunks() const { return attention_chunks_; }
    std::string description() const;

private:
    ModelOptions options_{};
    LinearKernelKind linear_kernel_ = LinearKernelKind::Bf16Cublas;
    SamplingKernelKind sampling_kernel_ = SamplingKernelKind::Fused;
    int attention_chunks_ = 0;
};

} // namespace lfm

#pragma once

#include "../detail/model_internal.hpp"

#include <algorithm>
#include <cmath>

namespace celeg {

template <typename Body>
void cpu_parallel_rows(CpuThreadPool& pool, size_t rows, const Body& body) {
    const size_t grain = std::max<size_t>(1, rows / std::max<size_t>(1, pool.size() * 4));
    pool.parallel_for(0, rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) body(row);
    });
}

inline void cpu_layer_gemm(CpuCompiledModel::Shared& shared,
                           CpuWorkspace& workspace,
                           const CpuLinearWeight& weight,
                           const float* input,
                           float* output,
                           size_t rows,
                           size_t hidden,
                           bool& normed_q8_ready,
                           float beta = 0.0f) {
    const bool cacheable = weight.gguf_native() && weight.cols == hidden &&
        input == workspace.normed.data();
    if (cacheable) {
        if (!normed_q8_ready) {
            shared.linear.prepare_gguf_activation(input, rows, hidden, workspace.chunk_q8);
            normed_q8_ready = true;
        }
        shared.linear.gemm_gguf(workspace.chunk_q8, weight, output, rows, beta);
    } else {
        shared.linear.gemm(weight, input, output, rows, beta);
    }
}

inline float cpu_moe_sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

inline void cpu_chunk_layer_gemm(CpuExecutionContext& context,
                                 const CpuLinearWeight& weight,
                                 const float* input,
                                 float* output,
                                 size_t rows,
                                 size_t hidden,
                                 bool& normed_q8_ready,
                                 float beta = 0.0f) {
    auto& workspace = context.workspace;
    auto& shared = context.shared;
    const bool cacheable = weight.gguf_native() && weight.cols == hidden &&
        input == workspace.chunk_normed.data();
    if (cacheable) {
        if (!normed_q8_ready) {
            shared.linear.prepare_gguf_activation(input, rows, hidden,
                                                  workspace.chunk_q8);
            normed_q8_ready = true;
        }
        shared.linear.gemm_gguf(workspace.chunk_q8, weight, output, rows, beta);
    } else {
        shared.linear.gemm(weight, input, output, rows, beta);
    }
}

} // namespace celeg

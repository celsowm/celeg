#pragma once

#include "../detail/model_internal.hpp"

#include <array>

namespace celeg {

// Applies the complete Q/K positional and normalization semantics shared by
// token, chunk, and packed CPU attention execution. Projection kernels remain
// owned by the caller because their layout differs between execution modes.
void apply_cpu_attention_qk(const AttentionSpec& layout,
                            const CpuCompiledModel::AttentionWeights& weights,
                            float* query,
                            float* key,
                            int scalar_position,
                            const std::array<int32_t, 3>& rope_position);

void apply_cpu_latent_attention_positions(
    const ExecutionTopology& shape, const AttentionSpec& layout,
    float* query_rope, float* key_rope, int scalar_position,
    const std::array<int32_t, 3>& rope_position);

void apply_cpu_attention_output_transform(const AttentionSpec& layout,
                                          float* output,
                                          const float* current_value);

void apply_cpu_attention_output_gate(float* output, const float* gate, size_t width);
void apply_cpu_attention_output_gate(float* output, const float* gate,
                                     size_t width,
                                     AttentionGateGranularity granularity,
                                     int heads, int head_dim);

void execute_cpu_attention_token(
    CpuExecutionContext& execution,
    CpuCompiledModel& model,
    size_t index,
    const CpuCompiledModel::AttentionWeights& attention,
    const CompiledLayerProgram& semantics,
    const std::array<int32_t, 3>& rope_position);

} // namespace celeg

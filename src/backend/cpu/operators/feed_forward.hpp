#pragma once

#include "../detail/model_internal.hpp"

namespace celeg {

void execute_cpu_mlp_only_token(CpuCompiledModel& model, size_t layer,
                                const CpuCompiledModel::MlpOnlyWeights& weights);

void execute_cpu_dense_feed_forward_token(
    CpuCompiledModel& model, size_t layer,
    const CpuCompiledModel::CommonWeights& weights);

void execute_cpu_dense_feed_forward_chunk(
    CpuCompiledModel& model, size_t layer,
    const CpuCompiledModel::CommonWeights& weights,
    size_t rows, bool& normed_q8_ready);

} // namespace celeg

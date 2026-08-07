#pragma once

#include "../detail/model_internal.hpp"
#include "celeg/model/program.hpp"

#include <span>
#include <utility>
#include <vector>

namespace celeg {

struct CpuMoeRoute {
    std::vector<int> experts;
    std::vector<float> weights;
};

// Resolves router probabilities, bias, top-k ordering, normalization, and
// routed scaling once for both token and chunk execution.
CpuMoeRoute route_cpu_moe(const RouterProgram& program,
                          std::span<const float> logits,
                          std::span<const float> expert_bias);

// Executes the complete routed/shared feed-forward semantics. The caller is
// responsible for the surrounding residual and normalization operations.
void execute_cpu_moe_token(CpuCompiledModel& model, size_t layer,
                           const CpuCompiledModel::MoeWeights& weights,
                           const MoeLayerProgram& semantics);

void execute_cpu_moe_chunk(CpuCompiledModel& model, size_t layer,
                           const CpuCompiledModel::MoeWeights& weights,
                           const MoeLayerProgram& semantics,
                           size_t rows, bool& normed_q8_ready);

} // namespace celeg

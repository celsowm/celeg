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

CpuMoeRoute route_cpu_moe(const RouterProgram& program,
                          std::span<const float> logits,
                          std::span<const float> expert_bias);

void execute_cpu_moe_token(CpuExecutionContext& context, size_t layer,
                           const CpuCompiledModel::MoeWeights& weights,
                           const MoeLayerProgram& semantics);

void execute_cpu_moe_chunk(CpuExecutionContext& context, size_t layer,
                           const CpuCompiledModel::MoeWeights& weights,
                           const MoeLayerProgram& semantics,
                           size_t rows, bool& normed_q8_ready);

}

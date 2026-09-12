#pragma once

#include "backend/cuda/runtime_types.hpp"
#include "celeg/model/program.hpp"
#include "celeg/model/resolved.hpp"

#include <algorithm>

namespace celeg {

/// Fused residuals assume unit multipliers throughout the residual path. A
/// program with per-layer residual scaling (or global embedding/logit
/// scaling on a convolution-free graph) cannot use them, so the option is
/// forced off here. Single-model setup and the concurrent engine's packed
/// executor must agree on this adjustment, otherwise their execution plans
/// fingerprint differently and packed sessions are rejected.
inline CudaModelOptions effective_cuda_options_for_program(
    CudaModelOptions options,
    const CompiledModelProgram& program,
    const ExecutionTopology& shape) {
    const bool non_default_residual = std::any_of(
        program.layers.begin(), program.layers.end(),
        [](const CompiledLayerProgram& layer) {
            return layer.residual.multiplier != 1.0f;
        });
    if (non_default_residual ||
        (shape.conv_layer_count == 0 &&
         (program.embedding_transform.multiplier != 1.0f ||
          program.logits_multiplier != 1.0f ||
          program.logits_divisor != 1.0f))) {
        options.fused_residuals = false;
    }
    return options;
}

}

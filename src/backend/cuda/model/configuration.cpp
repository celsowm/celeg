#include "detail/compiled_model.hpp"
#include "backend/cuda/compiler.hpp"
#include "checkpoint/detail/bootstrap.hpp"
#include "backend/cuda/weight_policy.hpp"
#include "celeg/checkpoint/packed/int8.hpp"

#include <algorithm>
#include <stdexcept>
namespace celeg {

void CudaCompiledModel::configure_model(
    const detail::ModelBootstrap& bootstrap) {
    resources_.model_ = bootstrap.model;
    resources_.program_ = CudaModelCompiler{}.compile(resources_.model_);
    resources_.topology_ = resources_.model_.topology;

    CudaModelOptions effective_options = resources_.plan_.options();
    if (effective_options.weight_mode == WeightMode::Bf16 &&
        has_packed_int8_matrix(*bootstrap.checkpoint.repository,
                               "model.layers.0.self_attn.q_proj.weight")) {
        effective_options.weight_mode = WeightMode::Int8;
    }

    const bool non_default_residual = std::any_of(
        resources_.program_.layers.begin(), resources_.program_.layers.end(),
        [](const CompiledLayerProgram& layer) {
            return layer.residual.multiplier != 1.0f;
        });
    if (non_default_residual ||
        (resources_.shape_.conv_layer_count == 0 &&
         (resources_.program_.embedding_transform.multiplier != 1.0f ||
          resources_.program_.logits_multiplier != 1.0f ||
          resources_.program_.logits_divisor != 1.0f))) {
        effective_options.fused_residuals = false;
    }

    resources_.plan_ = CudaExecutionPlan::compile(
        std::move(effective_options), max_context_, resources_.plan_.device());
    resources_.model_identity_ = resources_.model_.provenance.identity;
    check_moe_quantization_policy(resources_.options_.weight_mode, resources_.program_.has_moe());
}

}

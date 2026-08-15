#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/compiler.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/weight_policy.hpp"
#include "celeg/checkpoint/packed/int8.hpp"

#include <algorithm>
#include <stdexcept>
namespace celeg {

void CudaCompiledModel::configure_model(
    const detail::ModelBootstrap& bootstrap) {
    resources_.model_ = bootstrap.model;
    if (resources_.options_.weight_mode == WeightMode::Bf16 &&
        has_packed_int8_matrix(*bootstrap.checkpoint.repository,
                               "model.layers.0.self_attn.q_proj.weight")) {
        resources_.options_.weight_mode = WeightMode::Int8;
        resources_.plan_ = CudaExecutionPlan::compile(
            resources_.options_, max_context_, resources_.plan_.device());
    }
    resources_.program_ = CudaModelCompiler{}.compile(resources_.model_);
    resources_.topology_ = resources_.model_.topology;
    const bool non_default_residual = std::any_of(
        resources_.program_.layers.begin(), resources_.program_.layers.end(),
        [](const CompiledLayerProgram& layer) {
            return layer.residual.multiplier != 1.0f;
        });
    if (resources_.shape_.conv_layer_count == 0 &&
        (resources_.program_.embedding_transform.multiplier != 1.0f ||
         non_default_residual ||
         resources_.program_.logits_multiplier != 1.0f ||
         resources_.program_.logits_divisor != 1.0f)) {
        resources_.options_.fused_residuals = false;
    }
    resources_.model_identity_ = resources_.model_.provenance.identity;
    check_moe_quantization_policy(resources_.options_.weight_mode, resources_.program_.has_moe());
}

}


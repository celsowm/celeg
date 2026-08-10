#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/compiler.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/weight_policy.hpp"
#include "celeg/checkpoint/packed_int8.hpp"

#include <stdexcept>
namespace celeg {

void CudaCompiledModel::configure_model(
    const detail::ModelBootstrap& bootstrap) {
    resources_.model_ = bootstrap.model;
    if (resources_.options_.weight_mode == WeightMode::Bf16 &&
        has_packed_int8_matrix(*bootstrap.checkpoint.repository,
                               "model.layers.0.self_attn.q_proj.weight")) {
        // compressed-tensors checkpoints carry their quantization contract in
        // the tensors themselves; execute them as native W8A16 even when the
        // caller did not request a conversion mode.
        resources_.options_.weight_mode = WeightMode::Int8;
        resources_.plan_ = CudaExecutionPlan::compile(
            resources_.options_, max_context_, resources_.plan_.device());
    }
    resources_.program_ = CudaModelCompiler{}.compile(resources_.model_);
    resources_.shape_ = resources_.model_.topology;
    if (resources_.shape_.conv_layer_count == 0 &&
        (resources_.program_.embedding_transform.multiplier != 1.0f ||
         resources_.shape_.numerical_policy.attention_multiplier != 0.0f ||
         resources_.shape_.numerical_policy.residual_multiplier != 1.0f ||
         resources_.shape_.numerical_policy.logits_multiplier != 1.0f ||
         resources_.shape_.numerical_policy.logits_divisor != 1.0f)) {
        resources_.options_.fused_residuals = false;
    }
    resources_.model_identity_ = resources_.model_.provenance.identity;
    check_moe_quantization_policy(resources_.options_.weight_mode, resources_.shape_.num_experts > 0);
}

} // namespace celeg


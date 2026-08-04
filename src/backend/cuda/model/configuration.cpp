#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/compiler.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/weight_policy.hpp"

#include <stdexcept>
namespace celeg {

void CudaCompiledModel::configure_model(
    const detail::ModelBootstrap& bootstrap) {
    resources_.model_ = bootstrap.model;
    resources_.program_ = CudaModelCompiler{}.compile(resources_.model_);
    resources_.shape_ = resources_.model_.topology;
    if (resources_.shape_.conv_layer_count == 0 &&
        (resources_.shape_.embedding_multiplier != 1.0f ||
         resources_.shape_.attention_multiplier != 0.0f ||
         resources_.shape_.residual_multiplier != 1.0f ||
         resources_.shape_.logits_divisor != 1.0f)) {
        resources_.options_.fused_residuals = false;
    }
    resources_.model_identity_ = resources_.model_.identity;
    check_moe_quantization_policy(resources_.options_.weight_mode, resources_.shape_.num_experts > 0);
}

} // namespace celeg


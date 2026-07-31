#include "celeg/detail/model/impl.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/model/weights/policy.hpp"

#include <stdexcept>
namespace celeg {

void Model::Impl::configure_model(
    const detail::ModelBootstrap& bootstrap) {
    shape_ = bootstrap.shape;
    if (shape_.conv_layer_count == 0 &&
        (shape_.embedding_multiplier != 1.0f ||
         shape_.attention_multiplier != 0.0f ||
         shape_.residual_multiplier != 1.0f ||
         shape_.logits_divisor != 1.0f)) {
        options_.fused_residuals = false;
    }
    variant_ = bootstrap.variant;
    tensor_naming_ = &bootstrap.architecture_provider->tensor_naming();
    check_moe_quantization_policy(options_.weight_mode, shape_.num_experts > 0);
}

} // namespace celeg


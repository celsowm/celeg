#include "lfm/detail/model/impl.hpp"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/model/weights/policy.hpp"

#include <stdexcept>
namespace lfm {

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

} // namespace lfm


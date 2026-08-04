#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"

#include <cmath>
#include <stdexcept>

namespace celeg {

namespace {

constexpr float kPerLayerResidualScale = 0.7071067811865475f;

struct PerLayerInputDimensions {
    int layers;
    int ple;
    int width;
};

PerLayerInputDimensions per_layer_input_dimensions(const CudaCompiledModel& model) {
    const int layers = model.resources_.shape_.num_hidden_layers;
    const int ple = model.resources_.shape_.per_layer_input_size;
    return {layers, ple, layers * ple};
}

} // namespace

void CudaCompiledModel::validate_per_layer_input() const {
    if (!resources_.per_layer_embedding_ || !resources_.per_layer_embedding_->bf16 ||
        !resources_.per_layer_context_projection_ || !resources_.per_layer_projection_norm_) {
        throw std::logic_error("per-layer input weights are not BF16-resident");
    }
}

void CudaCompiledModel::initialize_per_layer_input_device(const int32_t* token) {
    if (!resources_.shape_.has_per_layer_input) return;
    validate_per_layer_input();
    const auto dimensions = per_layer_input_dimensions(*this);
    launch_embedding_slice_device(token, resources_.per_layer_embedding_->bf16,
                                  dimensions.width, 0, workspace_.per_layer_token_.data(), dimensions.width,
                                  stream_.get());
    launch_scale(workspace_.per_layer_token_.data(), dimensions.width,
                 std::sqrt(static_cast<float>(dimensions.ple)), stream_.get());
    linear(workspace_.hidden_.data(), *resources_.per_layer_context_projection_,
           workspace_.per_layer_context_.data(), 1, dimensions.width, resources_.shape_.hidden);
    launch_scale(workspace_.per_layer_context_.data(), dimensions.width,
                 1.0f / std::sqrt(static_cast<float>(resources_.shape_.hidden)), stream_.get());
    launch_rmsnorm(workspace_.per_layer_context_.data(), resources_.per_layer_projection_norm_,
                   workspace_.per_layer_context_.data(), dimensions.layers, dimensions.ple,
                   resources_.shape_.numerical_policy.norm_eps, stream_.get());
    launch_residual_add(workspace_.per_layer_context_.data(), workspace_.per_layer_token_.data(),
                        dimensions.width, stream_.get());
    launch_scale(workspace_.per_layer_context_.data(), dimensions.width,
                 kPerLayerResidualScale, stream_.get());
}

void CudaCompiledModel::initialize_per_layer_input_host(int32_t token) {
    if (!resources_.shape_.has_per_layer_input) return;
    validate_per_layer_input();
    const auto dimensions = per_layer_input_dimensions(*this);
    launch_embedding_slice(token, resources_.per_layer_embedding_->bf16, dimensions.width, 0,
                           workspace_.per_layer_token_.data(), dimensions.width, stream_.get());
    launch_scale(workspace_.per_layer_token_.data(), dimensions.width,
                 std::sqrt(static_cast<float>(dimensions.ple)), stream_.get());
    linear(workspace_.hidden_.data(), *resources_.per_layer_context_projection_,
           workspace_.per_layer_context_.data(), 1, dimensions.width, resources_.shape_.hidden);
    launch_scale(workspace_.per_layer_context_.data(), dimensions.width,
                 1.0f / std::sqrt(static_cast<float>(resources_.shape_.hidden)), stream_.get());
    launch_rmsnorm(workspace_.per_layer_context_.data(), resources_.per_layer_projection_norm_,
                   workspace_.per_layer_context_.data(), dimensions.layers, dimensions.ple,
                   resources_.shape_.numerical_policy.norm_eps, stream_.get());
    launch_residual_add(workspace_.per_layer_context_.data(), workspace_.per_layer_token_.data(),
                        dimensions.width, stream_.get());
    launch_scale(workspace_.per_layer_context_.data(), dimensions.width,
                 kPerLayerResidualScale, stream_.get());
}

void CudaCompiledModel::initialize_per_layer_input_batch(const int32_t* tokens, int rows) {
    if (!resources_.shape_.has_per_layer_input) return;
    validate_per_layer_input();
    const auto dimensions = per_layer_input_dimensions(*this);
    launch_embedding_slice_batch(tokens, rows, resources_.per_layer_embedding_->bf16,
                                 dimensions.width, 0, workspace_.prefill_per_layer_token_.data(),
                                 dimensions.width, stream_.get());
    launch_scale(workspace_.prefill_per_layer_token_.data(), rows * dimensions.width,
                 std::sqrt(static_cast<float>(dimensions.ple)), stream_.get());
    linear(workspace_.prefill_hidden_.data(), *resources_.per_layer_context_projection_,
           workspace_.prefill_per_layer_context_.data(), rows, dimensions.width,
           resources_.shape_.hidden);
    launch_scale(workspace_.prefill_per_layer_context_.data(), rows * dimensions.width,
                 1.0f / std::sqrt(static_cast<float>(resources_.shape_.hidden)), stream_.get());
    launch_rmsnorm(workspace_.prefill_per_layer_context_.data(), resources_.per_layer_projection_norm_,
                   workspace_.prefill_per_layer_context_.data(), rows * dimensions.layers, dimensions.ple,
                   resources_.shape_.numerical_policy.norm_eps, stream_.get());
    launch_residual_add(workspace_.prefill_per_layer_context_.data(),
                        workspace_.prefill_per_layer_token_.data(), rows * dimensions.width,
                        stream_.get());
    launch_scale(workspace_.prefill_per_layer_context_.data(), rows * dimensions.width,
                 kPerLayerResidualScale, stream_.get());
}

} // namespace celeg

#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"

namespace celeg {

void CudaCompiledModel::initialize_per_layer_input_device(const int32_t* token) {
    if (!resources_.per_layer_input_) return;
    const CudaPerLayerInputResources& per_layer = *resources_.per_layer_input_;
    const PerLayerInputPlan& plan = per_layer.plan;
    per_layer.embedding_layout->embed_token_device(
        token, workspace_.per_layer_token_.data(),
        static_cast<int>(plan.packed_width), stream_.get());
    launch_scale(workspace_.per_layer_token_.data(), static_cast<int>(plan.packed_width),
                 plan.token_scale, stream_.get());
    linear(workspace_.hidden_.data(), *per_layer.context_projection,
           workspace_.per_layer_context_.data(), 1, static_cast<int>(plan.packed_width),
           resources_.program_.hidden);
    launch_scale(workspace_.per_layer_context_.data(), static_cast<int>(plan.packed_width),
                 plan.context_scale, stream_.get());
    launch_rmsnorm(workspace_.per_layer_context_.data(), per_layer.projection_norm,
                   workspace_.per_layer_context_.data(), plan.layer_count,
                   plan.input_size, plan.norm_epsilon, stream_.get());
    launch_residual_add(workspace_.per_layer_context_.data(), workspace_.per_layer_token_.data(),
                        static_cast<int>(plan.packed_width), stream_.get());
    launch_scale(workspace_.per_layer_context_.data(), static_cast<int>(plan.packed_width),
                 plan.residual_scale, stream_.get());
}

void CudaCompiledModel::initialize_per_layer_input_host(int32_t token) {
    if (!resources_.per_layer_input_) return;
    const CudaPerLayerInputResources& per_layer = *resources_.per_layer_input_;
    const PerLayerInputPlan& plan = per_layer.plan;
    per_layer.embedding_layout->embed_token(
        token, workspace_.per_layer_token_.data(),
        static_cast<int>(plan.packed_width), stream_.get());
    launch_scale(workspace_.per_layer_token_.data(), static_cast<int>(plan.packed_width),
                 plan.token_scale, stream_.get());
    linear(workspace_.hidden_.data(), *per_layer.context_projection,
           workspace_.per_layer_context_.data(), 1,
           static_cast<int>(plan.packed_width), resources_.program_.hidden);
    launch_scale(workspace_.per_layer_context_.data(), static_cast<int>(plan.packed_width),
                 plan.context_scale, stream_.get());
    launch_rmsnorm(workspace_.per_layer_context_.data(), per_layer.projection_norm,
                   workspace_.per_layer_context_.data(), plan.layer_count,
                   plan.input_size, plan.norm_epsilon, stream_.get());
    launch_residual_add(workspace_.per_layer_context_.data(), workspace_.per_layer_token_.data(),
                        static_cast<int>(plan.packed_width), stream_.get());
    launch_scale(workspace_.per_layer_context_.data(), static_cast<int>(plan.packed_width),
                 plan.residual_scale, stream_.get());
}

void CudaCompiledModel::initialize_per_layer_input_batch(const int32_t* tokens, int rows) {
    if (!resources_.per_layer_input_) return;
    const CudaPerLayerInputResources& per_layer = *resources_.per_layer_input_;
    const PerLayerInputPlan& plan = per_layer.plan;
    const std::size_t elements = plan.checked_elements(static_cast<std::size_t>(rows));
    per_layer.embedding_layout->embed_batch(
        tokens, rows, workspace_.prefill_per_layer_token_.data(),
        static_cast<int>(plan.packed_width), stream_.get());
    launch_scale(workspace_.prefill_per_layer_token_.data(), static_cast<int>(elements),
                 plan.token_scale, stream_.get());
    linear(workspace_.prefill_hidden_.data(), *per_layer.context_projection,
           workspace_.prefill_per_layer_context_.data(), rows,
           static_cast<int>(plan.packed_width), resources_.program_.hidden);
    launch_scale(workspace_.prefill_per_layer_context_.data(), static_cast<int>(elements),
                 plan.context_scale, stream_.get());
    launch_rmsnorm(workspace_.prefill_per_layer_context_.data(), per_layer.projection_norm,
                   workspace_.prefill_per_layer_context_.data(),
                   static_cast<int>(elements / static_cast<std::size_t>(plan.input_size)),
                   plan.input_size, plan.norm_epsilon, stream_.get());
    launch_residual_add(workspace_.prefill_per_layer_context_.data(),
                   workspace_.prefill_per_layer_token_.data(), static_cast<int>(elements),
                        stream_.get());
    launch_scale(workspace_.prefill_per_layer_context_.data(), static_cast<int>(elements),
                 plan.residual_scale, stream_.get());
}

} // namespace celeg

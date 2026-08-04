#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"

namespace celeg {

void CudaCompiledModel::run_per_layer_input_decode(const LayerCommon& common_layer,
                                                   int layer) {
    if (!resources_.shape_.has_per_layer_input) return;
    const int ple = resources_.shape_.per_layer_input_size;
    launch_gelu_tanh(workspace_.per_layer_gate_.data(), workspace_.per_layer_gate_.data(),
                     ple, stream_.get());
    const __nv_bfloat16* context = workspace_.per_layer_context_.data() +
        static_cast<size_t>(layer) * ple;
    launch_multiply(workspace_.per_layer_gate_.data(), context, ple, stream_.get());
    CELEG_CUDA(cudaMemcpyAsync(workspace_.residual_.data(), workspace_.hidden_.data(),
                               workspace_.hidden_.bytes(), cudaMemcpyDeviceToDevice,
                               stream_.get()));
    linear(workspace_.per_layer_gate_.data(), *common_layer.per_layer_projection,
           workspace_.hidden_.data(), 1, resources_.shape_.hidden, ple);
    launch_rmsnorm(workspace_.hidden_.data(), common_layer.per_layer_input_norm,
                   workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                   resources_.shape_.numerical_policy.norm_eps, stream_.get());
    launch_scale_by_scalar(workspace_.hidden_.data(), common_layer.layer_scalar,
                           resources_.shape_.hidden, stream_.get());
    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                        resources_.shape_.hidden, stream_.get());
}

void CudaCompiledModel::run_per_layer_input_prefill(const LayerCommon& common_layer,
                                                    int rows, int layer) {
    if (!resources_.shape_.has_per_layer_input) return;
    const int layers = resources_.shape_.num_hidden_layers;
    const int ple = resources_.shape_.per_layer_input_size;
    linear(workspace_.prefill_hidden_.data(), *common_layer.per_layer_input_gate,
           workspace_.prefill_per_layer_gate_.data(), rows, ple,
           resources_.shape_.hidden);
    launch_gelu_tanh(workspace_.prefill_per_layer_gate_.data(),
                     workspace_.prefill_per_layer_gate_.data(), rows * ple,
                     stream_.get());
    launch_multiply_strided(workspace_.prefill_per_layer_gate_.data(),
                            workspace_.prefill_per_layer_context_.data(), rows, ple,
                            layers * ple, layer * ple, stream_.get());
    CELEG_CUDA(cudaMemcpyAsync(workspace_.prefill_residual_.data(),
                               workspace_.prefill_hidden_.data(),
                               workspace_.prefill_hidden_.bytes(),
                               cudaMemcpyDeviceToDevice, stream_.get()));
    linear(workspace_.prefill_per_layer_gate_.data(), *common_layer.per_layer_projection,
           workspace_.prefill_hidden_.data(), rows, resources_.shape_.hidden, ple);
    launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.per_layer_input_norm,
                   workspace_.prefill_hidden_.data(), rows, resources_.shape_.hidden,
                   resources_.shape_.numerical_policy.norm_eps, stream_.get());
    launch_scale_by_scalar(workspace_.prefill_hidden_.data(), common_layer.layer_scalar,
                           rows * resources_.shape_.hidden, stream_.get());
    launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.prefill_residual_.data(),
                        rows * resources_.shape_.hidden, stream_.get());
}

} // namespace celeg

#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"

namespace celeg {

void CudaCompiledModel::run_per_layer_input_decode(const LayerCommon& common_layer,
                                                   int layer) {
    const PerLayerInputPlan& plan = resources_.program_.per_layer_input;
    if (!plan.enabled) return;
    const int ple = plan.input_size;
    launch_gelu_tanh(workspace_.per_layer_gate_.data(), workspace_.per_layer_gate_.data(),
                     ple, stream_.get());
    const __nv_bfloat16* context = workspace_.per_layer_context_.data() +
        static_cast<size_t>(layer) * ple;
    launch_multiply(workspace_.per_layer_gate_.data(), context, ple, stream_.get());
    CELEG_CUDA(cudaMemcpyAsync(workspace_.residual_.data(), workspace_.hidden_.data(),
                               workspace_.hidden_.bytes(), cudaMemcpyDeviceToDevice,
                               stream_.get()));
    linear(workspace_.per_layer_gate_.data(), *common_layer.per_layer_projection,
           workspace_.hidden_.data(), 1, resources_.program_.hidden, ple);
    launch_rmsnorm(workspace_.hidden_.data(), common_layer.per_layer_input_norm,
                   workspace_.hidden_.data(), 1, resources_.program_.hidden,
                   plan.norm_epsilon, stream_.get());
    launch_scale_by_scalar(workspace_.hidden_.data(), common_layer.layer_scalar,
                           resources_.program_.hidden, stream_.get());
    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                        resources_.program_.hidden, stream_.get());
}

void CudaCompiledModel::run_per_layer_input_prefill(const LayerCommon& common_layer,
                                                    int rows, int layer) {
    const PerLayerInputPlan& plan = resources_.program_.per_layer_input;
    if (!plan.enabled) return;
    const int layers = plan.layer_count;
    const int ple = plan.input_size;
    const std::size_t packed_elements =
        plan.checked_elements(static_cast<std::size_t>(rows));
    const int row_elements = static_cast<int>(packed_elements /
                                               static_cast<std::size_t>(layers));
    linear(workspace_.prefill_hidden_.data(), *common_layer.per_layer_input_gate,
           workspace_.prefill_per_layer_gate_.data(), rows, ple,
           resources_.program_.hidden);
    launch_gelu_tanh(workspace_.prefill_per_layer_gate_.data(),
                     workspace_.prefill_per_layer_gate_.data(), row_elements,
                     stream_.get());
    launch_multiply_strided(workspace_.prefill_per_layer_gate_.data(),
                            workspace_.prefill_per_layer_context_.data(), rows, ple,
                            static_cast<int>(plan.packed_width), layer * ple, stream_.get());
    CELEG_CUDA(cudaMemcpyAsync(workspace_.prefill_residual_.data(),
                               workspace_.prefill_hidden_.data(),
                               workspace_.prefill_hidden_.bytes(),
                               cudaMemcpyDeviceToDevice, stream_.get()));
    linear(workspace_.prefill_per_layer_gate_.data(), *common_layer.per_layer_projection,
           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden, ple);
    launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.per_layer_input_norm,
                   workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                   plan.norm_epsilon, stream_.get());
    launch_scale_by_scalar(workspace_.prefill_hidden_.data(), common_layer.layer_scalar,
                           rows * resources_.program_.hidden, stream_.get());
    launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.prefill_residual_.data(),
                        rows * resources_.program_.hidden, stream_.get());
}

}

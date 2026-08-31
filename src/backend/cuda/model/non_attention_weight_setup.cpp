#include "non_attention_weight_setup.hpp"

#include "detail/compiled_model.hpp"
#include "backend/cuda/weight_setup_support.hpp"

#include <stdexcept>

namespace celeg {

void bind_cuda_non_attention_layer(CudaCompiledModel& model,
                                   const IWeightRepository& repo,
                                   const CompiledLayerProgram& semantics,
                                   int layer_index,
                                   const LayerCommon& common_layer) {
    CudaModelResources& resources = model.resources_;

    if (const auto* gated_delta =
            std::get_if<GatedDeltaNetSpec>(&semantics.mixer)) {
        GatedDeltaNetLayer layer;
        layer.common = common_layer;
        layer.spec = *gated_delta;
        const GatedDeltaNetSpec& spec = layer.spec;
        const int key_width = spec.key_heads * spec.key_head_dim;
        const int value_width = spec.value_heads * spec.value_head_dim;
        const int qkv_width = 2 * key_width + value_width;

        if (spec.factorized_projections) {
            layer.q = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetQuery, layer_index),
                {key_width, resources.program_.hidden});
            layer.k = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetKey, layer_index),
                {key_width, resources.program_.hidden});
            layer.v = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetValue, layer_index),
                {value_width, resources.program_.hidden});
            layer.z = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetOutputGate, layer_index),
                {value_width, resources.program_.hidden});
        } else {
            layer.qkv = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetQkv, layer_index),
                {qkv_width, resources.program_.hidden});
            layer.z = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetZ, layer_index),
                {value_width, resources.program_.hidden});
        }

        layer.b = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::GatedDeltaNetBeta, layer_index),
            {spec.value_heads, resources.program_.hidden});
        layer.a = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(
                resources.model_.weight_plan.requests,
                spec.factorized_projections
                    ? TensorRole::GatedDeltaNetDecay
                    : TensorRole::GatedDeltaNetAlpha,
                layer_index),
            {spec.decay_width(), resources.program_.hidden});

        if (spec.factorized_projections) {
            const auto* q_conv = resources.weight_loader_->load_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetQueryConv, layer_index),
                {key_width, 1, spec.conv_kernel});
            const auto* k_conv = resources.weight_loader_->load_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetKeyConv, layer_index),
                {key_width, 1, spec.conv_kernel});
            const auto* v_conv = resources.weight_loader_->load_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetValueConv, layer_index),
                {value_width, 1, spec.conv_kernel});
            layer.factorized_conv_weight.reset(
                static_cast<size_t>(qkv_width) * spec.conv_kernel);
            const size_t q_bytes =
                static_cast<size_t>(key_width) * spec.conv_kernel *
                sizeof(__nv_bfloat16);
            const size_t v_bytes =
                static_cast<size_t>(value_width) * spec.conv_kernel *
                sizeof(__nv_bfloat16);
            CELEG_CUDA(cudaMemcpy(layer.factorized_conv_weight.data(),
                                  q_conv, q_bytes, cudaMemcpyDeviceToDevice));
            CELEG_CUDA(cudaMemcpy(
                layer.factorized_conv_weight.data() + key_width * spec.conv_kernel,
                k_conv, q_bytes, cudaMemcpyDeviceToDevice));
            CELEG_CUDA(cudaMemcpy(
                layer.factorized_conv_weight.data() +
                    2 * key_width * spec.conv_kernel,
                v_conv, v_bytes, cudaMemcpyDeviceToDevice));
            layer.conv_weight = layer.factorized_conv_weight.data();
        } else {
            layer.conv_weight = resources.weight_loader_->load_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::GatedDeltaNetConv, layer_index),
                {qkv_width, 1, spec.conv_kernel});
        }

        layer.dt_bias = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::GatedDeltaNetDtBias, layer_index),
            {spec.decay_width()});
        layer.a_log = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::GatedDeltaNetALog, layer_index),
            {spec.value_heads});
        layer.norm = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::GatedDeltaNetNorm, layer_index),
            {spec.value_head_dim});
        layer.out = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::GatedDeltaNetOutput, layer_index),
            {resources.program_.hidden, value_width});
        layer.conv_state.reset(static_cast<size_t>(qkv_width) * spec.conv_kernel);
        layer.recurrent_state.reset(
            static_cast<size_t>(spec.value_heads) * spec.key_head_dim *
            spec.value_head_dim);
        resources.layers_.emplace_back(std::move(layer));
        return;
    }

    if (const auto* mamba = std::get_if<Mamba2Spec>(&semantics.mixer)) {
        Mamba2Layer layer;
        layer.common = common_layer;
        layer.spec = *mamba;
        const Mamba2Spec& spec = layer.spec;
        const int conv_dim =
            spec.intermediate_size + 2 * spec.group_count * spec.state_size;
        layer.in = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::Mamba2Input, layer_index),
            {2 * spec.intermediate_size +
                 2 * spec.group_count * spec.state_size + spec.num_heads,
             resources.program_.hidden});
        layer.conv_weight = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::Mamba2Conv, layer_index),
            {conv_dim, 1, spec.conv_kernel});
        layer.conv_bias = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::Mamba2ConvBias, layer_index),
            {conv_dim});
        layer.dt_bias = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::Mamba2DtBias, layer_index),
            {spec.num_heads});
        layer.a_log = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::Mamba2ALog, layer_index),
            {spec.num_heads});
        layer.d = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::Mamba2D, layer_index),
            {spec.num_heads});
        layer.norm = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::Mamba2Norm, layer_index),
            {spec.intermediate_size});
        layer.out = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::Mamba2Output, layer_index),
            {resources.program_.hidden, spec.intermediate_size});
        layer.conv_state.reset(static_cast<size_t>(conv_dim) * spec.conv_kernel);
        layer.ssm_state.reset(
            static_cast<size_t>(spec.intermediate_size) * spec.state_size);
        resources.layers_.emplace_back(std::move(layer));
        return;
    }

    if (const auto* mlp = std::get_if<MlpBlockSpec>(&semantics.mixer)) {
        MlpOnlyLayer layer;
        layer.common = common_layer;
        layer.spec = *mlp;
        layer.up = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::FfnUp, layer_index),
            {layer.spec.intermediate_size, resources.program_.hidden});
        layer.down = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::FfnDown, layer_index),
            {resources.program_.hidden, layer.spec.intermediate_size});
        resources.layers_.emplace_back(std::move(layer));
        return;
    }

    if (const auto* convolution =
            std::get_if<ShortConvolutionSpec>(&semantics.mixer)) {
        ConvolutionLayer layer;
        layer.common = common_layer;
        layer.spec = *convolution;
        layer.conv_in = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::ShortConvInput, layer_index),
            {3 * resources.program_.hidden, resources.program_.hidden});
        layer.conv_weight = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::ShortConvKernel, layer_index),
            {resources.program_.hidden, 1, layer.spec.cache_length});
        layer.conv_out = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::ShortConvOutput, layer_index),
            {resources.program_.hidden, resources.program_.hidden});
        layer.conv_state.reset(
            static_cast<size_t>(layer.spec.cache_length) * resources.program_.hidden);
        resources.layers_.emplace_back(std::move(layer));
        return;
    }

    throw std::logic_error("unsupported CUDA non-attention mixer during weight setup");
}

}

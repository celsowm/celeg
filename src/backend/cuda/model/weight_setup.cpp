#include "detail/compiled_model.hpp"
#include "checkpoint/detail/bootstrap.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "backend/cuda/weight_policy.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/weights_loader.hpp"
#include "backend/cuda/weight_setup_support.hpp"
#include "backend/cuda/weight_setup.hpp"
#include "attention_weight_setup.hpp"
#include "moe_weight_setup.hpp"

#include <filesystem>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace celeg {

namespace {
std::string layer_name(int index, const std::string& suffix) {
    return cuda_layer_name(index, suffix);
}

std::string tensor_name(std::span<const TensorRequest> requests, TensorRole role,
                        int layer = -1) {
    return cuda_tensor_name(requests, role, layer);
}

}

void CudaCompiledModel::load_checkpoint_weights(
    const std::string& model_path,
    const detail::ModelBootstrap& bootstrap) {
    CudaWeightSetup::load(*this, model_path, bootstrap,
        [this](const IWeightRepository& repo) {
    configure_cuda_expert_resources(*this);
    const int mtp_layer_count = resources_.options().enable_mtp
        ? resources_.dims().mtp_num_hidden_layers : 0;
    const int resource_layer_count = resources_.shape().num_hidden_layers +
        mtp_layer_count;
    workspace_.expert_caches_.resize(static_cast<size_t>(resource_layer_count));
    if (resources_.weights_->expert_controllers.size() <
        static_cast<size_t>(resource_layer_count)) {
        resources_.weights_->expert_controllers.resize(
            static_cast<size_t>(resource_layer_count));
    }
    workspace_.expert_catalog_.resize(static_cast<size_t>(resource_layer_count));
    if (resources_.weights_->expert_catalog.size() <
        static_cast<size_t>(resource_layer_count)) {
        resources_.weights_->expert_catalog.resize(
            static_cast<size_t>(resource_layer_count));
    }

    resources_.layers_.reserve(static_cast<size_t>(resources_.shape().num_hidden_layers));
    std::vector<int> shared_owner(2, -1);
    for (int i = 0; i < resources_.shape().num_hidden_layers; ++i) {
        LayerCommon common_layer;
        const CompiledLayerProgram& semantic_layer = resources_.program_.layers.at(
            static_cast<size_t>(i));
        const bool mixer_only_layer =
            std::holds_alternative<std::monostate>(semantic_layer.feed_forward);
        const auto load_norm = [&](TensorRole role, const NormSpec& spec) {
            const std::string name = spec.weightless()
                ? std::string{} : tensor_name(resources_.model_.weight_plan.requests, role, i);
            return resources_.weight_loader_->load_rms_norm_weight(
                repo, name, {resources_.program_.hidden}, spec.weight_kind);
        };
        if (semantic_layer.mixer_norm.before) {
            common_layer.mixer_norm_before = load_norm(
                TensorRole::AttentionInputNorm, *semantic_layer.mixer_norm.before);
        }
        if (semantic_layer.mixer_norm.after) {
            common_layer.mixer_norm_after = load_norm(
                TensorRole::AttentionPostNorm, *semantic_layer.mixer_norm.after);
        }
        if (!mixer_only_layer && semantic_layer.feed_forward_norm.before) {
            common_layer.feed_forward_norm_before = load_norm(
                TensorRole::FfnInputNorm, *semantic_layer.feed_forward_norm.before);
        }
        if (!mixer_only_layer && semantic_layer.feed_forward_norm.after) {
            common_layer.feed_forward_norm_after = load_norm(
                TensorRole::FfnOutputNorm, *semantic_layer.feed_forward_norm.after);
        }
        if (resources_.program_.per_layer_input.enabled) {
            common_layer.per_layer_input_gate = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerInputGate, i),
                {resources_.program_.per_layer_input.input_size, resources_.program_.hidden});
            common_layer.per_layer_projection = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerProjection, i),
                {resources_.program_.hidden, resources_.program_.per_layer_input.input_size});
            common_layer.per_layer_input_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerInputNorm, i),
                {resources_.program_.hidden});
            common_layer.layer_scalar = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::LayerScalar, i), {1});
        }
        if (mixer_only_layer) {
            common_layer.feed_forward = std::monostate{};
        } else if (const auto* moe_program =
                       std::get_if<MoeLayerProgram>(&semantic_layer.feed_forward)) {
            bind_cuda_moe_feed_forward(*this, repo, *moe_program, i, common_layer);
        } else {
            const auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(
                &semantic_layer.feed_forward);
            if (!dense || dense->intermediate_size <= 0) {
                throw std::runtime_error("compiled dense layer has no FFN width");
            }
            const int intermediate = dense->intermediate_size;
            const LinearWeight* w13 = resources_.weight_loader_->load_concat_linear_weight(
                repo, layer_name(i, "feed_forward.w13.weight"),
                {
                    {tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnGate, i),
                     {intermediate, resources_.program_.hidden}},
                    {tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnUp, i),
                     {intermediate, resources_.program_.hidden}},
                });
            const LinearWeight* w2 = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnDown, i),
                {resources_.program_.hidden, intermediate});
            common_layer.feed_forward = DenseFfnWeights{w13, w2};
        }

        if (bind_cuda_attention_layer(
                *this, repo, semantic_layer, i, common_layer, shared_owner)) {
            continue;
        }
        if (const auto* gated_delta =
                std::get_if<GatedDeltaNetSpec>(&semantic_layer.mixer)) {
            GatedDeltaNetLayer gated_delta_layer;
            gated_delta_layer.common = common_layer;
            gated_delta_layer.spec = *gated_delta;
            const GatedDeltaNetSpec& spec = gated_delta_layer.spec;
            const int key_width = spec.key_heads * spec.key_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            const int qkv_width = 2 * key_width + value_width;
            if (spec.factorized_projections) {
                gated_delta_layer.q = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetQuery, i),
                    {key_width, resources_.program_.hidden});
                gated_delta_layer.k = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetKey, i),
                    {key_width, resources_.program_.hidden});
                gated_delta_layer.v = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetValue, i),
                    {value_width, resources_.program_.hidden});
                gated_delta_layer.z = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetOutputGate, i),
                    {value_width, resources_.program_.hidden});
            } else {
                gated_delta_layer.qkv = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetQkv, i),
                    {qkv_width, resources_.program_.hidden});
                gated_delta_layer.z = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetZ, i),
                    {value_width, resources_.program_.hidden});
            }
            gated_delta_layer.b = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetBeta, i),
                {spec.value_heads, resources_.program_.hidden});
            gated_delta_layer.a = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  spec.factorized_projections
                                      ? TensorRole::GatedDeltaNetDecay
                                      : TensorRole::GatedDeltaNetAlpha, i),
                {spec.decay_width(), resources_.program_.hidden});
            if (spec.factorized_projections) {
                const auto* q_conv = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetQueryConv, i),
                    {key_width, 1, spec.conv_kernel});
                const auto* k_conv = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetKeyConv, i),
                    {key_width, 1, spec.conv_kernel});
                const auto* v_conv = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetValueConv, i),
                    {value_width, 1, spec.conv_kernel});
                gated_delta_layer.factorized_conv_weight.reset(
                    static_cast<size_t>(qkv_width) * spec.conv_kernel);
                const size_t q_bytes = static_cast<size_t>(key_width) * spec.conv_kernel *
                    sizeof(__nv_bfloat16);
                const size_t v_bytes = static_cast<size_t>(value_width) * spec.conv_kernel *
                    sizeof(__nv_bfloat16);
                CELEG_CUDA(cudaMemcpy(gated_delta_layer.factorized_conv_weight.data(),
                    q_conv, q_bytes, cudaMemcpyDeviceToDevice));
                CELEG_CUDA(cudaMemcpy(gated_delta_layer.factorized_conv_weight.data() +
                    key_width * spec.conv_kernel, k_conv, q_bytes,
                    cudaMemcpyDeviceToDevice));
                CELEG_CUDA(cudaMemcpy(gated_delta_layer.factorized_conv_weight.data() +
                    2 * key_width * spec.conv_kernel, v_conv, v_bytes,
                    cudaMemcpyDeviceToDevice));
                gated_delta_layer.conv_weight = gated_delta_layer.factorized_conv_weight.data();
            } else {
                gated_delta_layer.conv_weight = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::GatedDeltaNetConv, i),
                    {qkv_width, 1, spec.conv_kernel});
            }
            gated_delta_layer.dt_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetDtBias, i),
                {spec.decay_width()});
            gated_delta_layer.a_log = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetALog, i),
                {spec.value_heads});
            gated_delta_layer.norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetNorm, i),
                {spec.value_head_dim});
            gated_delta_layer.out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetOutput, i),
                {resources_.program_.hidden, value_width});
            const int conv_dim = qkv_width;
            gated_delta_layer.conv_state.reset(static_cast<size_t>(conv_dim) * spec.conv_kernel);
            gated_delta_layer.recurrent_state.reset(static_cast<size_t>(spec.value_heads) *
                spec.key_head_dim * spec.value_head_dim);
            resources_.layers_.emplace_back(std::move(gated_delta_layer));
        } else if (const auto* mamba =
                       std::get_if<Mamba2Spec>(&semantic_layer.mixer)) {
            Mamba2Layer mamba_layer;
            mamba_layer.common = common_layer;
            mamba_layer.spec = *mamba;
            const Mamba2Spec& spec = mamba_layer.spec;
            const int conv_dim = spec.intermediate_size +
                2 * spec.group_count * spec.state_size;
            mamba_layer.in = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Input, i),
                {2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                 spec.num_heads, resources_.program_.hidden});
            mamba_layer.conv_weight = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Conv, i),
                {conv_dim, 1, spec.conv_kernel});
            mamba_layer.conv_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2ConvBias, i),
                {conv_dim});
            mamba_layer.dt_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2DtBias, i),
                {spec.num_heads});
            mamba_layer.a_log = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2ALog, i),
                {spec.num_heads});
            mamba_layer.d = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2D, i),
                {spec.num_heads});
            mamba_layer.norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Norm, i),
                {spec.intermediate_size});
            mamba_layer.out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Output, i),
                {resources_.program_.hidden, spec.intermediate_size});
            mamba_layer.conv_state.reset(static_cast<size_t>(conv_dim) * spec.conv_kernel);
            mamba_layer.ssm_state.reset(static_cast<size_t>(spec.intermediate_size) * spec.state_size);
            resources_.layers_.emplace_back(std::move(mamba_layer));
        } else if (const auto* mlp =
                       std::get_if<MlpBlockSpec>(&semantic_layer.mixer)) {
            MlpOnlyLayer mlp_layer;
            mlp_layer.common = common_layer;
            mlp_layer.spec = *mlp;
            mlp_layer.up = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnUp, i),
                {mlp_layer.spec.intermediate_size, resources_.program_.hidden});
            mlp_layer.down = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnDown, i),
                {resources_.program_.hidden, mlp_layer.spec.intermediate_size});
            resources_.layers_.emplace_back(std::move(mlp_layer));
        } else {
            ConvolutionLayer convolution_layer;
            convolution_layer.common = common_layer;
            convolution_layer.spec = std::get<ShortConvolutionSpec>(resources_.program_.layers[i].mixer);
            convolution_layer.conv_in = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::ShortConvInput, i),
                {3 * resources_.program_.hidden, resources_.program_.hidden});
            convolution_layer.conv_weight = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::ShortConvKernel, i),
                {resources_.program_.hidden, 1, convolution_layer.spec.cache_length});
            convolution_layer.conv_out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::ShortConvOutput, i),
                {resources_.program_.hidden, resources_.program_.hidden});
            convolution_layer.conv_state.reset(
                static_cast<size_t>(convolution_layer.spec.cache_length) * resources_.program_.hidden);
            resources_.layers_.emplace_back(std::move(convolution_layer));
        }
    }
    load_mtp_weights(*this, repo);
        });
}

}
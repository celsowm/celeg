#include "detail/compiled_model.hpp"
#include "checkpoint/detail/bootstrap.hpp"
#include "backend/cuda/weight_setup_support.hpp"
#include "backend/cuda/weight_setup.hpp"
#include "attention_weight_setup.hpp"
#include "moe_weight_setup.hpp"
#include "non_attention_weight_setup.hpp"

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
        bind_cuda_non_attention_layer(
            *this, repo, semantic_layer, i, common_layer);
    }
    load_mtp_weights(*this, repo);
        });
}

}

#include "layer_weight_setup.hpp"

#include "detail/compiled_model.hpp"
#include "backend/cuda/weight_setup_support.hpp"
#include "moe_weight_setup.hpp"

#include <stdexcept>
#include <string>

namespace celeg {

void prepare_cuda_layer_weight_resources(CudaCompiledModel& model) {
    configure_cuda_expert_resources(model);

    CudaModelResources& resources = model.resources_;
    CudaWorkspace& workspace = model.workspace_;
    const int mtp_layer_count = resources.options().enable_mtp
        ? resources.dims().mtp_num_hidden_layers
        : 0;
    const int resource_layer_count =
        resources.shape().num_hidden_layers + mtp_layer_count;

    workspace.expert_caches_.resize(static_cast<size_t>(resource_layer_count));
    if (resources.weights_->expert_controllers.size() <
        static_cast<size_t>(resource_layer_count)) {
        resources.weights_->expert_controllers.resize(
            static_cast<size_t>(resource_layer_count));
    }
    workspace.expert_catalog_.resize(static_cast<size_t>(resource_layer_count));
    if (resources.weights_->expert_catalog.size() <
        static_cast<size_t>(resource_layer_count)) {
        resources.weights_->expert_catalog.resize(
            static_cast<size_t>(resource_layer_count));
    }
}

LayerCommon bind_cuda_layer_common(CudaCompiledModel& model,
                                   const IWeightRepository& repo,
                                   const CompiledLayerProgram& semantics,
                                   int layer_index) {
    CudaModelResources& resources = model.resources_;
    LayerCommon common;
    const bool mixer_only_layer =
        std::holds_alternative<std::monostate>(semantics.feed_forward);

    const auto load_norm = [&](TensorRole role, const NormSpec& spec) {
        const std::string name = spec.weightless()
            ? std::string{}
            : cuda_tensor_name(resources.model_.weight_plan.requests,
                               role, layer_index);
        return resources.weight_loader_->load_rms_norm_weight(
            repo, name, {resources.program_.hidden}, spec.weight_kind);
    };

    if (semantics.mixer_norm.before) {
        common.mixer_norm_before = load_norm(
            TensorRole::AttentionInputNorm, *semantics.mixer_norm.before);
    }
    if (semantics.mixer_norm.after) {
        common.mixer_norm_after = load_norm(
            TensorRole::AttentionPostNorm, *semantics.mixer_norm.after);
    }
    if (!mixer_only_layer && semantics.feed_forward_norm.before) {
        common.feed_forward_norm_before = load_norm(
            TensorRole::FfnInputNorm, *semantics.feed_forward_norm.before);
    }
    if (!mixer_only_layer && semantics.feed_forward_norm.after) {
        common.feed_forward_norm_after = load_norm(
            TensorRole::FfnOutputNorm, *semantics.feed_forward_norm.after);
    }

    if (resources.program_.per_layer_input.enabled) {
        common.per_layer_input_gate = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::PerLayerInputGate, layer_index),
            {resources.program_.per_layer_input.input_size,
             resources.program_.hidden});
        common.per_layer_projection = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::PerLayerProjection, layer_index),
            {resources.program_.hidden,
             resources.program_.per_layer_input.input_size});
        common.per_layer_input_norm = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::PerLayerInputNorm, layer_index),
            {resources.program_.hidden});
        common.layer_scalar = resources.weight_loader_->load_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::LayerScalar, layer_index),
            {1});
    }

    if (mixer_only_layer) {
        common.feed_forward = std::monostate{};
        return common;
    }

    if (const auto* moe = std::get_if<MoeLayerProgram>(&semantics.feed_forward)) {
        bind_cuda_moe_feed_forward(model, repo, *moe, layer_index, common);
        return common;
    }

    const auto* dense =
        std::get_if<CompiledDenseFeedForwardProgram>(&semantics.feed_forward);
    if (!dense || dense->intermediate_size <= 0) {
        throw std::runtime_error("compiled dense layer has no FFN width");
    }
    const int intermediate = dense->intermediate_size;
    const LinearWeight* w13 = resources.weight_loader_->load_concat_linear_weight(
        repo, cuda_layer_name(layer_index, "feed_forward.w13.weight"),
        {
            {cuda_tensor_name(resources.model_.weight_plan.requests,
                              TensorRole::FfnGate, layer_index),
             {intermediate, resources.program_.hidden}},
            {cuda_tensor_name(resources.model_.weight_plan.requests,
                              TensorRole::FfnUp, layer_index),
             {intermediate, resources.program_.hidden}},
        });
    const LinearWeight* w2 = resources.weight_loader_->load_linear_weight(
        repo,
        cuda_tensor_name(resources.model_.weight_plan.requests,
                         TensorRole::FfnDown, layer_index),
        {resources.program_.hidden, intermediate});
    common.feed_forward = DenseFfnWeights{w13, w2};
    return common;
}

}

#include "celeg/model/program.hpp"

#include "program_fingerprint.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {

namespace {

CompiledAttentionStateLayout lower_attention_state_layout(
    const AttentionSpec& attention) {
    CompiledAttentionStateLayout state_layout;
    state_layout.storage = attention.state_storage;
    std::visit([&](const auto& state) {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, OrdinaryKvStateSpec>) {
            state_layout.kind = CompiledStateLayoutKind::OrdinaryKv;
            state_layout.key_width = attention.key_value_width();
            state_layout.value_width = attention.key_value_width();
            state_layout.key_elements =
                static_cast<std::size_t>(state_layout.key_width);
            state_layout.value_elements =
                static_cast<std::size_t>(state_layout.value_width);
        } else if constexpr (std::is_same_v<State, LatentAttentionStateSpec>) {
            state_layout.kind = CompiledStateLayoutKind::Latent;
            state_layout.latent_width = 2 * state.latent_rank;
            state_layout.rotary_width = state.decoupled_rope
                ? state.rope_head_dim : 0;
            state_layout.latent_elements =
                static_cast<std::size_t>(state_layout.latent_width);
            state_layout.rotary_elements =
                static_cast<std::size_t>(state_layout.rotary_width);
        }
    }, attention.state);
    state_layout.persistent_elements = state_layout.key_elements +
        state_layout.value_elements + state_layout.latent_elements +
        state_layout.rotary_elements;
    state_layout.validate();
    return state_layout;
}

CompiledMixerProgram lower_mixer(const LayerSpec& layer) {
    return std::visit([](const auto& mixer) -> CompiledMixerProgram {
        using Mixer = std::decay_t<decltype(mixer)>;
        if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
            if (!mixer.rope_position() &&
                !std::holds_alternative<NoPositionEncodingSpec>(mixer.position)) {
                throw std::invalid_argument(
                    "model program does not implement this position policy");
            }
            return CompiledAttentionProgram{
                mixer, lower_attention_state_layout(mixer)};
        } else if constexpr (
            std::is_same_v<Mixer, ShortConvolutionSpec> ||
            std::is_same_v<Mixer, GatedDeltaNetSpec> ||
            std::is_same_v<Mixer, Mamba2Spec> ||
            std::is_same_v<Mixer, MlpBlockSpec>) {
            return mixer;
        } else {
            static_assert(always_false_v<Mixer>,
                          "unhandled mixer lowering variant");
        }
    }, layer.mixer);
}

MoeLayerProgram lower_moe(const MixtureOfExpertsSpec& moe, int hidden) {
    MoeLayerProgram semantic;
    semantic.router.expert_count = moe.num_experts;
    semantic.router.experts_per_token = moe.experts_per_token;
    semantic.router.normalization = moe.normalize_topk
        ? MoeNormalizationKind::SumSelected : MoeNormalizationKind::None;
    semantic.router.score = moe.router_softmax
        ? MoeRouterScoreKind::SoftmaxLogits
        : MoeRouterScoreKind::SigmoidProbabilities;
    semantic.router.has_expert_bias = moe.use_expert_bias;
    semantic.router.routed_scaling = moe.routed_scaling_factor;
    semantic.router.selection = moe.routing_group_count > 0
        ? MoeSelectionKind::GroupedTopK : MoeSelectionKind::TopK;
    semantic.router.group_count = moe.routing_group_count;
    semantic.router.experts_per_group = moe.routing_experts_per_group;
    semantic.router.groups_per_token = moe.routing_groups_per_token;
    semantic.router.group_score_top_k = moe.routing_group_score_top_k;
    semantic.routed.mlp.hidden_size = hidden;
    semantic.routed.mlp.intermediate_size = moe.intermediate_size;

    const std::size_t expert_matrix_elements =
        static_cast<std::size_t>(hidden) *
        static_cast<std::size_t>(moe.intermediate_size);
    semantic.routed.payload.regions = {
        {TensorRole::MoeExpertGate, expert_matrix_elements},
        {TensorRole::MoeExpertUp, expert_matrix_elements},
        {TensorRole::MoeExpertDown, expert_matrix_elements}};
    semantic.output.has_shared_expert = moe.has_shared_expert;
    semantic.output.combine_order = moe.shared_before_routed
        ? MoeCombineOrder::SharedThenRouted : MoeCombineOrder::RoutedThenShared;
    if (moe.has_shared_expert) {
        semantic.shared = SharedExpertProgram{
            ExpertMlpProgram{MoeActivation::SwiGLU, hidden,
                             moe.shared_intermediate_size > 0
                                 ? moe.shared_intermediate_size
                                 : moe.intermediate_size}};
    }
    semantic.residency.expert_count = moe.num_experts;
    semantic.residency.payload_elements = 3 * expert_matrix_elements;
    return semantic;
}

CompiledFeedForwardProgram lower_feed_forward(const LayerSpec& layer, int hidden) {
    CompiledFeedForwardProgram compiled;
    if (!layer.execute_feed_forward) {
        compiled.storage() = std::monostate{};
        return compiled;
    }

    std::visit([&](const auto& feed_forward) {
        using FeedForward = std::decay_t<decltype(feed_forward)>;
        if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
            compiled.storage() = CompiledDenseFeedForwardProgram{
                feed_forward.intermediate_size, feed_forward.activation};
        } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
            compiled.storage() = lower_moe(feed_forward, hidden);
        } else {
            static_assert(always_false_v<FeedForward>,
                          "unhandled feed-forward lowering variant");
        }
    }, layer.feed_forward);
    return compiled;
}

void validate_mlp_only_source(const LayerSpec& layer,
                              const MlpBlockSpec& mlp) {
    std::visit([&](const auto& feed_forward) {
        if (feed_forward.intermediate_size != mlp.intermediate_size) {
            throw std::invalid_argument(
                "MLP-only mixer and feed-forward widths do not agree");
        }
    }, layer.feed_forward);
}

} // namespace

CompiledModelProgram build_model_program(const ResolvedModel& model) {
    if (model.graph.layers.empty()) throw std::invalid_argument("model has no layers");

    CompiledModelProgram program;
    program.hidden = model.graph.hidden;
    program.identity = model.provenance.identity;
    program.norm_after_layers = model.graph.norm_after_layers;
    program.per_layer_input = PerLayerInputPlan::derive(model);
    program.final_norm = model.graph.final_norm;
    program.embedding_transform = model.graph.embedding_transform;
    program.logits_multiplier = model.graph.logits_multiplier;
    program.logits_divisor = model.graph.logits_divisor;
    program.final_logit_softcap = model.graph.final_logit_softcap;
    program.layers.reserve(model.graph.layers.size());

    program.weight_request_count = model.weight_plan.requests.size();
    for (std::size_t request_index = 0;
         request_index < model.weight_plan.requests.size(); ++request_index) {
        if (model.weight_plan.requests[request_index].layer < 0) {
            program.unlayered_weight_request_indices.push_back(request_index);
        }
    }

    for (std::size_t layer_index = 0;
         layer_index < model.graph.layers.size(); ++layer_index) {
        const LayerSpec& layer = model.graph.layers[layer_index];

        CompiledLayerProgram compiled;
        compiled.mixer = lower_mixer(layer);
        compiled.feed_forward = lower_feed_forward(layer, model.graph.hidden);
        compiled.operator_norm = layer.operator_norm;
        compiled.post_attention_norm = layer.post_attention_norm;
        compiled.feed_forward_norm = layer.feed_forward_norm;
        compiled.post_feed_forward_norm = layer.post_feed_forward_norm;
        compiled.residual = layer.residual;

        if (layer.mixer_kind() == MixerKind::GatedDeltaNet ||
            layer.mixer_kind() == MixerKind::Mamba2) {
            compiled.chunk_capability = CompiledChunkCapability::SequentialAdapter;
        }

        if (const auto* mlp = compiled.mlp_only()) {
            validate_mlp_only_source(layer, *mlp);
        }

        for (std::size_t request_index = 0;
             request_index < model.weight_plan.requests.size(); ++request_index) {
            if (model.weight_plan.requests[request_index].layer ==
                static_cast<int>(layer_index)) {
                compiled.weight_request_indices.push_back(request_index);
            }
        }
        if (compiled.weight_request_indices.empty()) {
            throw std::invalid_argument("layer has no resolved weight requests");
        }

        program.layers.push_back(std::move(compiled));
    }

    program.semantic_fingerprint =
        detail::program_semantic_fingerprint(model.graph);
    program.validate();
    return program;
}

} // namespace celeg
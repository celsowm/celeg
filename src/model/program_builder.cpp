#include "celeg/model/program.hpp"

#include "program_fingerprint.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {
namespace {

CompiledAttentionStateLayout lower_attention_state_layout(
    const AttentionSpec& attention) {
    return std::visit([&](const auto& state) -> CompiledAttentionStateLayout {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, OrdinaryKvStateSpec>) {
            CompiledOrdinaryKvStateLayout layout{
                attention.key_value_width(),
                attention.key_value_width(),
                state.storage};
            layout.validate();
            return layout;
        } else if constexpr (std::is_same_v<State, LatentAttentionStateSpec>) {
            CompiledLatentStateLayout layout{
                2 * state.latent_rank,
                state.decoupled_rope ? state.rope_head_dim : 0,
                state.storage};
            layout.validate();
            return layout;
        } else {
            static_assert(always_false_v<State>,
                          "unhandled attention state lowering variant");
        }
    }, attention.state);
}

CompiledAttentionExecution lower_attention_execution(const AttentionSpec& attention) {
    CompiledAttentionExecution result;
    result.has_key_value = !std::holds_alternative<SharedKvConsumer>(attention.kv_sharing);
    result.has_query_key_norm = attention.has_query_key_norm();
    result.has_rope = attention.rope_position() != nullptr;
    if (const RopePositionSpec* rope = attention.rope_position()) {
        result.rope_pairing = rope->pairing;
    }
    if (attention.output_gate) {
        result.gate_granularity = attention.output_gate->granularity;
    }
    if (const LatentAttentionStateSpec* latent = attention.latent_state()) {
        result.kind = latent->factorized()
            ? AttentionExecutionKind::FactorizedLatent
            : AttentionExecutionKind::Latent;
        result.has_decoupled_rope = latent->decoupled_rope;
        result.rotary_width = latent->decoupled_rope ? latent->rope_head_dim : 0;
    } else {
        result.rotary_width = attention.rope_position()
            ? static_cast<int>(static_cast<double>(attention.head_dim) *
                               attention.rope_position()->rotary_fraction)
            : 0;
    }
    result.validate();
    return result;
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
                mixer, lower_attention_state_layout(mixer),
                lower_attention_execution(mixer)};
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
    semantic.router.selection = moe.selection;
    semantic.routed.mlp.hidden_size = hidden;
    semantic.routed.mlp.intermediate_size = moe.intermediate_size;

    const std::size_t expert_matrix_elements =
        static_cast<std::size_t>(hidden) *
        static_cast<std::size_t>(moe.intermediate_size);
    semantic.routed.payload.regions = {
        {TensorRole::MoeExpertGate, expert_matrix_elements},
        {TensorRole::MoeExpertUp, expert_matrix_elements},
        {TensorRole::MoeExpertDown, expert_matrix_elements}};

    if (moe.shared) {
        semantic.shared = SharedExpertProgram{
            ExpertMlpProgram{MoeActivation::SwiGLU, hidden,
                             moe.shared->intermediate_size},
            moe.shared->combine_order};
    }
    return semantic;
}

CompiledFeedForwardProgram lower_feed_forward(const LayerSpec& layer, int hidden) {
    return std::visit([&](const auto& feed_forward) -> CompiledFeedForwardProgram {
        using FeedForward = std::decay_t<decltype(feed_forward)>;
        if constexpr (std::is_same_v<FeedForward, std::monostate>) {
            return std::monostate{};
        } else if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
            return CompiledDenseFeedForwardProgram{
                feed_forward.intermediate_size, feed_forward.activation};
        } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
            return lower_moe(feed_forward, hidden);
        } else {
            static_assert(always_false_v<FeedForward>,
                          "unhandled feed-forward lowering variant");
        }
    }, layer.feed_forward);
}

}

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
        compiled.mixer_norm = layer.mixer_norm;
        compiled.feed_forward_norm = layer.feed_forward_norm;
        compiled.residual = layer.residual;

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

}

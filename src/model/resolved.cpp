#include "celeg/model/resolved.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {

namespace {

void append_norm(std::ostringstream& out, const NormSpec& norm) {
    out << norm.epsilon << ':' << static_cast<int>(norm.weight_kind) << ';';
}

void append_rope(std::ostringstream& out, const RopePositionSpec& rope) {
    out << rope.theta << ':' << rope.rotary_fraction << ':'
        << static_cast<int>(rope.pairing) << ':'
        << static_cast<int>(rope.scaling.kind) << ':' << rope.scaling.factor << ':'
        << rope.scaling.original_context << ':' << rope.scaling.attention_factor << ':'
        << rope.scaling.beta_fast << ':' << rope.scaling.beta_slow << ':'
        << rope.scaling.low_frequency_factor << ':' << rope.scaling.high_frequency_factor;
    out << ":short=";
    for (float value : rope.scaling.short_factors) out << value << ',';
    out << ":long=";
    for (float value : rope.scaling.long_factors) out << value << ',';
}

void append_position(std::ostringstream& out, const PositionSpec& position) {
    std::visit([&out](const auto& value) {
        using Position = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Position, NoPositionEncodingSpec>) {
            out << "none;";
        } else if constexpr (std::is_same_v<Position, RopePositionSpec>) {
            out << "rope:";
            append_rope(out, value);
            out << ';';
        } else if constexpr (std::is_same_v<Position, MultiAxisRopeSpec>) {
            out << "multi:";
            append_rope(out, value.base);
            out << ':' << value.interleaved << ':' << value.axes << ':';
            for (int section : value.sections) out << section << ',';
            out << ';';
        } else {
            static_assert(always_false_v<Position>, "unhandled position variant");
        }
    }, position);
}

void append_attention(std::ostringstream& out, const AttentionSpec& attention) {
    out << attention.query_heads << ':' << attention.key_value_heads << ':'
        << attention.head_dim << ':' << attention.kv_sharing.group << ':'
        << attention.kv_sharing.publishes << ':' << attention.query_scale << ':';
    append_norm(out, attention.query_norm);
    append_norm(out, attention.key_norm);
    append_position(out, attention.position);
    out << "pattern:";
    std::visit([&out](const auto& pattern) {
        using Pattern = std::decay_t<decltype(pattern)>;
        if constexpr (std::is_same_v<Pattern, FullCausalPattern>) {
            out << "causal";
        } else if constexpr (std::is_same_v<Pattern, SlidingWindowPattern>) {
            out << "sliding:" << pattern.window;
        } else if constexpr (std::is_same_v<Pattern, BidirectionalPattern>) {
            out << "bidirectional";
        } else if constexpr (std::is_same_v<Pattern, PrefixLmPattern>) {
            out << "prefix:" << pattern.prefix_length;
        } else if constexpr (std::is_same_v<Pattern, BlockSparsePattern>) {
            out << "block:" << pattern.block_size << ':' << pattern.local_blocks << ':'
                << pattern.global_blocks;
        } else if constexpr (std::is_same_v<Pattern, DynamicSparsePattern>) {
            out << "dynamic:" << pattern.block_size << ':'
                << pattern.max_selected_blocks;
        } else {
            static_assert(always_false_v<Pattern>, "unhandled attention pattern variant");
        }
    }, attention.pattern);
    out << ":gate:" << static_cast<int>(attention.output_gate.kind) << ':'
        << attention.output_gate.packed_with_query << ':'
        << static_cast<int>(attention.output_gate.granularity);
    out << ":bias:";
    std::visit([&out](const auto& bias) {
        using Bias = std::decay_t<decltype(bias)>;
        if constexpr (std::is_same_v<Bias, NoAttentionBiasSpec>) {
            out << "none";
        } else if constexpr (std::is_same_v<Bias, AlibiBiasSpec>) {
            out << "alibi:";
            for (float slope : bias.slopes) out << slope << ',';
        } else if constexpr (std::is_same_v<Bias, RelativePositionBiasSpec>) {
            out << "relative:" << bias.bucket_count << ':' << bias.max_distance << ':'
                << bias.bidirectional;
        } else {
            static_assert(always_false_v<Bias>, "unhandled attention bias variant");
        }
    }, attention.bias);
    out << ":state:";
    std::visit([&out](const auto& state) {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, OrdinaryKvStateSpec>) {
            out << "ordinary:" << state.quantizable;
        } else if constexpr (std::is_same_v<State, LatentAttentionStateSpec>) {
            out << "latent:" << state.latent_rank << ':' << state.rope_head_dim << ':'
                << state.nope_head_dim << ':' << state.decoupled_rope << ':'
                << state.factorized << ':' << state.query_rank << ':'
                << state.value_head_dim;
            append_norm(out, state.query_latent_norm);
            append_norm(out, state.key_latent_norm);
        } else {
            static_assert(always_false_v<State>, "unhandled attention state variant");
        }
    }, attention.state);
    out << ":storage:" << static_cast<int>(attention.state_storage.key) << ':'
        << static_cast<int>(attention.state_storage.value) << ':'
        << static_cast<int>(attention.state_storage.latent) << ':'
        << static_cast<int>(attention.state_storage.rotary) << ':'
        << static_cast<int>(attention.state_storage.recurrent) << ':'
        << static_cast<int>(attention.state_storage.granularity) << ':'
        << attention.state_storage.paged;
    out << ":source:" << static_cast<int>(attention.sources.query) << ':'
        << static_cast<int>(attention.sources.key_value) << ':' << attention.sources.memory_slot;
    out << ":transform:";
    std::visit([&out](const auto& transform) {
        using Transform = std::decay_t<decltype(transform)>;
        if constexpr (std::is_same_v<Transform, NoAttentionOutputTransformSpec>) {
            out << "none";
        } else if constexpr (std::is_same_v<Transform, OrthogonalizeCurrentValueSpec>) {
            out << "orthogonalize:" << transform.minimum_norm_squared;
        } else {
            static_assert(always_false_v<Transform>, "unhandled attention transform variant");
        }
    }, attention.output_transform);
}

void append_mixer(std::ostringstream& out, const LayerSpec& layer) {
    std::visit([&out](const auto& mixer) {
        using Mixer = std::decay_t<decltype(mixer)>;
        if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
            out << "attention:";
            append_attention(out, mixer);
        } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
            out << "short-conv:" << mixer.cache_length << ':' << mixer.channels << ':'
                << mixer.bias;
        } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
            out << "gdn:" << mixer.conv_kernel << ':' << mixer.key_head_dim << ':'
                << mixer.value_head_dim << ':' << mixer.key_heads << ':' << mixer.value_heads
                << ':' << mixer.vector_decay << ':' << mixer.safe_decay << ':'
                << mixer.decay_lower_bound << ':' << mixer.sigmoid_output_gate << ':'
                << mixer.factorized_projections;
        } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
            out << "mamba2:" << mixer.conv_kernel << ':' << mixer.intermediate_size << ':'
                << mixer.state_size << ':' << mixer.time_step_rank << ':' << mixer.num_heads
                << ':' << mixer.head_dim << ':' << mixer.group_count << ':' << mixer.chunk_size
                << ':' << mixer.conv_bias << ':' << mixer.projection_bias;
        } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
            out << "mlp-only:" << mixer.intermediate_size << ':'
                << static_cast<int>(mixer.activation);
        } else {
            static_assert(always_false_v<Mixer>, "unhandled mixer fingerprint variant");
        }
    }, layer.mixer);
}

void append_feed_forward(std::ostringstream& out, const LayerSpec& layer) {
    std::visit([&out](const auto& feed_forward) {
        using FeedForward = std::decay_t<decltype(feed_forward)>;
        if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
            out << "dense:" << feed_forward.intermediate_size << ':'
                << static_cast<int>(feed_forward.activation);
        } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
            out << "moe:" << feed_forward.intermediate_size << ':' << feed_forward.num_experts
                << ':' << feed_forward.experts_per_token << ':' << feed_forward.normalize_topk
                << ':' << feed_forward.use_expert_bias << ':'
                << feed_forward.routed_scaling_factor << ':'
                << feed_forward.routing_group_count << ':'
                << feed_forward.routing_experts_per_group << ':'
                << feed_forward.routing_groups_per_token << ':'
                << feed_forward.routing_group_score_top_k << ':'
                << feed_forward.has_shared_expert << ':'
                << feed_forward.shared_intermediate_size << ':'
                << feed_forward.shared_before_routed << ':' << feed_forward.router_softmax;
        } else {
            static_assert(always_false_v<FeedForward>,
                          "unhandled feed-forward fingerprint variant");
        }
    }, layer.feed_forward);
}

} // namespace

std::string ModelGraph::fingerprint() const {
    std::ostringstream out;
    out << "hidden=" << hidden << ":final=";
    append_norm(out, final_norm);
    out << ":norm-boundaries=";
    for (int layer : norm_after_layers) out << layer << ',';
    out << ":embedding=" << embedding_transform.multiplier << ':'
        << embedding_transform.post_norm.has_value();
    if (embedding_transform.post_norm) append_norm(out, *embedding_transform.post_norm);
    out << ":logits=" << logits_divisor << ':' << logits_multiplier << ':'
        << final_logit_softcap << ":layers=";
    for (const LayerSpec& layer : layers) {
        append_norm(out, layer.operator_norm);
        append_norm(out, layer.post_attention_norm);
        append_norm(out, layer.pre_feed_forward_norm);
        append_norm(out, layer.post_feed_forward_norm);
        append_norm(out, layer.per_layer_input_norm);
        append_mixer(out, layer);
        append_norm(out, layer.feed_forward_norm);
        append_feed_forward(out, layer);
        out << ":residual=" << layer.residual.multiplier << ":input="
            << layer.per_layer_input.input_size << ':'
            << static_cast<int>(layer.per_layer_input.activation) << ':'
            << layer.per_layer_input.enabled << ":scalar=" << layer.layer_scalar
            << ":execute=" << layer.execute_feed_forward << ';';
    }
    return out.str();
}

ExecutionTopology ExecutionTopology::derive(const ModelGraph& graph) {
    ExecutionTopology result;
    const std::size_t layer_count = graph.layers.size();
    if (layer_count == 0) {
        throw std::invalid_argument("cannot derive runtime shape from an empty graph");
    }
    result.num_hidden_layers = static_cast<int>(layer_count);
    result.attention_slot_for_layer.assign(layer_count, -1);
    result.layer_for_attention_slot.clear();
    result.attention_layer_count = 0;
    result.conv_layer_count = 0;
    result.gated_delta_net_layer_count = 0;
    result.mamba2_layer_count = 0;
    result.mlp_only_layer_count = 0;
    result.mamba2_intermediate = 0;
    result.max_feed_forward_intermediate = 0;
    int maximum_conv_cache = 0;
    int maximum_conv_dim = 0;
    for (std::size_t index = 0; index < layer_count; ++index) {
        const LayerSpec& layer = graph.layers[index];
        std::visit([&](const auto& mixer) {
            using Mixer = std::decay_t<decltype(mixer)>;
            if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
                result.maximum_attention_projection_width_value = std::max(
                    result.maximum_attention_projection_width_value,
                    mixer.projection_width());
                result.maximum_attention_query_heads_value = std::max(
                    result.maximum_attention_query_heads_value, mixer.query_heads);
                result.maximum_attention_head_dim_value = std::max(
                    result.maximum_attention_head_dim_value, mixer.head_dim);
                result.maximum_attention_output_width_value = std::max(
                    result.maximum_attention_output_width_value,
                    mixer.uses_latent_state() ? mixer.latent_query_content_width()
                                               : mixer.query_width());
                result.maximum_attention_latent_query_rope_width_value = std::max(
                    result.maximum_attention_latent_query_rope_width_value,
                    mixer.latent_query_rope_width());
                result.maximum_attention_latent_projection_width_value = std::max(
                    result.maximum_attention_latent_projection_width_value,
                    mixer.latent_query_projection_width());
                result.maximum_attention_latent_output_width_value = std::max(
                    result.maximum_attention_latent_output_width_value,
                    mixer.latent_output_width());
                if (const auto* latent = mixer.latent_state()) {
                    result.maximum_attention_latent_rank_value = std::max(
                        result.maximum_attention_latent_rank_value, latent->latent_rank);
                    result.maximum_attention_latent_rope_width_value = std::max(
                        result.maximum_attention_latent_rope_width_value,
                        latent->decoupled_rope ? latent->rope_head_dim : 0);
                }
                result.attention_slot_for_layer[index] = result.attention_layer_count++;
                result.layer_for_attention_slot.push_back(static_cast<int>(index));
            } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
                maximum_conv_cache = std::max(maximum_conv_cache, mixer.cache_length);
                maximum_conv_dim = std::max(maximum_conv_dim, mixer.channels);
                ++result.conv_layer_count;
            } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                result.maximum_gated_delta_net_qkv_width_value = std::max(
                    result.maximum_gated_delta_net_qkv_width_value, mixer.qkv_width());
                result.maximum_gated_delta_net_output_width_value = std::max(
                    result.maximum_gated_delta_net_output_width_value, mixer.value_width());
                result.maximum_gated_delta_net_gate_width_value = std::max(
                    result.maximum_gated_delta_net_gate_width_value,
                    std::max(mixer.value_heads, mixer.decay_width()));
                ++result.gated_delta_net_layer_count;
            } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                result.mamba2_intermediate = std::max(
                    result.mamba2_intermediate, mixer.intermediate_size);
                result.maximum_mamba_projection_width_value = std::max(
                    result.maximum_mamba_projection_width_value,
                    2 * mixer.intermediate_size +
                    2 * mixer.group_count * mixer.state_size + mixer.num_heads);
                result.maximum_mamba_conv_width_value = std::max(
                    result.maximum_mamba_conv_width_value,
                    mixer.intermediate_size +
                    2 * mixer.group_count * mixer.state_size);
                ++result.mamba2_layer_count;
            } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
                result.max_feed_forward_intermediate = std::max(
                    result.max_feed_forward_intermediate, mixer.intermediate_size);
                ++result.mlp_only_layer_count;
            } else {
                static_assert(always_false_v<Mixer>, "unhandled mixer derivation variant");
            }
        }, layer.mixer);
        std::visit([&](const auto& feed_forward) {
            using FeedForward = std::decay_t<decltype(feed_forward)>;
            if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
                result.max_feed_forward_intermediate = std::max(
                    result.max_feed_forward_intermediate, feed_forward.intermediate_size);
            } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
                result.max_feed_forward_intermediate = std::max(
                    result.max_feed_forward_intermediate, feed_forward.intermediate_size);
            } else {
                static_assert(always_false_v<FeedForward>,
                              "unhandled feed-forward derivation variant");
            }
        }, layer.feed_forward);
    }
    result.conv_cache = maximum_conv_cache;
    result.conv_dim = maximum_conv_dim;
    return result;
}

RuntimeTopology compose_runtime_topology(CheckpointDimensions checkpoint,
                                         const ModelGraph& graph) {
    RuntimeTopology topology;
    topology.dims = std::move(checkpoint);
    topology.exec = ExecutionTopology::derive(graph);
    return topology;
}


void ModelGraph::validate() const {
    if (layers.empty()) {
        throw std::runtime_error("resolved model graph has no layers");
    }
    if (hidden <= 0 || !(final_norm.epsilon > 0.0f) || !std::isfinite(final_norm.epsilon) ||
        !std::isfinite(logits_divisor) || logits_divisor <= 0.0f ||
        !std::isfinite(logits_multiplier) ||
        !std::isfinite(final_logit_softcap) || final_logit_softcap < 0.0f) {
        throw std::runtime_error("resolved model graph has invalid policies");
    }
    final_norm.validate();
    embedding_transform.validate();
    for (size_t index = 0; index < norm_after_layers.size(); ++index) {
        const int layer = norm_after_layers[index];
        if (layer < 0 || layer >= static_cast<int>(layers.size()) - 1 ||
            (index > 0 && norm_after_layers[index - 1] >= layer)) {
            throw std::runtime_error("resolved model graph has invalid norm boundary");
        }
    }
    for (const LayerSpec& layer : layers) {
        if (!(layer.operator_norm.epsilon > 0.0f) ||
            !std::isfinite(layer.residual.multiplier) ||
            !std::isfinite(layer.layer_scalar)) {
            throw std::runtime_error("resolved model graph has invalid layer policy");
        }
        layer.operator_norm.validate();
        if (layer.feed_forward_norm.enabled()) layer.feed_forward_norm.validate();
        if (layer.post_attention_norm.enabled()) layer.post_attention_norm.validate();
        if (layer.pre_feed_forward_norm.enabled()) layer.pre_feed_forward_norm.validate();
        if (layer.post_feed_forward_norm.enabled()) layer.post_feed_forward_norm.validate();
        if (layer.per_layer_input.enabled && layer.per_layer_input.input_size <= 0) {
            throw std::runtime_error("enabled per-layer input has invalid width");
        }
        std::visit([](const auto& feed_forward) {
            using FeedForward = std::decay_t<decltype(feed_forward)>;
            if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec> ||
                          std::is_same_v<FeedForward, MlpBlockSpec>) {
                if (feed_forward.intermediate_size <= 0) {
                    throw std::runtime_error("dense layer has no positive FFN width");
                }
            } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
                if (feed_forward.intermediate_size <= 0 || feed_forward.num_experts <= 0 ||
                    feed_forward.experts_per_token <= 0 ||
                    feed_forward.experts_per_token > feed_forward.num_experts) {
                    throw std::runtime_error("MoE layer has invalid routed dimensions");
                }
                if (!std::isfinite(feed_forward.routed_scaling_factor) ||
                    feed_forward.routed_scaling_factor <= 0.0f) {
                    throw std::runtime_error("MoE layer has invalid routed scaling");
                }
                const bool grouped = feed_forward.routing_group_count > 0;
                if (grouped != (feed_forward.routing_experts_per_group > 0 &&
                                feed_forward.routing_groups_per_token > 0 &&
                                feed_forward.routing_group_score_top_k > 0)) {
                    throw std::runtime_error("MoE grouped routing fields are incomplete");
                }
                if (grouped &&
                    (feed_forward.routing_group_count *
                         feed_forward.routing_experts_per_group != feed_forward.num_experts ||
                     feed_forward.routing_groups_per_token >
                         feed_forward.routing_group_count ||
                     feed_forward.routing_group_score_top_k >
                         feed_forward.routing_experts_per_group)) {
                    throw std::runtime_error("MoE grouped routing fields are inconsistent");
                }
                if (feed_forward.has_shared_expert &&
                    feed_forward.shared_intermediate_size <= 0) {
                    throw std::runtime_error("MoE shared expert has no positive width");
                }
            } else {
                static_assert(always_false_v<FeedForward>,
                              "unhandled feed-forward semantic validation variant");
            }
        }, layer.feed_forward);
        if (const auto* attention = std::get_if<AttentionSpec>(&layer.mixer)) {
            if (attention->query_norm.enabled()) attention->query_norm.validate();
            if (attention->key_norm.enabled()) attention->key_norm.validate();
            switch (attention->output_gate.granularity) {
            case AttentionGateGranularity::OutputWise:
            case AttentionGateGranularity::HeadWise:
            case AttentionGateGranularity::ElementWise:
                break;
            default:
                throw std::runtime_error("invalid attention gate granularity");
            }
        }
    }
}

void ModelGraph::EmbeddingTransformSpec::validate() const {
    if (!std::isfinite(multiplier)) {
        throw std::runtime_error("embedding transform multiplier is invalid");
    }
    if (post_norm) post_norm->validate();
}

void AlibiBiasSpec::validate(int query_heads) const {
    if (query_heads <= 0 || slopes.size() != static_cast<size_t>(query_heads)) {
        throw std::runtime_error("ALiBi slope count does not match query heads");
    }
    for (float slope : slopes) {
        if (!(slope > 0.0f) || !std::isfinite(slope)) {
            throw std::runtime_error("ALiBi slopes must be finite and positive");
        }
    }
}

void RelativePositionBiasSpec::validate() const {
    if (bucket_count <= 0 || max_distance <= 0) {
        throw std::runtime_error("relative position bias dimensions must be positive");
    }
    if (bidirectional && (bucket_count < 2 || (bucket_count % 2) != 0)) {
        throw std::runtime_error(
            "bidirectional relative position bias requires an even bucket count");
    }
}

void AttentionStateStorageSpec::validate(const AttentionStateSpec& state) const {
    const auto validate_scalar = [](StateScalarType scalar) {
        switch (scalar) {
        case StateScalarType::FP32:
        case StateScalarType::FP16:
        case StateScalarType::BF16:
        case StateScalarType::FP8:
        case StateScalarType::INT8:
        case StateScalarType::INT4:
            return;
        }
        throw std::runtime_error("invalid attention state scalar type");
    };
    const auto validate_granularity = [](StateQuantizationGranularity granularity) {
        switch (granularity) {
        case StateQuantizationGranularity::PerTensor:
        case StateQuantizationGranularity::PerHead:
        case StateQuantizationGranularity::PerToken:
        case StateQuantizationGranularity::PerBlock:
            return;
        }
        throw std::runtime_error("invalid attention state quantization granularity");
    };
    validate_scalar(key);
    validate_scalar(value);
    validate_scalar(latent);
    validate_scalar(rotary);
    validate_scalar(recurrent);
    validate_granularity(granularity);
    if (const auto* latent_state = std::get_if<LatentAttentionStateSpec>(&state)) {
        if (latent_state->latent_rank <= 0 || latent_state->rope_head_dim < 0 ||
            latent_state->nope_head_dim < 0 ||
            latent_state->rope_head_dim + latent_state->nope_head_dim <= 0) {
            throw std::runtime_error("invalid latent attention state dimensions");
        }
        if (latent_state->factorized &&
            (latent_state->query_rank <= 0 || latent_state->value_head_dim <= 0 ||
             !latent_state->query_latent_norm.enabled() ||
             !latent_state->key_latent_norm.enabled())) {
            throw std::runtime_error("invalid factorized latent attention projections");
        }
    }
}

void TokenPolicy::validate() const {
    if (bos_token_id < 0 || eos_token_ids.empty() || pad_token_id < 0) {
        throw std::runtime_error("invalid resolved model token policy");
    }
}

void NumericalPolicy::validate() const {
    if (!(norm_eps > 0.0f) || !std::isfinite(norm_eps) ||
        (post_norm_eps != 0.0f && (!(post_norm_eps > 0.0f) || !std::isfinite(post_norm_eps))) ||
        !(logits_divisor > 0.0f) || !std::isfinite(logits_divisor) ||
        !std::isfinite(embedding_multiplier) ||
        !std::isfinite(attention_multiplier) ||
        !std::isfinite(residual_multiplier) ||
        !std::isfinite(logits_multiplier) ||
        final_logit_softcap < 0.0f || !std::isfinite(final_logit_softcap)) {
        throw std::runtime_error("invalid resolved model numerical policy");
    }
}

void CheckpointDimensions::validate() const {
    if (vocab_size <= 0 || max_position_embeddings <= 0 ||
        mtp_num_hidden_layers < 0) {
        throw std::runtime_error("invalid checkpoint dimensions");
    }
    if (!checkpoint_layer_for_layer.empty()) {
        for (int layer : checkpoint_layer_for_layer) {
            if (layer < 0) throw std::runtime_error("negative checkpoint layer mapping");
        }
    }
    token_policy.validate();
    const auto validate_token_id = [this](int token, const char* name) {
        if (token < 0 || token >= vocab_size) {
            throw std::runtime_error(std::string("resolved ") + name +
                                     " token id is outside the vocabulary");
        }
    };
    validate_token_id(token_policy.bos_token_id, "BOS");
    validate_token_id(token_policy.pad_token_id, "pad");
    for (int token : token_policy.eos_token_ids) validate_token_id(token, "EOS");
}

std::string ExecutionTopology::fingerprint() const {
    std::ostringstream out;
    out << "-l" << num_hidden_layers
        << "-cc" << conv_cache
        << "-ac" << attention_layer_count << "-conv" << conv_layer_count
        << "-gdn" << gated_delta_net_layer_count << "-m2" << mamba2_layer_count
        << "-mlp" << mlp_only_layer_count << "-m2i" << mamba2_intermediate
        << "-max-attn-proj" << maximum_attention_projection_width_value
        << "-max-attn-heads" << maximum_attention_query_heads_value
        << "-max-attn-dim" << maximum_attention_head_dim_value
        << "-max-attn-out" << maximum_attention_output_width_value
        << "-max-mamba-proj" << maximum_mamba_projection_width_value
        << "-max-gdn-qkv" << maximum_gated_delta_net_qkv_width_value
        << "-max-ff" << max_feed_forward_intermediate
        << "-ff" << max_feed_forward_intermediate;
    return out.str();
}

std::string ExecutionTopology::summary() const {
    std::ostringstream out;
    out << "layers=" << num_hidden_layers
        << " attention_layers=" << attention_layer_count
        << " conv_layers=" << conv_layer_count
        << " gated_delta_layers=" << gated_delta_net_layer_count
        << " mamba2_layers=" << mamba2_layer_count
        << " mlp_only_layers=" << mlp_only_layer_count;
    return out.str();
}

void ExecutionTopology::validate() const {
    if (num_hidden_layers <= 0) {
        throw std::runtime_error("invalid resolved model topology");
    }
    if (attention_layer_count + conv_layer_count + gated_delta_net_layer_count +
        mamba2_layer_count + mlp_only_layer_count != num_hidden_layers) {
        throw std::runtime_error("resolved layer counts are inconsistent");
    }
    if (max_feed_forward_intermediate <= 0 ||
        maximum_attention_projection_width_value < 0 ||
        maximum_gated_delta_net_qkv_width_value < 0 ||
        maximum_mamba_projection_width_value < 0) {
        throw std::runtime_error("resolved topology cache has invalid maxima");
    }
}

void ResolvedModel::validate() const {
    topology.validate();
    graph.validate();
    if (graph.layers.size() != static_cast<size_t>(topology.exec.num_hidden_layers)) {
        throw std::runtime_error("resolved graph/topology layer count mismatch");
    }
}

std::string RuntimeTopology::fingerprint() const {
    std::ostringstream out;
    out << exec.fingerprint()
        << "-mtp" << dims.mtp_num_hidden_layers
        << "-voc" << dims.vocab_size;
    out << "-map";
    for (int layer : dims.checkpoint_layer_for_layer) out << '-' << layer;
    return out.str();
}

std::string RuntimeTopology::summary() const {
    std::ostringstream out;
    out << exec.summary()
        << " mtp_layers=" << dims.mtp_num_hidden_layers
        << " vocab=" << dims.vocab_size;
    return out.str();
}

void RuntimeTopology::validate() const {
    dims.validate();
    exec.validate();
    if (!dims.checkpoint_layer_for_layer.empty() &&
        static_cast<int>(dims.checkpoint_layer_for_layer.size()) !=
            exec.num_hidden_layers) {
        throw std::runtime_error("checkpoint layer mapping length mismatch");
    }
}

} // namespace celeg

#include "celeg/model/resolved.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace celeg {

void derive_runtime_topology_from_graph(RuntimeTopology& topology,
                                        const ModelGraph& graph) {
    if (graph.layers.empty()) {
        throw std::invalid_argument("cannot derive runtime shape from an empty graph");
    }
    const std::size_t layer_count = graph.layers.size();
    topology.num_hidden_layers = static_cast<int>(layer_count);
    topology.mixer_kinds.resize(layer_count);
    topology.feed_forward_kinds.resize(layer_count);
    topology.execute_feed_forward.resize(layer_count);
    topology.feed_forward_intermediates.resize(layer_count);
    topology.feed_forward_activations.resize(layer_count);
    topology.attention_layouts.resize(layer_count);
    topology.gated_delta_net_layouts.clear();
    topology.mamba2_layouts.clear();
    topology.mlp_only_layouts.resize(layer_count);
    topology.attention_slot_for_layer.assign(layer_count, -1);
    topology.layer_for_attention_slot.clear();
    topology.attention_layer_count = 0;
    topology.conv_layer_count = 0;
    topology.gated_delta_net_layer_count = 0;
    topology.mamba2_layer_count = 0;
    topology.mlp_only_layer_count = 0;
    topology.num_dense_layers = 0;
    topology.mamba2_intermediate = 0;
    topology.max_feed_forward_intermediate = 0;
    topology.dense_intermediate = 0;
    topology.moe_intermediate = 0;
    topology.shared_expert_intermediate = 0;
    topology.num_experts = 0;
    topology.experts_per_token = 0;
    topology.normalize_topk = false;
    topology.moe_router_softmax = false;
    topology.use_expert_bias = false;
    topology.routed_scaling_factor = 1.0f;
    topology.has_per_layer_input = false;
    topology.per_layer_input_size = 0;
    int maximum_conv_cache = 0;
    int maximum_conv_dim = 0;
    for (std::size_t index = 0; index < layer_count; ++index) {
        const LayerSpec& layer = graph.layers[index];
        topology.mixer_kinds[index] = layer.mixer_kind();
        topology.feed_forward_kinds[index] = layer.feed_forward_kind();
        topology.execute_feed_forward[index] = layer.execute_feed_forward;
        std::visit([&](const auto& mixer) {
            using Mixer = std::decay_t<decltype(mixer)>;
            if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
                topology.attention_layouts[index] = mixer;
                topology.attention_slot_for_layer[index] = topology.attention_layer_count++;
                topology.layer_for_attention_slot.push_back(static_cast<int>(index));
            } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
                maximum_conv_cache = std::max(maximum_conv_cache, mixer.cache_length);
                maximum_conv_dim = std::max(maximum_conv_dim, mixer.channels);
                ++topology.conv_layer_count;
            } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                topology.gated_delta_net_layouts.resize(layer_count);
                topology.gated_delta_net_layouts[index] = mixer;
                ++topology.gated_delta_net_layer_count;
            } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                topology.mamba2_layouts.resize(layer_count);
                topology.mamba2_layouts[index] = mixer;
                topology.mamba2_intermediate = std::max(
                    topology.mamba2_intermediate, mixer.intermediate_size);
                ++topology.mamba2_layer_count;
            } else {
                topology.mlp_only_layouts[index] = mixer;
                ++topology.mlp_only_layer_count;
            }
        }, layer.mixer);
        std::visit([&](const auto& feed_forward) {
            using FeedForward = std::decay_t<decltype(feed_forward)>;
            if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
                topology.feed_forward_intermediates[index] = feed_forward.intermediate_size;
                topology.feed_forward_activations[index] = feed_forward.activation;
                ++topology.num_dense_layers;
                topology.dense_intermediate = std::max(
                    topology.dense_intermediate, feed_forward.intermediate_size);
            } else if constexpr (std::is_same_v<FeedForward, MlpBlockSpec>) {
                topology.feed_forward_intermediates[index] = feed_forward.intermediate_size;
                topology.feed_forward_activations[index] = feed_forward.activation;
                ++topology.num_dense_layers;
                topology.dense_intermediate = std::max(
                    topology.dense_intermediate, feed_forward.intermediate_size);
            } else {
                topology.feed_forward_intermediates[index] = feed_forward.intermediate_size;
                topology.feed_forward_activations[index] = ActivationKind::SwiGLU;
                topology.moe_intermediate = std::max(
                    topology.moe_intermediate, feed_forward.intermediate_size);
                topology.num_experts = std::max(topology.num_experts, feed_forward.num_experts);
                topology.experts_per_token = std::max(
                    topology.experts_per_token, feed_forward.experts_per_token);
                topology.normalize_topk = topology.normalize_topk || feed_forward.normalize_topk;
                topology.moe_router_softmax = topology.moe_router_softmax ||
                    feed_forward.router_softmax;
                topology.use_expert_bias = topology.use_expert_bias ||
                    feed_forward.use_expert_bias;
                topology.routed_scaling_factor = feed_forward.routed_scaling_factor;
                topology.shared_expert_intermediate = std::max(
                    topology.shared_expert_intermediate, feed_forward.shared_intermediate_size);
            }
        }, layer.feed_forward);
        if (layer.per_layer_input.enabled) {
            topology.has_per_layer_input = true;
            topology.per_layer_input_size = layer.per_layer_input.input_size;
        }
    }
    topology.conv_cache = maximum_conv_cache;
    topology.conv_dim = maximum_conv_dim;
    for (int width : topology.feed_forward_intermediates) {
        topology.max_feed_forward_intermediate = std::max(
            topology.max_feed_forward_intermediate, width);
    }
}

void ModelGraph::validate() const {
    if (layers.empty()) {
        throw std::runtime_error("resolved model graph has no layers");
    }
    if (!(final_norm.epsilon > 0.0f) || !std::isfinite(final_norm.epsilon) ||
        !std::isfinite(logits_divisor) || logits_divisor <= 0.0f ||
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
        if (const auto* attention = std::get_if<AttentionSpec>(&layer.mixer)) {
            if (attention->query_norm.enabled()) attention->query_norm.validate();
            if (attention->key_norm.enabled()) attention->key_norm.validate();
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

std::string RuntimeTopology::fingerprint() const {
    std::ostringstream out;
    out << "h" << hidden << "-l" << num_hidden_layers
        << "-mtp" << mtp_num_hidden_layers
        << "-voc" << vocab_size
        << "-int" << intermediate << "-cc" << conv_cache
        << "-e" << num_experts << "-k" << experts_per_token
        << "-mi" << moe_intermediate;
    out << "-m2i" << mamba2_intermediate;
    out << "-map";
    for (int layer : checkpoint_layer_for_layer) out << '-' << layer;
    for (const AttentionSpec& attention : attention_layouts) {
        if (const auto* rope = attention.rope_position()) {
            out << "-r" << rope->theta << '-' << rope->rotary_fraction
                << '-' << static_cast<int>(rope->scaling.kind)
                << '-' << rope->scaling.factor
                << '-' << rope->scaling.original_context;
        } else {
            out << "-rnone";
        }
        if (const auto* multi = attention.multi_axis_position()) {
            out << "-ma" << multi->interleaved;
            for (int section : multi->sections) out << '-' << section;
        }
        std::visit([&out](const auto& pattern) {
            using Pattern = std::decay_t<decltype(pattern)>;
            if constexpr (std::is_same_v<Pattern, FullCausalPattern>) {
                out << "-causal";
            } else if constexpr (std::is_same_v<Pattern, SlidingWindowPattern>) {
                out << "-slide" << pattern.window;
            } else if constexpr (std::is_same_v<Pattern, BidirectionalPattern>) {
                out << "-bidir";
            } else if constexpr (std::is_same_v<Pattern, PrefixLmPattern>) {
                out << "-prefix" << pattern.prefix_length;
            } else if constexpr (std::is_same_v<Pattern, BlockSparsePattern>) {
                out << "-block" << pattern.block_size << '-' << pattern.local_blocks
                    << '-' << pattern.global_blocks;
            } else if constexpr (std::is_same_v<Pattern, DynamicSparsePattern>) {
                out << "-dynamic" << pattern.block_size << '-'
                    << pattern.max_selected_blocks;
            }
        }, attention.pattern);
        out << "-state" << static_cast<int>(attention.state_storage.key) << '-'
            << static_cast<int>(attention.state_storage.value) << '-'
            << static_cast<int>(attention.state_storage.latent) << '-'
            << static_cast<int>(attention.state_storage.rotary) << '-'
            << static_cast<int>(attention.state_storage.recurrent) << '-'
            << static_cast<int>(attention.state_storage.granularity) << '-'
            << attention.state_storage.paged;
        std::visit([&out](const auto& state) {
            using State = std::decay_t<decltype(state)>;
            if constexpr (std::is_same_v<State, OrdinaryKvStateSpec>) {
                out << "-ordinary-kv";
            } else {
                out << "-latent" << state.latent_rank << '-'
                    << state.rope_head_dim << '-' << state.nope_head_dim << '-'
                    << state.decoupled_rope;
            }
        }, attention.state);
        out << "-src" << static_cast<int>(attention.sources.query) << '-'
            << static_cast<int>(attention.sources.key_value) << '-'
            << attention.sources.memory_slot;
    }
    for (MixerKind kind : mixer_kinds) {
        switch (kind) {
        case MixerKind::Attention: out << "-a"; break;
        case MixerKind::ShortConvolution: out << "-c"; break;
        case MixerKind::GatedDeltaNet: out << "-d"; break;
        case MixerKind::Mamba2: out << "-m2"; break;
        case MixerKind::MlpOnly: out << "-mlp"; break;
        }
    }
    for (const GatedDeltaNetSpec& layout : gated_delta_net_layouts) {
        out << "-gd" << layout.conv_kernel << '-' << layout.key_heads << 'x'
            << layout.key_head_dim << '-' << layout.value_heads << 'x'
            << layout.value_head_dim;
    }
    out << "-ff";
    for (int width : feed_forward_intermediates) out << '-' << width;
    out << "-exec";
    for (bool execute : execute_feed_forward) out << '-' << execute;
    out << "-tok" << token_policy.bos_token_id << '-' << token_policy.pad_token_id;
    for (int eos : token_policy.eos_token_ids) out << '-' << eos;
    out << "-num" << numerical_policy.norm_eps << '-'
        << numerical_policy.post_norm_eps << '-'
        << numerical_policy.embedding_multiplier << '-'
        << numerical_policy.attention_multiplier << '-'
        << numerical_policy.residual_multiplier << '-'
        << numerical_policy.logits_multiplier << '-'
        << numerical_policy.logits_divisor << '-'
        << numerical_policy.final_logit_softcap;
    return out.str();
}

std::string RuntimeTopology::summary() const {
    std::ostringstream out;
    out << "hidden=" << hidden << " intermediate=" << intermediate
        << " layers=" << num_hidden_layers
        << " mtp_layers=" << mtp_num_hidden_layers
        << " attention_layers=" << attention_layer_count
        << " conv_layers=" << conv_layer_count
        << " vocab=" << vocab_size;
    return out.str();
}

void RuntimeTopology::validate() const {
    if (hidden <= 0 || intermediate <= 0 || vocab_size <= 0 ||
        num_hidden_layers <= 0 || mtp_num_hidden_layers < 0) {
        throw std::runtime_error("invalid resolved model topology");
    }
    if (!checkpoint_layer_for_layer.empty()) {
        if (static_cast<int>(checkpoint_layer_for_layer.size()) != num_hidden_layers) {
            throw std::runtime_error("checkpoint layer mapping length mismatch");
        }
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
    for (int token : token_policy.eos_token_ids) {
        validate_token_id(token, "EOS");
    }
    numerical_policy.validate();
    if (static_cast<int>(mixer_kinds.size()) != num_hidden_layers) {
        throw std::runtime_error("resolved mixer schedule length mismatch: mixers=" +
            std::to_string(mixer_kinds.size()) + " expected=" +
            std::to_string(num_hidden_layers));
    }
    if (static_cast<int>(feed_forward_kinds.size()) != num_hidden_layers) {
        throw std::runtime_error("resolved feed-forward schedule length mismatch");
    }
    if (!execute_feed_forward.empty() &&
        static_cast<int>(execute_feed_forward.size()) != num_hidden_layers) {
        throw std::runtime_error("resolved FFN execution schedule length mismatch");
    }
    if (attention_layer_count + conv_layer_count + gated_delta_net_layer_count +
        mamba2_layer_count + mlp_only_layer_count != num_hidden_layers) {
        throw std::runtime_error("resolved layer counts are inconsistent");
    }
    if (num_experts > 0 && (moe_intermediate <= 0 || experts_per_token <= 0 ||
                            experts_per_token > num_experts)) {
        throw std::runtime_error("invalid resolved MoE topology");
    }
    if (static_cast<int>(attention_layouts.size()) != num_hidden_layers) {
        throw std::runtime_error("per-layer attention layout count mismatch");
    }
    if (gated_delta_net_layer_count > 0 &&
        static_cast<int>(gated_delta_net_layouts.size()) != num_hidden_layers) {
        throw std::runtime_error("per-layer GatedDeltaNet layout count mismatch");
    }
    for (int layer = 0; layer < num_hidden_layers; ++layer) {
        if (mixer_kinds[static_cast<size_t>(layer)] != MixerKind::GatedDeltaNet) continue;
        if (gated_delta_net_layouts.size() != static_cast<size_t>(num_hidden_layers)) {
            throw std::runtime_error("GatedDeltaNet layer has no layout table");
        }
        const GatedDeltaNetSpec& layout = gated_delta_net_layouts[static_cast<size_t>(layer)];
        if (layout.conv_kernel <= 0 || layout.key_head_dim <= 0 ||
            layout.value_head_dim <= 0 || layout.key_heads <= 0 ||
            layout.value_heads <= 0 || layout.value_heads % layout.key_heads != 0) {
            throw std::runtime_error("invalid GatedDeltaNet layout");
        }
    }
    if (!feed_forward_intermediates.empty() &&
        static_cast<int>(feed_forward_intermediates.size()) != num_hidden_layers) {
        throw std::runtime_error("per-layer FFN width count mismatch");
    }
    if (!feed_forward_intermediates.empty()) {
        for (int width : feed_forward_intermediates) {
            if (width <= 0) throw std::runtime_error("invalid per-layer FFN width");
        }
    }
    if (has_per_layer_input) {
        if (per_layer_input_size <= 0) {
            throw std::runtime_error("per-layer input size must be positive");
        }
        const std::size_t layers = static_cast<std::size_t>(num_hidden_layers);
        const std::size_t input_size = static_cast<std::size_t>(per_layer_input_size);
        if (layers > std::numeric_limits<std::size_t>::max() / input_size ||
            layers * input_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("per-layer input width exceeds supported range");
        }
    } else if (per_layer_input_size != 0) {
        throw std::runtime_error("per-layer input size set while feature is disabled");
    }
    for (int layer = 0; layer < num_hidden_layers; ++layer) {
        if (mixer_kinds[static_cast<size_t>(layer)] != MixerKind::Attention) continue;
        const AttentionSpec& layout = attention_layouts[static_cast<size_t>(layer)];
            if (layout.query_heads <= 0 || layout.key_value_heads <= 0 ||
                layout.query_heads % layout.key_value_heads != 0 ||
                layout.head_dim <= 0 || (layout.head_dim % 2) != 0 ||
                (layout.rope_position() != nullptr && layout.rotary_pairs() <= 0)) {
                throw std::runtime_error("invalid per-layer attention layout");
            }
            if (const auto* rope = layout.rope_position()) {
                try {
                    if (const auto* multi = layout.multi_axis_position()) {
                        multi->validate(layout.head_dim);
                    } else {
                        rope->validate(layout.head_dim);
                    }
                } catch (const std::exception& error) {
                    throw std::runtime_error(std::string("invalid per-layer position specification: ") +
                                             error.what());
                }
            }
            if (const auto* alibi = std::get_if<AlibiBiasSpec>(&layout.bias)) {
                alibi->validate(layout.query_heads);
            } else if (const auto* relative =
                           std::get_if<RelativePositionBiasSpec>(&layout.bias)) {
                relative->validate();
            }
            std::visit([](const auto& pattern) {
                using Pattern = std::decay_t<decltype(pattern)>;
                if constexpr (std::is_same_v<Pattern, SlidingWindowPattern>) {
                    if (pattern.window <= 0) throw std::runtime_error(
                        "sliding attention requires a window");
                } else if constexpr (std::is_same_v<Pattern, PrefixLmPattern>) {
                    if (pattern.prefix_length < 0) throw std::runtime_error(
                        "prefix-LM attention requires a non-negative prefix");
                } else if constexpr (std::is_same_v<Pattern, BlockSparsePattern>) {
                    if (pattern.block_size <= 0 || pattern.local_blocks < 0 ||
                        pattern.global_blocks < 0) throw std::runtime_error(
                        "block-sparse attention has invalid geometry");
                } else if constexpr (std::is_same_v<Pattern, DynamicSparsePattern>) {
                    if (pattern.block_size <= 0 || pattern.max_selected_blocks <= 0) {
                        throw std::runtime_error("dynamic sparse attention has invalid geometry");
                    }
                }
            }, layout.pattern);
            layout.state_storage.validate(layout.state);
    }
}

void ResolvedModel::validate() const {
    topology.validate();
    graph.validate();
    if (graph.layers.size() != static_cast<size_t>(topology.num_hidden_layers)) {
        throw std::runtime_error("resolved graph/topology layer count mismatch");
    }
    if (graph.embedding_transform.multiplier != topology.numerical_policy.embedding_multiplier ||
        graph.logits_divisor != topology.numerical_policy.logits_divisor ||
        graph.final_norm.epsilon != topology.numerical_policy.norm_eps) {
        throw std::runtime_error("resolved graph/topology numerical policy mismatch");
    }
    for (size_t layer = 0; layer < graph.layers.size(); ++layer) {
        if (graph.layers[layer].mixer_kind() != topology.mixer_kinds[layer] ||
            graph.layers[layer].feed_forward_kind() != topology.feed_forward_kinds[layer]) {
            throw std::runtime_error("resolved graph/topology operator schedule mismatch");
        }
    }
}

} // namespace celeg

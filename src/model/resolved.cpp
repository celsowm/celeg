#include "celeg/model/resolved.hpp"

#include "attention_validation.hpp"

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
    out << norm.epsilon << ':' << static_cast<int>(norm.weight_kind) << ':'
        << static_cast<int>(norm.granularity) << ';';
}

void append_optional_norm(std::ostringstream& out, const std::optional<NormSpec>& norm) {
    if (norm) {
        out << '1' << ':';
        append_norm(out, *norm);
    } else {
        out << '0' << ';';
    }
}

void append_rope(std::ostringstream& out, const RopePositionSpec& rope) {
    out << rope.theta << ':' << rope.rotary_fraction << ':'
        << static_cast<int>(rope.pairing) << ':';
    std::visit([&out](const auto& scaling) {
        using Scaling = std::decay_t<decltype(scaling)>;
        if constexpr (std::is_same_v<Scaling, NoRopeScaling>) {
            out << "none";
        } else if constexpr (std::is_same_v<Scaling, LinearRopeScaling>) {
            out << "linear:" << scaling.factor;
        } else if constexpr (std::is_same_v<Scaling, DynamicNtkRopeScaling>) {
            out << "dynamic_ntk:" << scaling.factor << ':' << scaling.original_context;
        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            out << "yarn:" << scaling.factor << ':' << scaling.original_context << ':'
                << scaling.attention_factor << ':' << scaling.beta_fast << ':'
                << scaling.beta_slow;
        } else if constexpr (std::is_same_v<Scaling, LongRopeScaling>) {
            out << "long:" << scaling.original_context << ":short=";
            for (float value : scaling.short_factors) out << value << ',';
            out << ":long=";
            for (float value : scaling.long_factors) out << value << ',';
        } else if constexpr (std::is_same_v<Scaling, Llama3FrequencyScaling>) {
            out << "llama3:" << scaling.factor << ':' << scaling.original_context << ':'
                << scaling.low_frequency_factor << ':' << scaling.high_frequency_factor;
        } else {
            static_assert(always_false_v<Scaling>, "unhandled RoPE scaling variant");
        }
    }, rope.scaling);
    out << ';';
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
        << attention.head_dim << ':';
    std::visit([&out](const auto& sharing) {
        using Sharing = std::decay_t<decltype(sharing)>;
        if constexpr (std::is_same_v<Sharing, PrivateKv>) {
            out << "private:";
        } else if constexpr (std::is_same_v<Sharing, SharedKvPublisher>) {
            out << "publisher:" << sharing.group << ':';
        } else if constexpr (std::is_same_v<Sharing, SharedKvConsumer>) {
            out << "consumer:" << sharing.group << ':';
        } else {
            static_assert(always_false_v<Sharing>, "unhandled kv sharing variant");
        }
    }, attention.kv_sharing);
    out << attention.query_scale << ':';
    append_optional_norm(out, attention.query_norm);
    append_optional_norm(out, attention.key_norm);
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
    out << ":position:";
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
    out << ":source:";
    std::visit([&out](const auto& source) {
        using Source = std::decay_t<decltype(source)>;
        if constexpr (std::is_same_v<Source, CurrentSequenceSource>) {
            out << "current";
        } else if constexpr (std::is_same_v<Source, ExternalMemorySource>) {
            out << "external:" << source.slot;
        } else {
            static_assert(always_false_v<Source>, "unhandled attention source variant");
        }
    }, attention.key_value_source);
    out << ":state:";
    std::visit([&out](const auto& state) {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, OrdinaryKvStateSpec>) {
            out << "ordinary:" << static_cast<int>(state.storage.key) << ':'
                << static_cast<int>(state.storage.value) << ':'
                << static_cast<int>(state.storage.layout);
        } else if constexpr (std::is_same_v<State, LatentAttentionStateSpec>) {
            out << "latent:" << state.latent_rank << ':' << state.rope_width << ':'
                << state.value_width << ':' << state.factorized_projection();
        } else {
            static_assert(always_false_v<State>, "unhandled attention state variant");
        }
    }, attention.state);
    out << ":gate:";
    if (!attention.output_gate) {
        out << "none";
    } else {
        out << attention.output_gate->packed_with_query << ':'
            << static_cast<int>(attention.output_gate->granularity);
    }
    out << ":output-transform:";
    std::visit([&out](const auto& transform) {
        using Transform = std::decay_t<decltype(transform)>;
        if constexpr (std::is_same_v<Transform, NoAttentionOutputTransformSpec>) {
            out << "none";
        } else if constexpr (std::is_same_v<Transform, OrthogonalizeCurrentValueSpec>) {
            out << "orthogonalize:" << transform.minimum_norm_squared;
        } else {
            static_assert(always_false_v<Transform>,
                          "unhandled attention output transform variant");
        }
    }, attention.output_transform);
    out << ';';
}

void append_mixer(std::ostringstream& out, const LayerSpec& layer) {
    std::visit([&out](const auto& mixer) {
        using Mixer = std::decay_t<decltype(mixer)>;
        if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
            out << "attention:";
            append_attention(out, mixer);
        } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
            out << "conv:" << mixer.kernel_size << ':' << mixer.hidden_size << ':'
                << mixer.bias;
        } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
            out << "gated-delta:" << mixer.conv_kernel << ':' << mixer.key_head_dim << ':'
                << mixer.value_head_dim << ':' << mixer.key_heads << ':'
                << mixer.value_heads << ':' << mixer.factorized_projections << ':'
                << mixer.vector_decay << ':' << mixer.safe_decay << ':'
                << mixer.decay_lower_bound << ':' << mixer.sigmoid_output_gate;
        } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
            out << "mamba2:" << mixer.conv_kernel << ':' << mixer.intermediate_size << ':'
                << mixer.state_size << ':' << mixer.num_heads << ':' << mixer.head_dim << ':'
                << mixer.group_count;
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
        if constexpr (std::is_same_v<FeedForward, std::monostate>) {
            out << "none";
        } else if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
            out << "dense:" << feed_forward.intermediate_size << ':'
                << static_cast<int>(feed_forward.activation);
        } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
            out << "moe:" << feed_forward.intermediate_size << ':'
                << feed_forward.num_experts << ':' << feed_forward.experts_per_token << ':'
                << feed_forward.normalize_weights << ':'
                << feed_forward.routed_scaling_factor << ':';
            std::visit([&out](const auto& selection) {
                using Selection = std::decay_t<decltype(selection)>;
                if constexpr (std::is_same_v<Selection, MoeTopKSelectionSpec>) {
                    out << "topk";
                } else if constexpr (std::is_same_v<Selection, MoeGroupedTopKSelectionSpec>) {
                    out << "grouped:" << selection.group_count << ':'
                        << selection.experts_per_group << ':'
                        << selection.groups_per_token << ':'
                        << selection.group_score_top_k;
                } else {
                    static_assert(always_false_v<Selection>,
                                  "unhandled MoE selection fingerprint variant");
                }
            }, feed_forward.selection);
            if (feed_forward.shared) {
                out << ":shared:" << feed_forward.shared->intermediate_size << ':'
                    << static_cast<int>(feed_forward.shared->combine_order);
            }
            out << ':' << feed_forward.router_softmax;
        } else {
            static_assert(always_false_v<FeedForward>,
                          "unhandled feed-forward fingerprint variant");
        }
    }, layer.feed_forward);
}

}

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
        << final_logit_softcap << ":per-layer-input="
        << per_layer_input.has_value();
    if (per_layer_input) {
        out << ':' << per_layer_input->input_size << ':'
            << static_cast<int>(per_layer_input->activation) << ':';
        append_norm(out, per_layer_input->norm);
    }
    out << ":layers=";
    for (const LayerSpec& layer : layers) {
        out << "mixer-norm:";
        append_optional_norm(out, layer.mixer_norm.before);
        append_optional_norm(out, layer.mixer_norm.after);
        append_mixer(out, layer);
        out << ":ffn-norm:";
        append_optional_norm(out, layer.feed_forward_norm.before);
        append_optional_norm(out, layer.feed_forward_norm.after);
        append_feed_forward(out, layer);
        out << ":residual=" << layer.residual.multiplier
            << ":scalar=" << layer.layer_scalar << ';';
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
    for (std::size_t index = 0; index < layer_count; ++index) {
        const LayerSpec& layer = graph.layers[index];
        std::visit([&](const auto& mixer) {
            using Mixer = std::decay_t<decltype(mixer)>;
            if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
                result.attention_slot_for_layer[index] = result.attention_layer_count++;
                result.layer_for_attention_slot.push_back(static_cast<int>(index));
            } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
                ++result.conv_layer_count;
            } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                ++result.gated_delta_net_layer_count;
            } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                ++result.mamba2_layer_count;
            } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
                ++result.mlp_only_layer_count;
            } else {
                static_assert(always_false_v<Mixer>, "unhandled mixer derivation variant");
            }
        }, layer.mixer);
    }
    return result;
}

void ExecutionTopology::validate() const {
    if (num_hidden_layers <= 0 ||
        attention_slot_for_layer.size() != static_cast<size_t>(num_hidden_layers)) {
        throw std::runtime_error("execution topology is inconsistent");
    }
    int expected_attention_slot = 0;
    for (size_t layer = 0; layer < attention_slot_for_layer.size(); ++layer) {
        const int slot = attention_slot_for_layer[layer];
        if (slot < -1 || slot >= attention_layer_count) {
            throw std::runtime_error("execution topology has invalid attention slot");
        }
        if (slot >= 0) {
            if (slot != expected_attention_slot ||
                expected_attention_slot >= static_cast<int>(layer_for_attention_slot.size()) ||
                layer_for_attention_slot[static_cast<size_t>(expected_attention_slot)] !=
                    static_cast<int>(layer)) {
                throw std::runtime_error("execution topology attention slots are not canonical");
            }
            ++expected_attention_slot;
        }
    }
    if (expected_attention_slot != attention_layer_count ||
        layer_for_attention_slot.size() != static_cast<size_t>(attention_layer_count) ||
        conv_layer_count < 0 || gated_delta_net_layer_count < 0 ||
        mamba2_layer_count < 0 || mlp_only_layer_count < 0 ||
        attention_layer_count + conv_layer_count + gated_delta_net_layer_count +
                mamba2_layer_count + mlp_only_layer_count != num_hidden_layers) {
        throw std::runtime_error("execution topology layer counts are inconsistent");
    }
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
    if (per_layer_input) {
        if (per_layer_input->input_size <= 0) {
            throw std::runtime_error("enabled per-layer input has invalid width");
        }
        per_layer_input->norm.validate();
    }
    for (size_t index = 0; index < norm_after_layers.size(); ++index) {
        const int layer = norm_after_layers[index];
        if (layer < 0 || layer >= static_cast<int>(layers.size()) - 1 ||
            (index > 0 && norm_after_layers[index - 1] >= layer)) {
            throw std::runtime_error("resolved model graph has invalid norm boundary");
        }
    }
    for (const LayerSpec& layer : layers) {
        if (!std::isfinite(layer.residual.multiplier) ||
            !std::isfinite(layer.layer_scalar)) {
            throw std::runtime_error("resolved model graph has invalid layer policy");
        }
        if (layer.mixer_norm.before) layer.mixer_norm.before->validate();
        if (layer.mixer_norm.after) layer.mixer_norm.after->validate();
        if (layer.feed_forward_norm.before) layer.feed_forward_norm.before->validate();
        if (layer.feed_forward_norm.after) layer.feed_forward_norm.after->validate();
        std::visit([](const auto& feed_forward) {
            using FeedForward = std::decay_t<decltype(feed_forward)>;
            if constexpr (std::is_same_v<FeedForward, std::monostate>) {
                return;
            } else if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
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
                std::visit([&](const auto& selection) {
                    using Selection = std::decay_t<decltype(selection)>;
                    if constexpr (std::is_same_v<Selection, MoeTopKSelectionSpec>) {
                        return;
                    } else if constexpr (std::is_same_v<Selection,
                                                        MoeGroupedTopKSelectionSpec>) {
                        if (selection.group_count <= 0 ||
                            selection.experts_per_group <= 0 ||
                            selection.groups_per_token <= 0 ||
                            selection.group_score_top_k <= 0 ||
                            selection.group_count * selection.experts_per_group !=
                                feed_forward.num_experts ||
                            selection.groups_per_token > selection.group_count ||
                            selection.group_score_top_k > selection.experts_per_group) {
                            throw std::runtime_error(
                                "MoE grouped routing fields are inconsistent");
                        }
                    } else {
                        static_assert(always_false_v<Selection>,
                                      "unhandled MoE selection validation variant");
                    }
                }, feed_forward.selection);
                if (feed_forward.shared && feed_forward.shared->intermediate_size <= 0) {
                    throw std::runtime_error("MoE shared expert has no positive width");
                }
            } else {
                static_assert(always_false_v<FeedForward>,
                              "unhandled feed-forward semantic validation variant");
            }
        }, layer.feed_forward);
        if (const auto* mlp = std::get_if<MlpBlockSpec>(&layer.mixer);
            mlp && mlp->intermediate_size <= 0) {
            throw std::runtime_error("MLP-only layer has no positive width");
        }
        if (const auto* attention = std::get_if<AttentionSpec>(&layer.mixer)) {
            if (attention->query_norm) attention->query_norm->validate();
            if (attention->key_norm) attention->key_norm->validate();
            std::visit([&](const auto& position) {
                using Position = std::decay_t<decltype(position)>;
                if constexpr (std::is_same_v<Position, NoPositionEncodingSpec>) {
                    return;
                } else if constexpr (std::is_same_v<Position, RopePositionSpec>) {
                    position.validate(attention->head_dim);
                } else if constexpr (std::is_same_v<Position, MultiAxisRopeSpec>) {
                    position.validate(attention->head_dim);
                } else {
                    static_assert(always_false_v<Position>,
                                  "unhandled position validation variant");
                }
            }, attention->position);
            std::visit([&](const auto& bias) {
                using Bias = std::decay_t<decltype(bias)>;
                if constexpr (std::is_same_v<Bias, NoAttentionBiasSpec>) {
                    return;
                } else if constexpr (std::is_same_v<Bias, AlibiBiasSpec>) {
                    bias.validate(attention->query_heads);
                } else if constexpr (std::is_same_v<Bias, RelativePositionBiasSpec>) {
                    bias.validate();
                } else {
                    static_assert(always_false_v<Bias>,
                                  "unhandled attention bias validation variant");
                }
            }, attention->bias);
            if (const auto* sliding = std::get_if<SlidingWindowPattern>(&attention->pattern);
                sliding && sliding->window <= 0) {
                throw std::runtime_error("sliding-window attention has invalid window");
            }
            if (attention->output_gate.has_value()) {
                switch (attention->output_gate->granularity) {
                case AttentionGateGranularity::OutputWise:
                case AttentionGateGranularity::HeadWise:
                case AttentionGateGranularity::ElementWise:
                    break;
                default:
                    throw std::runtime_error("invalid attention gate granularity");
                }
            }
            validate_attention_representation(*attention);
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
    if (bucket_count < 2 || max_distance <= 0) {
        throw std::runtime_error(
            "relative position bias requires at least two buckets and a positive max distance");
    }
    if (bidirectional && (bucket_count < 4 || (bucket_count % 2) != 0)) {
        throw std::runtime_error(
            "bidirectional relative position bias requires an even bucket count of at least four");
    }
}

void RopePositionSpec::validate(int head_dim) const {
    if (!(theta > 0.0) || !std::isfinite(theta) ||
        !(rotary_fraction > 0.0) || rotary_fraction > 1.0 ||
        !std::isfinite(rotary_fraction) || head_dim <= 0) {
        throw std::runtime_error("RoPE parameters are invalid");
    }
    if (static_cast<int>(rotary_fraction * static_cast<double>(head_dim)) % 2 != 0) {
        throw std::runtime_error("RoPE rotary width must be even");
    }
    std::visit([&](const auto& scaling) {
        using Scaling = std::decay_t<decltype(scaling)>;
        if constexpr (std::is_same_v<Scaling, NoRopeScaling>) {
            return;
        } else if constexpr (std::is_same_v<Scaling, LinearRopeScaling>) {
            if (!(scaling.factor > 0.0) || !std::isfinite(scaling.factor)) {
                throw std::runtime_error("linear RoPE scaling factor is invalid");
            }
        } else if constexpr (std::is_same_v<Scaling, DynamicNtkRopeScaling>) {
            if (!(scaling.factor > 0.0) || !std::isfinite(scaling.factor) ||
                scaling.original_context <= 0) {
                throw std::runtime_error("dynamic NTK RoPE parameters are invalid");
            }
        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            if (!(scaling.factor > 0.0) || !std::isfinite(scaling.factor) ||
                !(scaling.attention_factor > 0.0) || !std::isfinite(scaling.attention_factor) ||
                !(scaling.beta_fast > 0.0) || !(scaling.beta_slow > 0.0) ||
                scaling.original_context <= 0) {
                throw std::runtime_error("YaRN RoPE parameters are invalid");
            }
        } else if constexpr (std::is_same_v<Scaling, LongRopeScaling>) {
            if (scaling.original_context <= 0 || scaling.short_factors.empty() ||
                scaling.short_factors.size() != scaling.long_factors.size()) {
                throw std::runtime_error("LongRoPE factors are invalid");
            }
            for (float factor : scaling.short_factors) {
                if (!(factor > 0.0f) || !std::isfinite(factor)) {
                    throw std::runtime_error("LongRoPE short factors are invalid");
                }
            }
            for (float factor : scaling.long_factors) {
                if (!(factor > 0.0f) || !std::isfinite(factor)) {
                    throw std::runtime_error("LongRoPE long factors are invalid");
                }
            }
        } else if constexpr (std::is_same_v<Scaling, Llama3FrequencyScaling>) {
            if (!(scaling.factor > 0.0) || !std::isfinite(scaling.factor) ||
                scaling.original_context <= 0 ||
                !(scaling.low_frequency_factor > 0.0) ||
                !(scaling.high_frequency_factor > 0.0) ||
                scaling.low_frequency_factor > scaling.high_frequency_factor) {
                throw std::runtime_error("Llama3 RoPE scaling parameters are invalid");
            }
        } else {
            static_assert(always_false_v<Scaling>, "unhandled RoPE scaling validation variant");
        }
    }, scaling);
}

void MultiAxisRopeSpec::validate(int head_dim) const {
    base.validate(head_dim);
    if (axes <= 0 || axes != static_cast<int>(sections.size())) {
        throw std::runtime_error("multi-axis RoPE has invalid axis count");
    }
    int sum = 0;
    for (int section : sections) {
        if (section <= 0) {
            throw std::runtime_error("multi-axis RoPE has non-positive section width");
        }
        sum += section;
    }
    if (sum * 2 != static_cast<int>(base.rotary_fraction * head_dim)) {
        throw std::runtime_error("multi-axis RoPE sections do not cover rotary width");
    }
}

void NormSpec::validate() const {
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        throw std::runtime_error("normalization epsilon must be positive and finite");
    }
}

void ResolvedModel::validate() const {
    provenance.validate();
    graph.validate();
    topology.validate();
    weight_plan.validate(graph.layers.size());
}

}
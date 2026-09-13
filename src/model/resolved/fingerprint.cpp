#include "celeg/model/resolved.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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
        } else if constexpr (std::is_same_v<Scaling, ProportionalRopeScaling>) {
            out << "proportional:" << scaling.factor;
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
    append_optional_norm(out, attention.value_norm);
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
    out << ":gate:";
    if (attention.output_gate.has_value()) {
        out << "sigmoid:" << attention.output_gate->packed_with_query << ':'
            << static_cast<int>(attention.output_gate->granularity);
    } else {
        out << "none";
    }
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
            out << "ordinary:" << state.quantizable << ":storage:"
                << static_cast<int>(state.storage.key) << ':'
                << static_cast<int>(state.storage.value) << ':'
                << static_cast<int>(state.storage.granularity) << ':'
                << state.storage.paged;
        } else if constexpr (std::is_same_v<State, LatentAttentionStateSpec>) {
            out << "latent:" << state.latent_rank << ':' << state.rope_head_dim << ':'
                << state.nope_head_dim << ':' << state.decoupled_rope << ':';
            std::visit([&out](const auto& projection) {
                using Projection = std::decay_t<decltype(projection)>;
                if constexpr (std::is_same_v<Projection, DirectLatentProjection>) {
                    out << "direct";
                } else if constexpr (std::is_same_v<Projection, FactorizedLatentProjection>) {
                    out << "factorized:" << projection.query_rank << ':'
                        << projection.value_head_dim;
                    append_norm(out, projection.query_latent_norm);
                    append_norm(out, projection.key_latent_norm);
                } else {
                    static_assert(always_false_v<Projection>,
                                  "unhandled latent projection variant");
                }
            }, state.projection);
            out << ":storage:" << static_cast<int>(state.storage.latent) << ':'
                << static_cast<int>(state.storage.rotary) << ':'
                << static_cast<int>(state.storage.granularity) << ':'
                << state.storage.paged;
        } else {
            static_assert(always_false_v<State>, "unhandled attention state variant");
        }
    }, attention.state);
    out << ":source:";
    std::visit([&out](const auto& source) {
        using Source = std::decay_t<decltype(source)>;
        if constexpr (std::is_same_v<Source, CurrentSequenceSource>) {
            out << "current_sequence";
        } else if constexpr (std::is_same_v<Source, ExternalMemorySource>) {
            out << "external_memory:" << source.slot;
        } else {
            static_assert(always_false_v<Source>, "unhandled attention key/value source variant");
        }
    }, attention.key_value_source);
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
        if constexpr (std::is_same_v<FeedForward, std::monostate>) {
            out << "none";
        } else if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
            out << "dense:" << feed_forward.intermediate_size << ':'
                << static_cast<int>(feed_forward.activation) << ':'
                << feed_forward.parallel_intermediate_size;
        } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
            out << "moe:" << feed_forward.intermediate_size << ':' << feed_forward.num_experts
                << ':' << feed_forward.experts_per_token << ':' << feed_forward.normalize_topk
                << ':' << feed_forward.use_expert_bias << ':'
                << feed_forward.routed_scaling_factor << ':';
            std::visit([&out](const auto& selection) {
                using Selection = std::decay_t<decltype(selection)>;
                if constexpr (std::is_same_v<Selection, MoeTopKSelectionSpec>) {
                    out << "top-k";
                } else if constexpr (std::is_same_v<Selection, MoeGroupedTopKSelectionSpec>) {
                    out << "grouped:" << selection.group_count << ':'
                        << selection.experts_per_group << ':'
                        << selection.groups_per_token << ':'
                        << selection.group_score_top_k;
                } else {
                    static_assert(always_false_v<Selection>,
                                  "unhandled MoE selection variant");
                }
            }, feed_forward.selection);
            out << ":shared:" << feed_forward.shared.has_value();
            if (feed_forward.shared) {
                out << ':' << feed_forward.shared->intermediate_size << ':'
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

}

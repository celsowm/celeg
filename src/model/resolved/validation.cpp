#include "celeg/model/resolved.hpp"

#include "../attention_validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace celeg {

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
                if (feed_forward.parallel_intermediate_size < 0) {
                    throw std::runtime_error("dense layer has invalid parallel FFN width");
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
                                      "unhandled MoE selection variant");
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
            if (attention->value_norm) {
                attention->value_norm->validate();
                if (attention->uses_latent_state()) {
                    throw std::runtime_error(
                        "value normalization is meaningless for latent attention");
                }
            }
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
    if (bucket_count <= 0 || max_distance <= 0) {
        throw std::runtime_error("relative position bias dimensions must be positive");
    }
    if (bidirectional && (bucket_count < 2 || (bucket_count % 2) != 0)) {
        throw std::runtime_error(
            "bidirectional relative position bias requires an even bucket count");
    }
}

namespace {

void validate_state_scalar(StateScalarType scalar) {
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
}

void validate_state_granularity(StateQuantizationGranularity granularity) {
    switch (granularity) {
    case StateQuantizationGranularity::PerTensor:
    case StateQuantizationGranularity::PerHead:
    case StateQuantizationGranularity::PerToken:
    case StateQuantizationGranularity::PerBlock:
        return;
    }
    throw std::runtime_error("invalid attention state quantization granularity");
}

}

void OrdinaryKvStorageSpec::validate() const {
    validate_state_scalar(key);
    validate_state_scalar(value);
    validate_state_granularity(granularity);
}

void LatentStorageSpec::validate() const {
    validate_state_scalar(latent);
    validate_state_scalar(rotary);
    validate_state_granularity(granularity);
}

void LatentAttentionStateSpec::validate() const {
    storage.validate();
    if (latent_rank <= 0 || rope_head_dim < 0 || nope_head_dim < 0 ||
        rope_head_dim + nope_head_dim <= 0) {
        throw std::runtime_error("invalid latent attention state dimensions");
    }
    if (const auto* factorized = factorized_projection();
        factorized &&
        (factorized->query_rank <= 0 || factorized->value_head_dim <= 0 ||
         !std::isfinite(factorized->query_latent_norm.epsilon) ||
         factorized->query_latent_norm.epsilon <= 0.0f ||
         !std::isfinite(factorized->key_latent_norm.epsilon) ||
         factorized->key_latent_norm.epsilon <= 0.0f)) {
        throw std::runtime_error("invalid factorized latent attention projections");
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

void ResolvedModel::validate() const {
    topology.validate();
    graph.validate();
    if (graph.layers.size() != static_cast<size_t>(topology.exec.num_hidden_layers)) {
        throw std::runtime_error("resolved graph/topology layer count mismatch");
    }
}

}

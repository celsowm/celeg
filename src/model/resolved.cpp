#include "celeg/model/resolved.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace celeg {

void ModelGraph::validate() const {
    if (layers.empty()) {
        throw std::runtime_error("resolved model graph has no layers");
    }
    if (!(final_norm.epsilon > 0.0f) || !std::isfinite(final_norm.epsilon) ||
        !std::isfinite(embedding_multiplier) ||
        !std::isfinite(logits_divisor) || logits_divisor <= 0.0f ||
        !std::isfinite(final_logit_softcap) || final_logit_softcap < 0.0f) {
        throw std::runtime_error("resolved model graph has invalid policies");
    }
    for (const LayerSpec& layer : layers) {
        if (!(layer.operator_norm.epsilon > 0.0f) ||
            !std::isfinite(layer.residual.multiplier) ||
            !std::isfinite(layer.layer_scalar)) {
            throw std::runtime_error("resolved model graph has invalid layer policy");
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
        !(logits_divisor > 0.0f) || !std::isfinite(logits_divisor) ||
        !std::isfinite(embedding_multiplier) ||
        !std::isfinite(attention_multiplier) ||
        !std::isfinite(residual_multiplier) ||
        final_logit_softcap < 0.0f || !std::isfinite(final_logit_softcap)) {
        throw std::runtime_error("invalid resolved model numerical policy");
    }
}

std::string RuntimeTopology::fingerprint() const {
    std::ostringstream out;
    out << "h" << hidden << "-l" << num_hidden_layers
        << "-voc" << vocab_size
        << "-int" << intermediate << "-cc" << conv_cache
        << "-e" << num_experts << "-k" << experts_per_token
        << "-mi" << moe_intermediate;
    out << "-m2i" << mamba2_intermediate;
    for (MixerKind kind : mixer_kinds) {
        switch (kind) {
        case MixerKind::Attention: out << "-a"; break;
        case MixerKind::ShortConvolution: out << "-c"; break;
        case MixerKind::GatedDeltaNet: out << "-d"; break;
        case MixerKind::Mamba2: out << "-m2"; break;
        case MixerKind::MlpOnly: out << "-mlp"; break;
        }
    }
    out << "-ff";
    for (int width : feed_forward_intermediates) out << '-' << width;
    out << "-tok" << token_policy.bos_token_id << '-' << token_policy.pad_token_id;
    for (int eos : token_policy.eos_token_ids) out << '-' << eos;
    out << "-num" << numerical_policy.norm_eps << '-'
        << numerical_policy.embedding_multiplier << '-'
        << numerical_policy.attention_multiplier << '-'
        << numerical_policy.residual_multiplier << '-'
        << numerical_policy.logits_divisor << '-'
        << numerical_policy.final_logit_softcap;
    return out.str();
}

std::string RuntimeTopology::summary() const {
    std::ostringstream out;
    out << "hidden=" << hidden << " intermediate=" << intermediate
        << " layers=" << num_hidden_layers
        << " attention_layers=" << attention_layer_count
        << " conv_layers=" << conv_layer_count
        << " vocab=" << vocab_size;
    return out.str();
}

void RuntimeTopology::validate() const {
    if (hidden <= 0 || intermediate <= 0 || vocab_size <= 0 ||
        num_hidden_layers <= 0) {
        throw std::runtime_error("invalid resolved model topology");
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
                (layout.positional_encoding == PositionalEncodingKind::Rope &&
                 (layout.rope_theta <= 0.0 || layout.rotary_fraction <= 0.0 ||
                  layout.rotary_fraction > 1.0)) ||
                (layout.positional_encoding == PositionalEncodingKind::None &&
                 layout.query_key_norm) ||
                layout.rotary_fraction > 1.0) {
                throw std::runtime_error("invalid per-layer attention layout");
            }
            if (layout.mask == AttentionMaskKind::SlidingCausal &&
                layout.sliding_window <= 0) {
                throw std::runtime_error("sliding attention requires a window");
            }
    }
}

void ResolvedModel::validate() const {
    topology.validate();
    graph.validate();
    if (graph.layers.size() != static_cast<size_t>(topology.num_hidden_layers)) {
        throw std::runtime_error("resolved graph/topology layer count mismatch");
    }
    if (graph.embedding_multiplier != topology.numerical_policy.embedding_multiplier ||
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

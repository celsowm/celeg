#include "celeg/model/resolved.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace celeg {

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
    for (MixerKind kind : mixer_kinds) out << (kind == MixerKind::Attention ? "-a" : "-c");
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
    numerical_policy.validate();
    if (static_cast<int>(mixer_kinds.size()) != num_hidden_layers) {
        throw std::runtime_error("resolved mixer schedule length mismatch: mixers=" +
            std::to_string(mixer_kinds.size()) + " expected=" +
            std::to_string(num_hidden_layers));
    }
    if (static_cast<int>(feed_forward_kinds.size()) != num_hidden_layers) {
        throw std::runtime_error("resolved feed-forward schedule length mismatch");
    }
    if (attention_layer_count + conv_layer_count != num_hidden_layers) {
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
    for (const AttentionSpec& layout : attention_layouts) {
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

} // namespace celeg

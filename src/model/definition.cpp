#include "celeg/model/definition.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace celeg {

void TransformerDimensions::validate() const {
    if (hidden_size <= 0 || intermediate_size <= 0 || num_layers <= 0 ||
        num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0 ||
        vocab_size <= 0 || max_context <= 0) {
        throw std::invalid_argument("model dimensions must be positive");
    }
    if (num_attention_heads % num_key_value_heads != 0) {
        throw std::invalid_argument(
            "attention heads must be divisible by key-value heads");
    }
}

void RopeScalingSpec::validate(int rotary_dimension) const {
    if (rotary_dimension <= 0 || (rotary_dimension % 2) != 0) {
        throw std::invalid_argument("RoPE rotary dimension must be positive and even");
    }
    if (kind != RopeScalingKind::None && (!(factor > 0.0) || !std::isfinite(factor))) {
        throw std::invalid_argument("RoPE scaling factor must be positive");
    }
    if ((kind == RopeScalingKind::Linear || kind == RopeScalingKind::DynamicNtk ||
         kind == RopeScalingKind::Yarn || kind == RopeScalingKind::Long ||
         kind == RopeScalingKind::Llama3Frequency) && original_context <= 0) {
        throw std::invalid_argument("scaled RoPE requires an original context length");
    }
    if (!std::isfinite(attention_factor) || attention_factor <= 0.0) {
        throw std::invalid_argument("RoPE attention factor must be positive");
    }
    if (kind == RopeScalingKind::Yarn &&
        (!std::isfinite(beta_fast) || !std::isfinite(beta_slow) ||
         beta_fast < 0.0 || beta_slow < 0.0 || beta_fast > beta_slow)) {
        throw std::invalid_argument("invalid YaRN beta range");
    }
    if (kind == RopeScalingKind::Llama3Frequency &&
        (!(low_frequency_factor > 0.0) || !(high_frequency_factor > 0.0) ||
         !std::isfinite(low_frequency_factor) || !std::isfinite(high_frequency_factor))) {
        throw std::invalid_argument("invalid Llama-3 RoPE frequency factors");
    }
    if (kind == RopeScalingKind::Long) {
        if (short_factors.size() != static_cast<size_t>(rotary_dimension / 2) ||
            long_factors.size() != static_cast<size_t>(rotary_dimension / 2)) {
            throw std::invalid_argument("LongRoPE factors must match rotary dimension");
        }
        for (float value : short_factors) {
            if (!(value > 0.0f) || !std::isfinite(value)) {
                throw std::invalid_argument("invalid LongRoPE short factor");
            }
        }
        for (float value : long_factors) {
            if (!(value > 0.0f) || !std::isfinite(value)) {
                throw std::invalid_argument("invalid LongRoPE long factor");
            }
        }
    }
}

void RopePositionSpec::validate(int head_dimension) const {
    if (!(theta > 0.0) || !std::isfinite(theta) ||
        !(rotary_fraction > 0.0) || rotary_fraction > 1.0 ||
        !std::isfinite(rotary_fraction)) {
        throw std::invalid_argument("invalid RoPE position specification");
    }
    const int rotary_dimension = static_cast<int>(head_dimension * rotary_fraction);
    scaling.validate(rotary_dimension);
}

void MultiAxisRopeSpec::validate(int head_dimension) const {
    if (axes != 3 || !interleaved) {
        throw std::invalid_argument("multi-axis RoPE currently requires three interleaved axes");
    }
    base.validate(head_dimension);
    for (int section : sections) {
        if (section <= 0) throw std::invalid_argument("M-RoPE sections must be positive");
    }
}

void ModelNumerics::validate() const {
    if (!(norm_epsilon > 0.0f) || !std::isfinite(norm_epsilon) ||
        !std::isfinite(embedding_multiplier) ||
        !std::isfinite(attention_multiplier) ||
        !std::isfinite(attention_output_multiplier) ||
        !std::isfinite(residual_multiplier) ||
        !(logits_divisor > 0.0f) || !std::isfinite(logits_divisor)) {
        throw std::invalid_argument("model numerical parameters are invalid");
    }
}

void ModelDefinition::validate() const {
    dimensions.validate();
    std::visit([this](const auto& position) {
        using T = std::decay_t<decltype(position)>;
        if constexpr (std::is_same_v<T, RopePositionSpec>) {
            position.validate(dimensions.head_dim);
        } else if constexpr (std::is_same_v<T, MultiAxisRopeSpec>) {
            position.validate(dimensions.head_dim);
        }
    }, position);
    numerics.validate();
    if (tokens.bos < 0 || tokens.eos.empty() || tokens.pad < 0) {
        throw std::invalid_argument("model token ids are invalid");
    }
    if (architecture.empty()) {
        throw std::invalid_argument("model architecture identifier is empty");
    }
}

} // namespace celeg

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

void validate_rope_scaling(const RopeScalingSpec& scaling, int rotary_dimension) {
    if (rotary_dimension <= 0 || (rotary_dimension % 2) != 0) {
        throw std::invalid_argument("RoPE rotary dimension must be positive and even");
    }
    std::visit([&](const auto& value) {
        using Scaling = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Scaling, NoRopeScaling>) {
            return;
        } else if constexpr (std::is_same_v<Scaling, LinearRopeScaling>) {
            if (!(value.factor > 0.0) || !std::isfinite(value.factor)) {
                throw std::invalid_argument("RoPE scaling factor must be positive");
            }
        } else if constexpr (std::is_same_v<Scaling, DynamicNtkRopeScaling>) {
            if (!(value.factor > 0.0) || !std::isfinite(value.factor)) {
                throw std::invalid_argument("RoPE scaling factor must be positive");
            }
            if (value.original_context <= 0) {
                throw std::invalid_argument("scaled RoPE requires an original context length");
            }
        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            if (!(value.factor >= 1.0) || !std::isfinite(value.factor)) {
                throw std::invalid_argument("YaRN scaling factor must be at least one");
            }
            if (value.original_context <= 0) {
                throw std::invalid_argument("YaRN requires an original context length");
            }
            if (!std::isfinite(value.attention_factor) || value.attention_factor <= 0.0) {
                throw std::invalid_argument("RoPE attention factor must be positive");
            }
            if (!std::isfinite(value.beta_fast) || !std::isfinite(value.beta_slow) ||
                value.beta_fast <= 0.0 || value.beta_slow <= 0.0 ||
                value.beta_fast < value.beta_slow) {
                throw std::invalid_argument("invalid YaRN beta range");
            }
        } else if constexpr (std::is_same_v<Scaling, LongRopeScaling>) {
            if (value.original_context <= 0) {
                throw std::invalid_argument("scaled RoPE requires an original context length");
            }
            if (value.short_factors.size() != static_cast<size_t>(rotary_dimension / 2) ||
                value.long_factors.size() != static_cast<size_t>(rotary_dimension / 2)) {
                throw std::invalid_argument("LongRoPE factors must match rotary dimension");
            }
            for (float v : value.short_factors) {
                if (!(v > 0.0f) || !std::isfinite(v)) {
                    throw std::invalid_argument("invalid LongRoPE short factor");
                }
            }
            for (float v : value.long_factors) {
                if (!(v > 0.0f) || !std::isfinite(v)) {
                    throw std::invalid_argument("invalid LongRoPE long factor");
                }
            }
        } else if constexpr (std::is_same_v<Scaling, Llama3FrequencyScaling>) {
            if (!(value.factor > 0.0) || !std::isfinite(value.factor)) {
                throw std::invalid_argument("RoPE scaling factor must be positive");
            }
            if (value.original_context <= 0) {
                throw std::invalid_argument("scaled RoPE requires an original context length");
            }
            if (!(value.low_frequency_factor > 0.0) || !(value.high_frequency_factor > 0.0) ||
                !std::isfinite(value.low_frequency_factor) ||
                !std::isfinite(value.high_frequency_factor)) {
                throw std::invalid_argument("invalid Llama-3 RoPE frequency factors");
            }
        } else {
            static_assert(always_false_v<Scaling>, "unhandled RoPE scaling variant");
        }
    }, scaling);
}

void RopePositionSpec::validate(int head_dimension) const {
    if (!(theta > 0.0) || !std::isfinite(theta) ||
        !(rotary_fraction > 0.0) || rotary_fraction > 1.0 ||
        !std::isfinite(rotary_fraction)) {
        throw std::invalid_argument("invalid RoPE position specification");
    }
    const int rotary_dimension = static_cast<int>(head_dimension * rotary_fraction);
    validate_rope_scaling(scaling, rotary_dimension);
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

}

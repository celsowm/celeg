#include "celeg/model/definition.hpp"

#include <cmath>
#include <stdexcept>

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
    if (hidden_size != num_attention_heads * head_dim) {
        throw std::invalid_argument("hidden size must equal attention heads * head dimension");
    }
}

void RopeSpec::validate() const {
    if (kind == PositionalEncodingKind::Rope && theta <= 0.0) {
        throw std::invalid_argument("RoPE theta must be positive");
    }
    if (scaling_factor.has_value() && *scaling_factor <= 0.0) {
        throw std::invalid_argument("RoPE scaling factor must be positive");
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
    rope.validate();
    numerics.validate();
    if (architecture.empty()) {
        throw std::invalid_argument("model architecture identifier is empty");
    }
}

} // namespace celeg

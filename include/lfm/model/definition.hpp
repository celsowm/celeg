#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lfm {

// Backend-neutral dimensions shared by transformer model families. Model
// providers translate checkpoint metadata into this value once; execution
// code should not repeatedly reinterpret provider-specific configuration.
struct TransformerDimensions {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_context = 0;

    void validate() const;
};

enum class PositionalEncodingKind : uint8_t {
    None,
    Rope,
};

struct RopeSpec {
    PositionalEncodingKind kind = PositionalEncodingKind::None;
    double theta = 0.0;
    std::optional<double> scaling_factor;

    void validate() const;
};

struct ModelNumerics {
    float norm_epsilon = 0.0f;
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 0.0f;
    float attention_output_multiplier = 1.0f;
    float residual_multiplier = 1.0f;
    float logits_divisor = 1.0f;

    void validate() const;
};

struct TokenIds {
    int bos = -1;
    int eos = -1;
    int pad = -1;
};

// Immutable, validated common definition. `architecture` is an opaque
// provider identifier; architecture-specific fields stay in the provider's
// own specification instead of being added to this common structure.
struct ModelDefinition {
    TransformerDimensions dimensions;
    RopeSpec rope;
    ModelNumerics numerics;
    TokenIds tokens;
    std::string architecture;
    std::string source_format;

    void validate() const;
};

} // namespace lfm

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace celeg {

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

enum class RopeScalingKind : uint8_t {
    None,
    Linear,
    DynamicNtk,
    Yarn,
    Long,
    Llama3Frequency,
};

struct RopeScalingSpec {
    RopeScalingKind kind = RopeScalingKind::None;
    double factor = 1.0;
    int original_context = 0;
    double attention_factor = 1.0;
    double beta_fast = 0.0;
    double beta_slow = 0.0;
    double low_frequency_factor = 1.0;
    double high_frequency_factor = 1.0;
    std::vector<float> short_factors;
    std::vector<float> long_factors;

    void validate(int rotary_dimension) const;
};

struct NoPositionEncodingSpec {};

struct RopePositionSpec {
    double theta = 0.0;
    double rotary_fraction = 1.0;
    RopeScalingSpec scaling;

    void validate(int head_dimension) const;
};

struct MultiAxisRopeSpec {
    RopePositionSpec base;
    std::array<int, 3> sections{0, 0, 0};
    bool interleaved = false;
    int axes = 3;

    void validate(int head_dimension) const;
};

using PositionSpec = std::variant<NoPositionEncodingSpec, RopePositionSpec,
                                  MultiAxisRopeSpec>;

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
    std::vector<int> eos;
    int pad = -1;
};

// Immutable, validated common definition. `architecture` is an opaque
// provider identifier; architecture-specific fields stay in the provider's
// own specification instead of being added to this common structure.
struct ModelDefinition {
    TransformerDimensions dimensions;
    PositionSpec position;
    ModelNumerics numerics;
    TokenIds tokens;
    std::string architecture;
    std::string source_format;

    void validate() const;
};

} // namespace celeg

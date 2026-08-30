#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace celeg {

template <typename T>
inline constexpr bool always_false_v = false;

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

struct NoRopeScaling {
    friend bool operator==(const NoRopeScaling&, const NoRopeScaling&) = default;
};

struct LinearRopeScaling {
    double factor = 1.0;

    friend bool operator==(const LinearRopeScaling&, const LinearRopeScaling&) = default;
};

struct DynamicNtkRopeScaling {
    double factor = 1.0;
    int original_context = 0;

    friend bool operator==(const DynamicNtkRopeScaling&, const DynamicNtkRopeScaling&) = default;
};

struct YarnRopeScaling {
    double factor = 1.0;
    double attention_factor = 1.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    int original_context = 0;

    friend bool operator==(const YarnRopeScaling&, const YarnRopeScaling&) = default;
};

struct LongRopeScaling {
    int original_context = 0;
    std::vector<float> short_factors;
    std::vector<float> long_factors;

    friend bool operator==(const LongRopeScaling&, const LongRopeScaling&) = default;
};

struct Llama3FrequencyScaling {
    double factor = 1.0;
    int original_context = 0;
    double low_frequency_factor = 1.0;
    double high_frequency_factor = 1.0;

    friend bool operator==(const Llama3FrequencyScaling&, const Llama3FrequencyScaling&) = default;
};

using RopeScalingSpec = std::variant<
    NoRopeScaling,
    LinearRopeScaling,
    DynamicNtkRopeScaling,
    YarnRopeScaling,
    LongRopeScaling,
    Llama3FrequencyScaling>;

void validate_rope_scaling(const RopeScalingSpec& scaling, int rotary_dimension);

enum class RopePairingKind : uint8_t {
    SplitHalf,
    AdjacentPairs,
};

struct NoPositionEncodingSpec {};

struct RopePositionSpec {
    double theta = 0.0;
    double rotary_fraction = 1.0;
    RopeScalingSpec scaling;
    RopePairingKind pairing = RopePairingKind::SplitHalf;

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

struct ModelDefinition {
    TransformerDimensions dimensions;
    PositionSpec position;
    ModelNumerics numerics;
    TokenIds tokens;
    std::string architecture;
    std::string source_format;

    void validate() const;
};

}

#include <metal_stdlib>

using namespace metal;

struct CelegRopePairComponents {
    uint first;
    uint second;
};

enum CelegRopeScalingMode : uint {
    CelegRopeScalingNone = 0u,
    CelegRopeScalingLinear = 1u,
    CelegRopeScalingDynamicNtk = 2u,
    CelegRopeScalingYarn = 3u,
    CelegRopeScalingLong = 4u,
    CelegRopeScalingLlama3 = 5u,
    CelegRopeScalingProportional = 6u,
};

#ifndef CELEG_ROPE_SCALING_SPEC_DEFINED
#define CELEG_ROPE_SCALING_SPEC_DEFINED
struct CelegRopeScalingSpec {
    uint mode;
    uint original_context;
    float rotary_fraction;
    float factor;
    float beta_fast;
    float beta_slow;
    float attention_factor;
    float low_frequency_factor;
    float high_frequency_factor;
};
#endif

inline float celeg_rope_unscaled_frequency(
    float theta, uint pair, uint rotary_dimension) {
    return pow(theta,
               -2.0f * static_cast<float>(pair) /
                   static_cast<float>(rotary_dimension));
}

inline CelegRopePairComponents celeg_rope_pair_components(
    uint pair, uint pair_count, uint pairing_mode) {
    if (pairing_mode == 1u) {
        return {2u * pair, 2u * pair + 1u};
    }
    return {pair, pair_count + pair};
}

inline uint celeg_mrope_axis_for_pair(
    uint pair, uint section0, uint section1, uint interleaved) {
    if (interleaved != 0u) return pair % 3u;
    if (pair < section0) return 0u;
    if (pair < section0 + section1) return 1u;
    return 2u;
}

inline float celeg_rope_scaled_frequency(
    float theta,
    float rotary_fraction,
    uint pair,
    uint rotary_dimension,
    uint position,
    uint scaling_mode,
    float factor,
    float beta_fast,
    float beta_slow,
    uint original_context,
    float low_frequency_factor,
    float high_frequency_factor,
    device const float* short_factors,
    device const float* long_factors) {
    float base = theta;
    const float context = static_cast<float>(position);
    if (scaling_mode == CelegRopeScalingDynamicNtk &&
        original_context > 0u && context > static_cast<float>(original_context)) {
        const float ratio = factor * context / static_cast<float>(original_context) -
                            (factor - 1.0f);
        const float denominator = max(2.0f, static_cast<float>(rotary_dimension) - 2.0f);
        base *= pow(max(1.0f, ratio),
                    static_cast<float>(rotary_dimension) / denominator);
    }

    float result = celeg_rope_unscaled_frequency(base, pair, rotary_dimension);
    if (scaling_mode == CelegRopeScalingNone ||
        scaling_mode == CelegRopeScalingDynamicNtk) {
        return result;
    }
    if (scaling_mode == CelegRopeScalingLinear) {
        return result / factor;
    }
    if (scaling_mode == CelegRopeScalingYarn) {
        constexpr float pi = 3.14159265358979323846f;
        const float original = static_cast<float>(original_context);
        const float log_base = log(base);
        const float low_dimension = static_cast<float>(rotary_dimension) *
            log(original / (beta_fast * 2.0f * pi)) / (2.0f * log_base);
        const float high_dimension = static_cast<float>(rotary_dimension) *
            log(original / (beta_slow * 2.0f * pi)) / (2.0f * log_base);
        float low = max(floor(low_dimension), 0.0f);
        float high = min(ceil(high_dimension),
                         static_cast<float>(rotary_dimension - 1u));
        if (low == high) high += 0.001f;
        const float ramp = clamp((static_cast<float>(pair) - low) / (high - low),
                                 0.0f, 1.0f);
        const float extrapolation = 1.0f - ramp;
        return result * (extrapolation + (1.0f - extrapolation) / factor);
    }
    if (scaling_mode == CelegRopeScalingLong) {
        const bool use_long = original_context > 0u &&
            context > static_cast<float>(original_context);
        const float rope_factor = use_long ? long_factors[pair] : short_factors[pair];
        return result / rope_factor;
    }
    if (scaling_mode == CelegRopeScalingLlama3) {
        constexpr float pi = 3.14159265358979323846f;
        const float original = static_cast<float>(original_context);
        const float wavelength = 2.0f * pi / result;
        if (wavelength < original / high_frequency_factor) return result;
        if (wavelength > original / low_frequency_factor) return result / factor;
        const float span = high_frequency_factor - low_frequency_factor;
        const float blend = span > 0.0f
            ? (wavelength * high_frequency_factor / original - 1.0f) / span
            : 0.0f;
        return result /
            (1.0f + clamp(blend, 0.0f, 1.0f) * (factor - 1.0f));
    }
    if (scaling_mode == CelegRopeScalingProportional) {
        const float fraction = rotary_fraction > 0.0f ? rotary_fraction : 1.0f;
        return pow(result, fraction) / factor;
    }
    return result;
}

inline float celeg_rope_attention_scale(uint scaling_mode, float attention_factor) {
    return scaling_mode == CelegRopeScalingYarn
        ? attention_factor * attention_factor
        : 1.0f;
}

kernel void celeg_rope_geometry_semantics_probe(
    device uint* pair_components [[buffer(0)]],
    device uint* axes [[buffer(1)]],
    constant uint& pair [[buffer(2)]],
    constant uint& pair_count [[buffer(3)]],
    constant uint& pairing_mode [[buffer(4)]],
    constant uint& section0 [[buffer(5)]],
    constant uint& section1 [[buffer(6)]],
    constant uint& interleaved [[buffer(7)]]) {
    const CelegRopePairComponents components =
        celeg_rope_pair_components(pair, pair_count, pairing_mode);
    pair_components[0] = components.first;
    pair_components[1] = components.second;
    axes[0] = celeg_mrope_axis_for_pair(pair, section0, section1, interleaved);
}

kernel void celeg_rope_scaling_semantics_probe(
    device float* output [[buffer(0)]],
    device const float* short_factors [[buffer(1)]],
    device const float* long_factors [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant float& rotary_fraction [[buffer(4)]],
    constant uint& pair [[buffer(5)]],
    constant uint& rotary_dimension [[buffer(6)]],
    constant uint& position [[buffer(7)]],
    constant uint& scaling_mode [[buffer(8)]],
    constant float& factor [[buffer(9)]],
    constant float& beta_fast [[buffer(10)]],
    constant float& beta_slow [[buffer(11)]],
    constant uint& original_context [[buffer(12)]],
    constant float& low_frequency_factor [[buffer(13)]],
    constant float& high_frequency_factor [[buffer(14)]],
    constant float& attention_factor [[buffer(15)]]) {
    output[0] = celeg_rope_scaled_frequency(
        theta, rotary_fraction, pair, rotary_dimension, position, scaling_mode,
        factor, beta_fast, beta_slow, original_context,
        low_frequency_factor, high_frequency_factor, short_factors, long_factors);
    output[1] = celeg_rope_attention_scale(scaling_mode, attention_factor);
}

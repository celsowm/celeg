#pragma once

#include "celeg/model/definition.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace celeg::metal_model_detail {

enum class MetalRopeScalingMode : std::uint32_t {
    None = 0u,
    Linear = 1u,
    DynamicNtk = 2u,
    Yarn = 3u,
    Long = 4u,
    Llama3 = 5u,
    Proportional = 6u,
};

struct MetalRopeScalingSpec {
    std::uint32_t mode = static_cast<std::uint32_t>(MetalRopeScalingMode::None);
    std::uint32_t original_context = 0u;
    float rotary_fraction = 1.0f;
    float factor = 1.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    float attention_factor = 1.0f;
    float low_frequency_factor = 1.0f;
    float high_frequency_factor = 1.0f;
};

static_assert(sizeof(MetalRopeScalingSpec) == 36u,
              "Metal RoPE scaling ABI must remain a packed 9x32-bit block");

struct MetalRopeScalingBinding {
    MetalRopeScalingSpec spec;
    const std::vector<float>* short_factors = nullptr;
    const std::vector<float>* long_factors = nullptr;

    bool scaled() const noexcept {
        return spec.mode != static_cast<std::uint32_t>(MetalRopeScalingMode::None);
    }
};

inline MetalRopeScalingBinding make_metal_rope_scaling_binding(
    const RopePositionSpec& rope) {
    MetalRopeScalingBinding binding;
    binding.spec.rotary_fraction = static_cast<float>(rope.rotary_fraction);

    std::visit([&](const auto& scaling) {
        using Scaling = std::decay_t<decltype(scaling)>;
        if constexpr (std::is_same_v<Scaling, NoRopeScaling>) {
            binding.spec.mode =
                static_cast<std::uint32_t>(MetalRopeScalingMode::None);
        } else if constexpr (std::is_same_v<Scaling, LinearRopeScaling>) {
            binding.spec.mode =
                static_cast<std::uint32_t>(MetalRopeScalingMode::Linear);
            binding.spec.factor = static_cast<float>(scaling.factor);
        } else if constexpr (std::is_same_v<Scaling, DynamicNtkRopeScaling>) {
            binding.spec.mode =
                static_cast<std::uint32_t>(MetalRopeScalingMode::DynamicNtk);
            binding.spec.factor = static_cast<float>(scaling.factor);
            binding.spec.original_context =
                static_cast<std::uint32_t>(scaling.original_context);
        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            binding.spec.mode =
                static_cast<std::uint32_t>(MetalRopeScalingMode::Yarn);
            binding.spec.factor = static_cast<float>(scaling.factor);
            binding.spec.beta_fast = static_cast<float>(scaling.beta_fast);
            binding.spec.beta_slow = static_cast<float>(scaling.beta_slow);
            binding.spec.attention_factor =
                static_cast<float>(scaling.attention_factor);
            binding.spec.original_context =
                static_cast<std::uint32_t>(scaling.original_context);
        } else if constexpr (std::is_same_v<Scaling, LongRopeScaling>) {
            binding.spec.mode =
                static_cast<std::uint32_t>(MetalRopeScalingMode::Long);
            binding.spec.original_context =
                static_cast<std::uint32_t>(scaling.original_context);
            binding.short_factors = &scaling.short_factors;
            binding.long_factors = &scaling.long_factors;
        } else if constexpr (std::is_same_v<Scaling, Llama3FrequencyScaling>) {
            binding.spec.mode =
                static_cast<std::uint32_t>(MetalRopeScalingMode::Llama3);
            binding.spec.factor = static_cast<float>(scaling.factor);
            binding.spec.original_context =
                static_cast<std::uint32_t>(scaling.original_context);
            binding.spec.low_frequency_factor =
                static_cast<float>(scaling.low_frequency_factor);
            binding.spec.high_frequency_factor =
                static_cast<float>(scaling.high_frequency_factor);
        } else if constexpr (std::is_same_v<Scaling, ProportionalRopeScaling>) {
            binding.spec.mode =
                static_cast<std::uint32_t>(MetalRopeScalingMode::Proportional);
            binding.spec.factor = static_cast<float>(scaling.factor);
        } else {
            static_assert(always_false_v<Scaling>,
                          "unhandled Metal RoPE scaling variant");
        }
    }, rope.scaling);

    return binding;
}

}

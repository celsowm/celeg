#include "backend/metal/model/runtime/rope_scaling.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace celeg;
using namespace celeg::metal_model_detail;

namespace {

bool close(float a, float b) {
    return std::abs(a - b) < 1.0e-6f;
}

RopePositionSpec rope_with(RopeScalingSpec scaling, double fraction = 1.0) {
    RopePositionSpec rope;
    rope.theta = 10000.0;
    rope.rotary_fraction = fraction;
    rope.scaling = std::move(scaling);
    return rope;
}

}

int main() {
    static_assert(sizeof(MetalRopeScalingSpec) == 36u);
    static_assert(static_cast<std::uint32_t>(MetalRopeScalingMode::None) == 0u);
    static_assert(static_cast<std::uint32_t>(MetalRopeScalingMode::Linear) == 1u);
    static_assert(static_cast<std::uint32_t>(MetalRopeScalingMode::DynamicNtk) == 2u);
    static_assert(static_cast<std::uint32_t>(MetalRopeScalingMode::Yarn) == 3u);
    static_assert(static_cast<std::uint32_t>(MetalRopeScalingMode::Long) == 4u);
    static_assert(static_cast<std::uint32_t>(MetalRopeScalingMode::Llama3) == 5u);
    static_assert(static_cast<std::uint32_t>(MetalRopeScalingMode::Proportional) == 6u);

    {
        const auto binding = make_metal_rope_scaling_binding(
            rope_with(NoRopeScaling{}, 0.5));
        assert(!binding.scaled());
        assert(close(binding.spec.rotary_fraction, 0.5f));
        assert(binding.short_factors == nullptr);
        assert(binding.long_factors == nullptr);
    }

    {
        const auto binding = make_metal_rope_scaling_binding(
            rope_with(LinearRopeScaling{2.5}));
        assert(binding.spec.mode == 1u);
        assert(close(binding.spec.factor, 2.5f));
    }

    {
        const auto binding = make_metal_rope_scaling_binding(
            rope_with(DynamicNtkRopeScaling{4.0, 8192}));
        assert(binding.spec.mode == 2u);
        assert(close(binding.spec.factor, 4.0f));
        assert(binding.spec.original_context == 8192u);
    }

    {
        YarnRopeScaling yarn;
        yarn.factor = 8.0;
        yarn.attention_factor = 1.25;
        yarn.beta_fast = 16.0;
        yarn.beta_slow = 2.0;
        yarn.original_context = 4096;
        const auto binding = make_metal_rope_scaling_binding(rope_with(yarn));
        assert(binding.spec.mode == 3u);
        assert(close(binding.spec.factor, 8.0f));
        assert(close(binding.spec.attention_factor, 1.25f));
        assert(close(binding.spec.beta_fast, 16.0f));
        assert(close(binding.spec.beta_slow, 2.0f));
        assert(binding.spec.original_context == 4096u);
    }

    {
        LongRopeScaling long_rope;
        long_rope.original_context = 4096;
        long_rope.short_factors = {1.0f, 1.5f, 2.0f};
        long_rope.long_factors = {2.0f, 3.0f, 4.0f};
        const auto binding = make_metal_rope_scaling_binding(rope_with(long_rope));
        assert(binding.spec.mode == 4u);
        assert(binding.spec.original_context == 4096u);
        assert(binding.short_factors != nullptr);
        assert(binding.long_factors != nullptr);
        assert(*binding.short_factors == long_rope.short_factors);
        assert(*binding.long_factors == long_rope.long_factors);
    }

    {
        Llama3FrequencyScaling llama3;
        llama3.factor = 8.0;
        llama3.original_context = 8192;
        llama3.low_frequency_factor = 1.0;
        llama3.high_frequency_factor = 4.0;
        const auto binding = make_metal_rope_scaling_binding(rope_with(llama3));
        assert(binding.spec.mode == 5u);
        assert(close(binding.spec.factor, 8.0f));
        assert(binding.spec.original_context == 8192u);
        assert(close(binding.spec.low_frequency_factor, 1.0f));
        assert(close(binding.spec.high_frequency_factor, 4.0f));
    }

    {
        const auto binding = make_metal_rope_scaling_binding(
            rope_with(ProportionalRopeScaling{1.75}, 0.625));
        assert(binding.spec.mode == 6u);
        assert(close(binding.spec.factor, 1.75f));
        assert(close(binding.spec.rotary_fraction, 0.625f));
    }

    return 0;
}

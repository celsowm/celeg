#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace celeg {

enum class CompiledMixer : unsigned char { Attention, ShortConvolution };
enum class CompiledFeedForward : unsigned char { Dense, MixtureOfExperts };

struct PerLayerInputPlan {
    bool enabled = false;
    int layer_count = 0;
    int input_size = 0;
    std::size_t packed_width = 0;
    float token_scale = 1.0f;
    float context_scale = 1.0f;
    float residual_scale = 1.0f;
    float norm_epsilon = 0.0f;
    ActivationKind activation = ActivationKind::GeluTanh;

    static PerLayerInputPlan derive(const ResolvedModel& model);
    std::size_t checked_elements(std::size_t rows) const;
    void validate() const;
};

// Immutable execution description produced before a backend starts serving.
// It contains no checkpoint or architecture probing state, so decode can use
// direct indices and function selection instead of format/architecture tests.
struct CompiledLayerProgram {
    CompiledMixer mixer;
    CompiledFeedForward feed_forward;
    std::vector<std::size_t> weight_request_indices;
};

struct CompiledModelProgram {
    std::string identity;
    std::size_t weight_request_count = 0;
    std::vector<CompiledLayerProgram> layers;
    std::vector<std::size_t> unlayered_weight_request_indices;
    PerLayerInputPlan per_layer_input;

    bool has_moe() const;
    void validate() const;
};

CompiledModelProgram build_model_program(const ResolvedModel& model);

} // namespace celeg

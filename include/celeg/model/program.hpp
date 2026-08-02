#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace celeg {

enum class CompiledMixer : unsigned char { Attention, ShortConvolution };
enum class CompiledFeedForward : unsigned char { Dense, MixtureOfExperts };

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
    std::string architecture_id;
    std::string source_format;
    std::size_t weight_request_count = 0;
    std::vector<CompiledLayerProgram> layers;
    std::vector<std::size_t> unlayered_weight_request_indices;

    bool has_moe() const;
    void validate() const;
};

CompiledModelProgram build_model_program(const ResolvedModel& model);

} // namespace celeg

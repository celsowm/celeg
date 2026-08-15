#pragma once

#include "celeg/model/graph.hpp"

#include <stdexcept>

namespace celeg::prefill_detail {

enum class LatentPrefillPath {
    Factorized,
    Projected,
};

inline LatentPrefillPath latent_prefill_path(const AttentionSpec& layout) {
    const auto* latent = layout.latent_state();
    if (!latent) {
        throw std::logic_error(
            "CUDA latent prefill selected without latent attention state");
    }
    return latent->factorized
        ? LatentPrefillPath::Factorized
        : LatentPrefillPath::Projected;
}

}

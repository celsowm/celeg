#pragma once

#include "celeg/model/program.hpp"

#include <variant>

namespace celeg {

inline bool cuda_can_fuse_mixer_residual(
    bool fused_residuals,
    const CompiledLayerProgram& semantics) {
    return fused_residuals &&
        !semantics.mixer_norm.after.has_value() &&
        !std::holds_alternative<std::monostate>(semantics.feed_forward);
}

}

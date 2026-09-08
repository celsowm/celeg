#pragma once

#include "celeg/model/program.hpp"

#include <span>
#include <vector>

namespace celeg {

/**
 * @brief Pure CPU MoE routing policy.
 *
 * No Metal, no I/O, no command buffer — unit-testable in isolation.
 */
struct MetalMoeRoute {
    std::vector<int> experts;
    std::vector<float> weights;
};

MetalMoeRoute route_metal_moe(
    const RouterProgram& program,
    std::span<const float> logits,
    std::span<const float> expert_bias);

}

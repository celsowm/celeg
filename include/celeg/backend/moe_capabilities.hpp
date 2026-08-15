#pragma once

#include "celeg/model/program.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace celeg {

struct MoeBackendCapabilities {
    bool grouped_selection = false;
    bool shared_experts = false;
    bool stacked_payload = false;
    bool fused_payload = false;
};

inline void validate_moe_backend_capabilities(
    const CompiledModelProgram& program,
    std::string_view backend,
    MoeBackendCapabilities capabilities) {
    for (const auto& layer : program.layers) {
        const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward);
        if (!moe) continue;
        if (std::holds_alternative<MoeGroupedTopKSelectionSpec>(
                moe->router.selection) &&
            !capabilities.grouped_selection) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support grouped MoE selection");
        }
        if (moe->shared && !capabilities.shared_experts) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support shared MoE experts");
        }
        if (moe->routed.payload.layout == MoePayloadLayout::Stacked &&
            !capabilities.stacked_payload) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support stacked MoE payloads");
        }
        if (moe->routed.payload.layout == MoePayloadLayout::Fused &&
            !capabilities.fused_payload) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support fused MoE payloads");
        }
    }
}

} // namespace celeg

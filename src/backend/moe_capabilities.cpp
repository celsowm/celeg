#include "celeg/backend/moe_capabilities.hpp"

#include "celeg/model/program.hpp"

#include <stdexcept>
#include <string>

namespace celeg {

void validate_moe_backend_capabilities(const CompiledModelProgram& program,
                                       std::string_view backend,
                                       MoeBackendCapabilities capabilities) {
    for (const auto& layer : program.layers) {
        if (!layer.moe) continue;
        const MoeLayerProgram& moe = *layer.moe;
        if (moe.router.selection == MoeSelectionKind::GroupedTopK &&
            !capabilities.grouped_selection) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support grouped MoE selection");
        }
        if (moe.shared && !capabilities.shared_experts) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support shared MoE experts");
        }
        if (moe.routed.payload.layout == MoePayloadLayout::Stacked &&
            !capabilities.stacked_payload) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support stacked MoE payloads");
        }
        if (moe.routed.payload.layout == MoePayloadLayout::Fused &&
            !capabilities.fused_payload) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support fused MoE payloads");
        }
    }
}

} // namespace celeg

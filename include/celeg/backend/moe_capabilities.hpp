#pragma once

#include <string_view>

namespace celeg {

struct CompiledModelProgram;

struct MoeBackendCapabilities {
    bool grouped_selection = false;
    bool shared_experts = false;
    bool stacked_payload = false;
    bool fused_payload = false;
};

void validate_moe_backend_capabilities(const CompiledModelProgram& program,
                                       std::string_view backend,
                                       MoeBackendCapabilities capabilities);

} // namespace celeg

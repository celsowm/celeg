#pragma once

// Phase 1 (plan section 1.2) explicit MoE quantization policy.
//
// The CUDA MoE experts only have a BF16 implementation today; INT4 / INT8
// expert kernels do not exist. Before Phase 1 the runtime silently accepted
// a safetensors MoE checkpoint under any --weight-mode and then crashed in
// model setup when it tried to dereference the null BF16 router pointer (the
// router got quantized into Int8/Int4 storage by the global WeightMode).
//
// The Phase 1 contract makes unsupported combinations fail at model
// construction with a clear diagnostic instead of crashing mid-load:
//
//   * safetensors + MoE + --weight-mode bf16   -> supported
//   * safetensors + MoE + --weight-mode int8   -> REJECTED (no expert kernel)
//   * safetensors + MoE + --weight-mode int4   -> REJECTED (no expert kernel)
//   * GGUF        + MoE + --weight-mode native -> supported (Q4_K / Q6_K
//                                                   experts on-device)
//
// Adding INT8/INT4 expert kernels is Phase 7 work (plan section 7.7); until
// then mixed or unsupported modes are reported honestly rather than silently
// misreported.

#include "celeg/model/execution/runtime_types.hpp"

#include <string>
#include <stdexcept>

namespace celeg {

// Throws std::invalid_argument when (mode, is_moe) is an unsupported MoE
// quantization pair. No-op otherwise. The diagnostic names both "MoE" and the
// offending mode so the caller can surface a useful error without having to
// re-interpret it.
inline void check_moe_quantization_policy(WeightMode mode, bool is_moe) {
    if (!is_moe) return;
    switch (mode) {
        case WeightMode::Bf16:
        case WeightMode::NativeGguf:
            return;
        case WeightMode::Int8:
        case WeightMode::Int4:
            break;
    }
    const char* mode_name =
        mode == WeightMode::Int8 ? "int8" :
        mode == WeightMode::Int4 ? "int4" : "unknown";
    throw std::invalid_argument(
        std::string("unsupported MoE quantization policy: --weight-mode ") +
        mode_name + " has no MoE expert kernel. MoE experts only support "
        "BF16 (safetensors) or native Q4_K/Q6_K blocks (GGUF). Mixed "
        "policies are Phase 7 work; see docs/ARCHITECTURE_RULES.md "
        "section 7.7.");
}

} // namespace celeg

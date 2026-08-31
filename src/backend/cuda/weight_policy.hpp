#pragma once

#include "runtime_types.hpp"

#include <stdexcept>
#include <string>

namespace celeg {

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

    throw std::invalid_argument(
        std::string("unsupported MoE quantization policy: --weight-mode ") +
        std::string(weight_mode_name(mode)) +
        " has no MoE expert kernel. MoE experts only support "
        "BF16 (safetensors) or native Q4_K/Q6_K blocks (GGUF). Mixed "
        "policies are Phase 7 work; see docs/ARCHITECTURE_EVIDENCE.md "
        "section 7.7.");
}

}

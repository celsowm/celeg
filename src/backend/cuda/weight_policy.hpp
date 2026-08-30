#pragma once


#include "celeg/backend/cuda/runtime_types.hpp"

#include <string>
#include <stdexcept>

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
    const char* mode_name =
        mode == WeightMode::Int8 ? "int8" :
        mode == WeightMode::Int4 ? "int4" : "unknown";
    throw std::invalid_argument(
        std::string("unsupported MoE quantization policy: --weight-mode ") +
        mode_name + " has no MoE expert kernel. MoE experts only support "
        "BF16 (safetensors) or native Q4_K/Q6_K blocks (GGUF). Mixed "
        "policies are Phase 7 work; see docs/ARCHITECTURE_EVIDENCE.md "
        "section 7.7.");
}

}

#pragma once

#include <cstdint>

namespace lfm {

enum class CpuWeightFormat : uint8_t {
    Q4Group32,
    Q4Group64,
};

enum class CpuKvCacheMode : uint8_t {
    Fp32,
    Bf16,
};

} // namespace lfm

#pragma once

#include "celeg/backend/cpu/gguf.hpp"

#include <cstddef>

namespace celeg::detail {

enum class LinearStorageKind { Q4, Int8, Gguf, Bf16 };

LinearStorageKind classify_linear_weight(const CpuLinearWeight& weight);
size_t linear_segment_rows(const CpuLinearMatrix& segment);

}

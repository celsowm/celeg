#pragma once

#include "lfm/backend/cpu/gguf.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

namespace lfm::detail {

float cpu_gguf_dot_avx2(const std::byte* packed_row, GgmlType type,
                        const CpuQ8KBlock* activation, size_t cols);
void cpu_quantize_q8k_avx2(const float* input, size_t cols,
                           CpuQ8KBlock* output);

} // namespace lfm::detail

#endif

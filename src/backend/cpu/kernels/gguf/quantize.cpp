#include "celeg/backend/cpu/gguf.hpp"
#include "../gguf_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace celeg {
namespace {

void quantize_q8k_scalar(const float* input, size_t cols, CpuQ8KBlock* output) {
    for (size_t block_index = 0; block_index < cols / 256; ++block_index) {
        const float* source = input + block_index * 256;
        CpuQ8KBlock& block = output[block_index];
        float maximum = 0.0f;
        for (size_t i = 0; i < 256; ++i) {
            maximum = std::max(maximum, std::abs(source[i]));
        }
        block.d = maximum == 0.0f ? 0.0f : maximum / 127.0f;
        const float inverse = block.d == 0.0f ? 0.0f : 1.0f / block.d;
        for (size_t group = 0; group < 16; ++group) {
            int sum = 0;
            for (size_t i = 0; i < 16; ++i) {
                const size_t index = group * 16 + i;
                int value = static_cast<int>(std::nearbyint(source[index] * inverse));
                value = std::clamp(value, -127, 127);
                block.qs[index] = static_cast<int8_t>(value);
                sum += value;
            }
            block.bsums[group] = static_cast<int16_t>(sum);
        }
    }
}

}

std::vector<CpuQ8KBlock> cpu_quantize_q8k(const float* input, size_t cols,
                                          CpuIsa isa) {
    if (!input || cols == 0 || (cols % 256) != 0) {
        throw std::invalid_argument("Q8_K activation must be 256-element aligned");
    }
    std::vector<CpuQ8KBlock> result(cols / 256);
    cpu_quantize_q8k_into(input, cols, isa, result.data());
    return result;
}

void cpu_quantize_q8k_into(const float* input, size_t cols, CpuIsa isa,
                           CpuQ8KBlock* output) {
    if (!input || !output || cols == 0 || (cols % 256) != 0) {
        throw std::invalid_argument("Q8_K activation must be 256-element aligned");
    }
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    if (isa != CpuIsa::Scalar) {
        detail::cpu_quantize_q8k_avx2(input, cols, output);
        return;
    }
#else
    (void)isa;
#endif
    quantize_q8k_scalar(input, cols, output);
}

}

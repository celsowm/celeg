#include "lfm/backend/cpu/gguf.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include "gguf_avx2.hpp"
#endif

namespace lfm {
namespace {

#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ6K) == 210);

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x03ffu;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            result = sign |
                static_cast<uint32_t>(127 - 15 - shift) << 23 |
                mantissa << 13;
        }
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | mantissa << 13;
    } else {
        result = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    return std::bit_cast<float>(result);
}

void q4k_scale_min(const BlockQ4K& block, int sub, uint8_t& scale,
                   uint8_t& minimum) {
    if (sub < 4) {
        scale = block.scales[sub] & 63;
        minimum = block.scales[sub + 4] & 63;
    } else {
        scale = static_cast<uint8_t>(
            (block.scales[sub + 4] & 0x0f) |
            ((block.scales[sub - 4] >> 6) << 4));
        minimum = static_cast<uint8_t>(
            (block.scales[sub + 4] >> 4) |
            ((block.scales[sub] >> 6) << 4));
    }
}

int q4k_value(const BlockQ4K& block, int col) {
    const int sub = col >> 5;
    const int within = col & 31;
    const uint8_t byte = block.qs[(sub >> 1) * 32 + within];
    return (sub & 1) ? (byte >> 4) : (byte & 0x0f);
}

int q6k_value(const BlockQ6K& block, int col) {
    const int half = col >> 7;
    const int index = col & 127;
    const int lane = index & 31;
    const int group = index >> 5;
    const uint8_t* ql = block.ql + half * 64;
    const uint8_t* qh = block.qh + half * 32;
    if (group == 0) {
        return (ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4);
    }
    if (group == 1) {
        return (ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4);
    }
    if (group == 2) {
        return (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
    }
    return (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
}

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

} // namespace

size_t CpuGgufMatrix::row_bytes() const {
    const GgmlTypeTrait trait = ggml_type_trait(type);
    if (trait.block_size <= 0 || cols % static_cast<uint32_t>(trait.block_size) != 0) {
        return 0;
    }
    return static_cast<size_t>(cols / static_cast<uint32_t>(trait.block_size)) *
           static_cast<size_t>(trait.type_size);
}

void CpuGgufMatrix::validate() const {
    if (type != GgmlType::Q4_K && type != GgmlType::Q6_K) {
        throw std::invalid_argument("CPU GGUF matrix requires Q4_K or Q6_K");
    }
    if (rows == 0 || cols == 0 || !data || row_bytes() == 0 ||
        bytes != static_cast<size_t>(rows) * row_bytes()) {
        throw std::invalid_argument("invalid CPU GGUF matrix");
    }
}

CpuLinearWeight CpuLinearWeight::from_q4(Q4GroupMatrix matrix) {
    matrix.validate();
    CpuLinearWeight result;
    result.rows = matrix.rows;
    result.cols = matrix.cols;
    result.segments.emplace_back(std::move(matrix));
    return result;
}

CpuLinearWeight CpuLinearWeight::from_gguf(CpuGgufMatrix matrix) {
    matrix.validate();
    CpuLinearWeight result;
    result.rows = matrix.rows;
    result.cols = matrix.cols;
    result.segments.emplace_back(matrix);
    return result;
}

size_t CpuLinearWeight::memory_bytes() const {
    size_t total = 0;
    for (const CpuLinearMatrix& segment : segments) {
        total += std::visit([](const auto& value) {
            return value.memory_bytes();
        }, segment);
    }
    return total;
}

bool CpuLinearWeight::gguf_native() const {
    return !segments.empty() &&
        std::all_of(segments.begin(), segments.end(), [](const CpuLinearMatrix& value) {
            return std::holds_alternative<CpuGgufMatrix>(value);
        });
}

void CpuLinearWeight::validate() const {
    if (rows == 0 || cols == 0 || segments.empty()) {
        throw std::invalid_argument("invalid CPU linear weight");
    }
    size_t segment_rows = 0;
    for (const CpuLinearMatrix& segment : segments) {
        std::visit([&](const auto& value) {
            value.validate();
            if (value.cols != cols) {
                throw std::invalid_argument("CPU linear segment width mismatch");
            }
            segment_rows += value.rows;
        }, segment);
    }
    if (segment_rows != rows) {
        throw std::invalid_argument("CPU linear segment row count mismatch");
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
#endif
    quantize_q8k_scalar(input, cols, output);
}

float cpu_gguf_dot_scalar(const std::byte* packed_row, GgmlType type,
                          const CpuQ8KBlock* activation, size_t cols) {
    if (!packed_row || !activation || cols == 0 || (cols % 256) != 0) {
        throw std::invalid_argument("invalid CPU GGUF dot arguments");
    }
    float total = 0.0f;
    const size_t blocks = cols / 256;
    if (type == GgmlType::Q4_K) {
        const auto* weights = reinterpret_cast<const BlockQ4K*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ4K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            const float d = fp16_to_float(weight.d);
            const float dmin = fp16_to_float(weight.dmin);
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                q4k_scale_min(weight, sub, scale, minimum);
                int dot = 0;
                const int base = sub * 32;
                for (int i = 0; i < 32; ++i) {
                    dot += q4k_value(weight, base + i) *
                           static_cast<int>(x.qs[base + i]);
                }
                const int sum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                block_total += d * static_cast<float>(scale * dot) -
                               dmin * static_cast<float>(minimum * sum);
            }
            total += x.d * block_total;
        }
        return total;
    }
    if (type == GgmlType::Q6_K) {
        const auto* weights = reinterpret_cast<const BlockQ6K*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ6K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            const float d = fp16_to_float(weight.d);
            int block_total = 0;
            for (int sub = 0; sub < 16; ++sub) {
                int dot = 0;
                const int base = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    dot += (q6k_value(weight, base + i) - 32) *
                           static_cast<int>(x.qs[base + i]);
                }
                block_total += static_cast<int>(weight.scales[sub]) * dot;
            }
            total += d * x.d * static_cast<float>(block_total);
        }
        return total;
    }
    throw std::invalid_argument("unsupported CPU GGUF dot type");
}

CpuGgufDotFunction select_cpu_gguf_dot_kernel(CpuIsa isa) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    if (isa != CpuIsa::Scalar) return detail::cpu_gguf_dot_avx2;
#endif
    return cpu_gguf_dot_scalar;
}

void cpu_gguf_dequantize_row(const CpuGgufMatrix& matrix, size_t row,
                             float* output) {
    matrix.validate();
    if (row >= matrix.rows || !output) {
        throw std::invalid_argument("invalid CPU GGUF embedding row");
    }
    const std::byte* packed = matrix.data + row * matrix.row_bytes();
    const size_t blocks = matrix.cols / 256;
    if (matrix.type == GgmlType::Q4_K) {
        const auto* weights = reinterpret_cast<const BlockQ4K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ4K& weight = weights[b];
            const float d = fp16_to_float(weight.d);
            const float dmin = fp16_to_float(weight.dmin);
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                q4k_scale_min(weight, sub, scale, minimum);
                for (int i = 0; i < 32; ++i) {
                    const int col = sub * 32 + i;
                    output[b * 256 + static_cast<size_t>(col)] =
                        d * static_cast<float>(scale * q4k_value(weight, col)) -
                        dmin * static_cast<float>(minimum);
                }
            }
        }
        return;
    }
    const auto* weights = reinterpret_cast<const BlockQ6K*>(packed);
    for (size_t b = 0; b < blocks; ++b) {
        const BlockQ6K& weight = weights[b];
        const float d = fp16_to_float(weight.d);
        for (int sub = 0; sub < 16; ++sub) {
            for (int i = 0; i < 16; ++i) {
                const int col = sub * 16 + i;
                output[b * 256 + static_cast<size_t>(col)] =
                    d * static_cast<float>(weight.scales[sub]) *
                    static_cast<float>(q6k_value(weight, col) - 32);
            }
        }
    }
}

} // namespace lfm

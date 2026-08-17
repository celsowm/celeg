#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/checkpoint/gguf_blocks.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include "gguf_avx2.hpp"
#endif

namespace celeg {
namespace {

using celeg::gguf_blocks::q4k_scale_min;

#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct BlockQ5K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

struct BlockQ2K {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};

struct BlockQ3K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};

struct BlockQ4_0 {
    uint16_t d;
    uint8_t qs[16];
};

struct BlockQ5_0 {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ5K) == 176);
static_assert(sizeof(BlockQ2K) == 84);
static_assert(sizeof(BlockQ3K) == 110);
static_assert(sizeof(BlockQ6K) == 210);
static_assert(sizeof(BlockQ4_0) == 18);
static_assert(sizeof(BlockQ5_0) == 22);
static_assert(sizeof(BlockQ8_0) == 34);

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
                static_cast<uint32_t>(127 - 14 - shift) << 23 |
                mantissa << 13;
        }
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | mantissa << 13;
    } else {
        result = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    return std::bit_cast<float>(result);
}

int q4k_value(const BlockQ4K& block, int col) {
    const int sub = col >> 5;
    const int within = col & 31;
    const uint8_t byte = block.qs[(sub >> 1) * 32 + within];
    return (sub & 1) ? (byte >> 4) : (byte & 0x0f);
}

int q5k_value(const BlockQ5K& block, int col) {
    const int sub = col >> 5;
    const int within = col & 31;
    const uint8_t packed = block.qs[(sub >> 1) * 32 + within];
    const int low = (sub & 1) ? (packed >> 4) : (packed & 0x0f);
    return low | (((block.qh[within] >> sub) & 1) << 4);
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

int q2k_value(const BlockQ2K& block, int col) {
    const int half = col / 128;
    const int local = col % 128;
    const int lane = local % 32;
    const int shift = (local / 32) * 2;
    return (block.qs[half * 32 + lane] >> shift) & 3;
}

void q3k_scales(const BlockQ3K& block, int8_t* output) {
    uint32_t words[4]{};
    uint32_t source[3]{};
    std::memcpy(source, block.scales, sizeof(block.scales));
    constexpr uint32_t low_nibble = 0x0f0f0f0f;
    constexpr uint32_t high_bits = 0x03030303;
    const uint32_t packed = source[2];
    words[2] = ((source[0] >> 4) & low_nibble) |
               (((packed >> 4) & high_bits) << 4);
    words[3] = ((source[1] >> 4) & low_nibble) |
               (((packed >> 6) & high_bits) << 4);
    words[0] = (source[0] & low_nibble) |
               (((packed >> 0) & high_bits) << 4);
    words[1] = (source[1] & low_nibble) |
               (((packed >> 2) & high_bits) << 4);
    for (int i = 0; i < 16; ++i) {
        output[i] = static_cast<int8_t>(
            reinterpret_cast<const uint8_t*>(words)[i]) - 32;
    }
}

int q3k_value(const BlockQ3K& block, int col) {
    const int half = col / 128;
    const int local = col % 128;
    const int lane = local % 32;
    const int shift = (local / 32) * 2;
    const int high = (block.hmask[lane] & (1u << (local / 32))) ? 0 : 4;
    return ((block.qs[half * 32 + lane] >> shift) & 3) - high;
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

}

size_t CpuGgufMatrix::row_bytes() const {
    const GgmlTypeTrait trait = ggml_type_trait(type);
    if (trait.block_size <= 0 || cols % static_cast<uint32_t>(trait.block_size) != 0) {
        return 0;
    }
    return static_cast<size_t>(cols / static_cast<uint32_t>(trait.block_size)) *
           static_cast<size_t>(trait.type_size);
}

void CpuGgufMatrix::validate() const {
    if (type != GgmlType::Q2_K && type != GgmlType::Q3_K &&
        type != GgmlType::Q4_0 && type != GgmlType::Q5_0 &&
        type != GgmlType::Q8_0 && type != GgmlType::Q4_K && type != GgmlType::Q5_K &&
        type != GgmlType::Q6_K) {
        throw std::invalid_argument("CPU GGUF matrix requires Q2_K, Q3_K, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K or Q6_K");
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

void CpuInt8Matrix::validate() const {
    if (rows == 0 || cols == 0 || !values || !scales ||
        values->size() != static_cast<size_t>(rows) * cols ||
        scales->size() != static_cast<size_t>(rows)) {
        throw std::runtime_error("invalid CPU INT8 matrix");
    }
}

CpuLinearWeight CpuLinearWeight::from_int8(CpuInt8Matrix matrix) {
    matrix.validate();
    CpuLinearWeight result;
    result.rows = matrix.rows;
    result.cols = matrix.cols;
    result.segments.emplace_back(std::move(matrix));
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
        throw std::invalid_argument("invalid CPU linear weight rows=" +
            std::to_string(rows) + " cols=" + std::to_string(cols) +
            " segments=" + std::to_string(segments.size()));
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
    if (type == GgmlType::Q2_K) {
        const auto* weights = reinterpret_cast<const BlockQ2K*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ2K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            int isum = 0;
            int summs = 0;
            for (int sub = 0; sub < 16; ++sub) {
                summs += x.bsums[sub] * (weight.scales[sub] >> 4);
                const int scale = weight.scales[sub] & 0x0f;
                for (int i = 0; i < 16; ++i) {
                    const int col = sub * 16 + i;
                    isum += scale * q2k_value(weight, col) * x.qs[col];
                }
            }
            total += x.d * (fp16_to_float(weight.d) * isum -
                            fp16_to_float(weight.dmin) * summs);
        }
        return total;
    }
    if (type == GgmlType::Q3_K) {
        const auto* weights = reinterpret_cast<const BlockQ3K*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ3K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            int8_t scales[16]{};
            q3k_scales(weight, scales);
            int isum = 0;
            for (int sub = 0; sub < 16; ++sub) {
                for (int i = 0; i < 16; ++i) {
                    const int col = sub * 16 + i;
                    isum += static_cast<int>(scales[sub]) *
                        q3k_value(weight, col) * x.qs[col];
                }
            }
            total += x.d * fp16_to_float(weight.d) * isum;
        }
        return total;
    }
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
                q4k_scale_min(sub, weight.scales, scale, minimum);
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
    if (type == GgmlType::Q5_K) {
        const auto* weights = reinterpret_cast<const BlockQ5K*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ5K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            const float d = fp16_to_float(weight.d);
            const float dmin = fp16_to_float(weight.dmin);
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                q4k_scale_min(sub, weight.scales, scale, minimum);
                int dot = 0;
                const int base = sub * 32;
                for (int i = 0; i < 32; ++i) {
                    dot += q5k_value(weight, base + i) *
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
    if (type == GgmlType::Q8_0) {
        const auto* weights = reinterpret_cast<const BlockQ8_0*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                const BlockQ8_0& weight = weights[b * 8 + static_cast<size_t>(sub)];
                const float d = fp16_to_float(weight.d);
                const int base = sub * 32;
                int dot = 0;
                for (int i = 0; i < 32; ++i) {
                    dot += static_cast<int>(weight.qs[i]) *
                           static_cast<int>(x.qs[base + i]);
                }
                block_total += d * static_cast<float>(dot);
            }
            total += x.d * block_total;
        }
        return total;
    }
    if (type == GgmlType::Q4_0) {
        const auto* weights = reinterpret_cast<const BlockQ4_0*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                const BlockQ4_0& weight = weights[b * 8 + static_cast<size_t>(sub)];
                const float d = fp16_to_float(weight.d);
                int dot = 0;
                for (int j = 0; j < 16; ++j) {
                    dot += (weight.qs[j] & 0x0f) *
                           static_cast<int>(x.qs[static_cast<size_t>(sub) * 32 + j]);
                    dot += (weight.qs[j] >> 4) *
                           static_cast<int>(x.qs[static_cast<size_t>(sub) * 32 + j + 16]);
                }
                const int bsum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                block_total += d * static_cast<float>(dot - 8 * bsum);
            }
            total += x.d * block_total;
        }
        return total;
    }
    if (type == GgmlType::Q5_0) {
        const auto* weights = reinterpret_cast<const BlockQ5_0*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                const BlockQ5_0& weight = weights[b * 8 + static_cast<size_t>(sub)];
                const float d = fp16_to_float(weight.d);
                uint32_t qh;
                std::memcpy(&qh, weight.qh, sizeof(qh));
                int dot = 0;
                for (int j = 0; j < 16; ++j) {
                    const int low = (weight.qs[j] & 0x0f) | (((qh >> j) & 1u) << 4);
                    const int high = (weight.qs[j] >> 4) |
                                     (((qh >> (j + 16)) & 1u) << 4);
                    dot += low *
                           static_cast<int>(x.qs[static_cast<size_t>(sub) * 32 + j]);
                    dot += high *
                           static_cast<int>(x.qs[static_cast<size_t>(sub) * 32 + j + 16]);
                }
                const int bsum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                block_total += d * static_cast<float>(dot - 16 * bsum);
            }
            total += x.d * block_total;
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

CpuGgufDot4Function select_cpu_gguf_dot4_kernel(CpuIsa isa) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    if (isa != CpuIsa::Scalar) return detail::cpu_gguf_dot4_avx2;
#else
    (void)isa;
#endif
    return nullptr;
}

void cpu_gguf_dequantize_row(const CpuGgufMatrix& matrix, size_t row,
                             float* output) {
    matrix.validate();
    if (row >= matrix.rows || !output) {
        throw std::invalid_argument("invalid CPU GGUF embedding row");
    }
    const std::byte* packed = matrix.data + row * matrix.row_bytes();
    const size_t blocks = matrix.cols / 256;
    if (matrix.type == GgmlType::Q2_K) {
        const auto* weights = reinterpret_cast<const BlockQ2K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ2K& weight = weights[b];
            const float d = fp16_to_float(weight.d);
            const float dmin = fp16_to_float(weight.dmin);
            for (int col = 0; col < 256; ++col) {
                const int sub = col / 16;
                output[b * 256 + static_cast<size_t>(col)] =
                    d * static_cast<float>(weight.scales[sub] & 0x0f) *
                        q2k_value(weight, col) -
                    dmin * static_cast<float>(weight.scales[sub] >> 4);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q3_K) {
        const auto* weights = reinterpret_cast<const BlockQ3K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ3K& weight = weights[b];
            int8_t scales[16]{};
            q3k_scales(weight, scales);
            const float d = fp16_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                const int sub = col / 16;
                output[b * 256 + static_cast<size_t>(col)] =
                    d * static_cast<float>(scales[sub]) *
                    q3k_value(weight, col);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q4_0) {
        const auto* weights = reinterpret_cast<const BlockQ4_0*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ4_0& weight = weights[b];
            const float d = fp16_to_float(weight.d);
            // GGML packs each block as two halves, not interleaved pairs:
            // qs[j] holds element j in its low nibble and element j+16 in
            // its high nibble (j in [0,16)).
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed_value = weight.qs[j];
                output[b * 32 + static_cast<size_t>(j)] =
                    d * static_cast<float>((packed_value & 0x0f) - 8);
                output[b * 32 + static_cast<size_t>(j + 16)] =
                    d * static_cast<float>((packed_value >> 4) - 8);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q5_0) {
        const auto* weights = reinterpret_cast<const BlockQ5_0*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ5_0& weight = weights[b];
            const float d = fp16_to_float(weight.d);
            uint32_t qh;
            std::memcpy(&qh, weight.qh, sizeof(qh));
            // Same split-half nibble layout as Q4_0; the high (5th) bit of
            // element j lives at bit j of qh, and of element j+16 at bit
            // j+16 of qh.
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed_value = weight.qs[j];
                const int high0 = (qh >> j) & 1;
                const int high1 = (qh >> (j + 16)) & 1;
                output[b * 32 + static_cast<size_t>(j)] =
                    d * static_cast<float>(((packed_value & 0x0f) | (high0 << 4)) - 16);
                output[b * 32 + static_cast<size_t>(j + 16)] =
                    d * static_cast<float>(((packed_value >> 4) | (high1 << 4)) - 16);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q8_0) {
        const auto* weights = reinterpret_cast<const BlockQ8_0*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ8_0& weight = weights[b];
            const float d = fp16_to_float(weight.d);
            for (int col = 0; col < 32; ++col) {
                output[b * 32 + static_cast<size_t>(col)] =
                    d * static_cast<float>(weight.qs[col]);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q4_K) {
        const auto* weights = reinterpret_cast<const BlockQ4K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ4K& weight = weights[b];
            const float d = fp16_to_float(weight.d);
            const float dmin = fp16_to_float(weight.dmin);
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                q4k_scale_min(sub, weight.scales, scale, minimum);
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
    if (matrix.type == GgmlType::Q5_K) {
        const auto* weights = reinterpret_cast<const BlockQ5K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ5K& weight = weights[b];
            const float d = fp16_to_float(weight.d);
            const float dmin = fp16_to_float(weight.dmin);
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                q4k_scale_min(sub, weight.scales, scale, minimum);
                for (int i = 0; i < 32; ++i) {
                    const int col = sub * 32 + i;
                    output[b * 256 + static_cast<size_t>(col)] =
                        d * static_cast<float>(scale * q5k_value(weight, col)) -
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

}

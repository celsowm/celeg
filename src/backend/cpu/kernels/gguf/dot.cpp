#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "blocks.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include "../gguf_avx2.hpp"
#endif

#include <stdexcept>
#include <string>

namespace celeg {
using namespace cpu_gguf_detail;

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
            total += x.d * (fp16_bits_to_float(weight.d) * isum -
                            fp16_bits_to_float(weight.dmin) * summs);
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
            total += x.d * fp16_bits_to_float(weight.d) * isum;
        }
        return total;
    }
    if (type == GgmlType::Q4_K) {
        const auto* weights = reinterpret_cast<const BlockQ4K*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ4K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            const float d = fp16_bits_to_float(weight.d);
            const float dmin = fp16_bits_to_float(weight.dmin);
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
            const float d = fp16_bits_to_float(weight.d);
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
            const float d = fp16_bits_to_float(weight.d);
            const float dmin = fp16_bits_to_float(weight.dmin);
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
    if (type == GgmlType::Q4_1) {
        const auto* weights = reinterpret_cast<const BlockQ4_1*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                const BlockQ4_1& weight = weights[b * 8 + static_cast<size_t>(sub)];
                const float d = fp16_bits_to_float(weight.d);
                const float m = fp16_bits_to_float(weight.dmin);
                int dot = 0;
                for (int j = 0; j < 16; ++j) {
                    const int low = weight.qs[j] & 0x0f;
                    const int high = weight.qs[j] >> 4;
                    dot += low *
                        static_cast<int>(x.qs[static_cast<size_t>(sub) * 32 + j]);
                    dot += high *
                        static_cast<int>(x.qs[static_cast<size_t>(sub) * 32 + j + 16]);
                }
                const int sum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                block_total += d * static_cast<float>(dot) +
                    m * static_cast<float>(sum);
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
                const float d = fp16_bits_to_float(weight.d);
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
                const float d = fp16_bits_to_float(weight.d);
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
                const float d = fp16_bits_to_float(weight.d);
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
    // The IQ formats share a shape: the packed magnitudes are small signed
    // integers, and every sub-block carries a float multiplier. Accumulating
    // the integer products per sub-block and applying the multiplier once
    // keeps the inner loop in integer arithmetic, exactly as the K-quants
    // above do.
    if (type == GgmlType::IQ2_S) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq2S*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq2S& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                // The two halves of a sub-block use the two nibbles of
                // scales[ib32], so they accumulate separately.
                int dot[2] = {0, 0};
                for (int i = 0; i < 32; ++i) {
                    const int col = ib32 * 32 + i;
                    dot[i / 16] += gguf_iq::iq2s_value(weight, col) *
                                   static_cast<int>(x.qs[col]);
                }
                block_total += gguf_iq::iq2s_sub_scale(weight, ib32, 0) *
                                   static_cast<float>(dot[0]) +
                               gguf_iq::iq2s_sub_scale(weight, ib32, 1) *
                                   static_cast<float>(dot[1]);
            }
            total += x.d * fp16_bits_to_float(weight.d) * block_total;
        }
        return total;
    }
    if (type == GgmlType::IQ3_XXS) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq3XXS*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq3XXS& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                int dot = 0;
                for (int i = 0; i < 32; ++i) {
                    const int col = ib32 * 32 + i;
                    dot += gguf_iq::iq3xxs_value(weight, col) *
                           static_cast<int>(x.qs[col]);
                }
                block_total += gguf_iq::iq3xxs_sub_scale(gguf_iq::iq3xxs_aux(weight, ib32)) *
                               static_cast<float>(dot);
            }
            total += x.d * fp16_bits_to_float(weight.d) * block_total;
        }
        return total;
    }
    if (type == GgmlType::IQ3_S) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq3S*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq3S& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                int dot = 0;
                for (int i = 0; i < 32; ++i) {
                    const int col = ib32 * 32 + i;
                    dot += gguf_iq::iq3s_value(weight, col) * static_cast<int>(x.qs[col]);
                }
                block_total += gguf_iq::iq3s_sub_scale(weight, ib32) *
                               static_cast<float>(dot);
            }
            total += x.d * fp16_bits_to_float(weight.d) * block_total;
        }
        return total;
    }
    if (type == GgmlType::IQ4_XS) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq4XS*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq4XS& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int ib = 0; ib < 8; ++ib) {
                int dot = 0;
                for (int i = 0; i < 32; ++i) {
                    const int col = ib * 32 + i;
                    dot += gguf_iq::iq4xs_value(weight, col) * static_cast<int>(x.qs[col]);
                }
                block_total += gguf_iq::iq4xs_sub_scale(weight, ib) *
                               static_cast<float>(dot);
            }
            total += x.d * fp16_bits_to_float(weight.d) * block_total;
        }
        return total;
    }
    if (type == GgmlType::IQ4_NL) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq4NL*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                const gguf_iq::BlockIq4NL& weight = weights[b * 8 + static_cast<size_t>(sub)];
                int dot = 0;
                for (int i = 0; i < 32; ++i) {
                    dot += gguf_iq::iq4nl_value(weight, i) *
                           static_cast<int>(x.qs[static_cast<size_t>(sub) * 32 + i]);
                }
                block_total += fp16_bits_to_float(weight.d) * static_cast<float>(dot);
            }
            total += x.d * block_total;
        }
        return total;
    }
    throw std::invalid_argument(std::string("unsupported CPU GGUF dot type: ") +
                                ggml_type_name(type));
}

CpuGgufDotFunction select_cpu_gguf_dot_kernel(CpuIsa isa) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    if (isa != CpuIsa::Scalar) return detail::cpu_gguf_dot_avx2;
#else
    (void)isa;
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

}

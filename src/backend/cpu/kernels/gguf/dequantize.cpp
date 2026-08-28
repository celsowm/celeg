#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "blocks.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace celeg {
using namespace cpu_gguf_detail;

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
            const float d = fp16_bits_to_float(weight.d);
            const float dmin = fp16_bits_to_float(weight.dmin);
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
            const float d = fp16_bits_to_float(weight.d);
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
            const float d = fp16_bits_to_float(weight.d);
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
            const float d = fp16_bits_to_float(weight.d);
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
    if (matrix.type == GgmlType::Q4_1) {
        const auto* weights = reinterpret_cast<const BlockQ4_1*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ4_1& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            const float m = fp16_bits_to_float(weight.dmin);
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed_value = weight.qs[j];
                output[b * 32 + static_cast<size_t>(j)] =
                    d * static_cast<float>(packed_value & 0x0f) + m;
                output[b * 32 + static_cast<size_t>(j + 16)] =
                    d * static_cast<float>(packed_value >> 4) + m;
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q8_0) {
        const auto* weights = reinterpret_cast<const BlockQ8_0*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ8_0& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
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
            const float d = fp16_bits_to_float(weight.d);
            const float dmin = fp16_bits_to_float(weight.dmin);
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
            const float d = fp16_bits_to_float(weight.d);
            const float dmin = fp16_bits_to_float(weight.dmin);
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
    if (matrix.type == GgmlType::IQ2_S) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq2S*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq2S& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                const float scale =
                    gguf_iq::iq2s_sub_scale(weight, col / 32, (col % 32) / 16);
                output[b * 256 + static_cast<size_t>(col)] =
                    d * scale * static_cast<float>(gguf_iq::iq2s_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ3_XXS) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq3XXS*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq3XXS& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                const float scale =
                    gguf_iq::iq3xxs_sub_scale(gguf_iq::iq3xxs_aux(weight, col / 32));
                output[b * 256 + static_cast<size_t>(col)] =
                    d * scale * static_cast<float>(gguf_iq::iq3xxs_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ3_S) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq3S*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq3S& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                output[b * 256 + static_cast<size_t>(col)] =
                    d * gguf_iq::iq3s_sub_scale(weight, col / 32) *
                    static_cast<float>(gguf_iq::iq3s_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ4_XS) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq4XS*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq4XS& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                output[b * 256 + static_cast<size_t>(col)] =
                    d * gguf_iq::iq4xs_sub_scale(weight, col / 32) *
                    static_cast<float>(gguf_iq::iq4xs_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ4_NL) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq4NL*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const gguf_iq::BlockIq4NL& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 32; ++col) {
                output[b * 32 + static_cast<size_t>(col)] =
                    d * static_cast<float>(gguf_iq::iq4nl_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type != GgmlType::Q6_K) {
        throw std::invalid_argument(std::string("unsupported CPU GGUF dequantize type: ") +
                                    ggml_type_name(matrix.type));
    }
    const auto* weights = reinterpret_cast<const BlockQ6K*>(packed);
    for (size_t b = 0; b < blocks; ++b) {
        const BlockQ6K& weight = weights[b];
        const float d = fp16_bits_to_float(weight.d);
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

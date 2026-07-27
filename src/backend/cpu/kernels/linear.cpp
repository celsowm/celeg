#include "lfm/backend/cpu/kernels.hpp"

#include "lfm/model/weights/quantization.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include "quantized_dot_avx2_msvc.hpp"
#endif

namespace lfm {

CpuLinearEngine::CpuLinearEngine(CpuIsa isa, CpuThreadPool& pool)
    : isa_(isa), pool_(&pool), dot_(select_q4_dot_kernel(isa)),
      q8_dot_(select_q4_q8_dot_kernel(isa)), dynamic_q8_(q8_dot_ != nullptr) {
    const CpuCapabilities caps = detect_cpu_capabilities();
    if (!caps.supports(isa)) throw std::invalid_argument("requested CPU ISA is not supported by this host");
}

void CpuLinearEngine::gemv(const Q4GroupMatrix& weight, const float* input,
                           float* output, float beta) const {
    weight.validate();
    if (!input || !output) throw std::invalid_argument("null CPU GEMV buffer");
    const size_t row_bytes = weight.packed_values_per_row();
    const size_t grain = std::max<size_t>(1, weight.rows / std::max<size_t>(1, pool_->size() * 8));
    if (dynamic_q8_) {
        const Q8GroupVector activation = quantize_float_groupwise_q8(input, weight.cols, weight.group_size);
        pool_->parallel_for(0, weight.rows, grain, [&](size_t begin, size_t end) {
            for (size_t row = begin; row < end; ++row) {
                const float value = q8_dot_(weight.values.data() + row * row_bytes,
                    weight.scales_bf16.data() + row * weight.groups_per_row,
                    activation.values.data(), activation.scales.data(), activation.sums.data(),
                    weight.cols, weight.group_size, weight.groups_per_row);
                output[row] = beta == 0.0f ? value : value + beta * output[row];
            }
        });
        return;
    }
    pool_->parallel_for(0, weight.rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const float value = dot_(weight.values.data() + row * row_bytes,
                weight.scales_bf16.data() + row * weight.groups_per_row,
                input, weight.cols, weight.group_size, weight.groups_per_row);
            output[row] = beta == 0.0f ? value : value + beta * output[row];
        }
    });
}

void CpuLinearEngine::gemm(const Q4GroupMatrix& weight, const float* input,
                           float* output, size_t rows, float beta) const {
    weight.validate();
    if ((!input || !output) && rows != 0) throw std::invalid_argument("null CPU GEMM buffer");
    if (rows == 0) return;
    const size_t row_bytes = weight.packed_values_per_row();
    std::vector<Q8GroupVector> quantized;
    if (dynamic_q8_) {
        quantized.reserve(rows);
        for (size_t r = 0; r < rows; ++r) {
            quantized.push_back(quantize_float_groupwise_q8(input + r * weight.cols,
                weight.cols, weight.group_size));
        }
    }
    // Keep an output-row tile resident while sweeping the activation batch.
    // The old flattened (row, output) schedule streamed the complete weight
    // matrix once for every prompt token.  This schedule reuses each packed
    // weight row for a small activation microtile before moving on.
    constexpr size_t output_tile = 4;
    const size_t tiles = (static_cast<size_t>(weight.rows) + output_tile - 1) / output_tile;
    const size_t grain = std::max<size_t>(1, tiles / std::max<size_t>(1, pool_->size() * 4));
    pool_->parallel_for(0, tiles, grain, [&](size_t begin, size_t end) {
        for (size_t tile = begin; tile < end; ++tile) {
            const size_t output_begin = tile * output_tile;
            const size_t output_end = std::min(output_begin + output_tile,
                                               static_cast<size_t>(weight.rows));
            for (size_t out = output_begin; out < output_end; ++out) {
                const uint8_t* packed = weight.values.data() + out * row_bytes;
                const uint16_t* scales = weight.scales_bf16.data() +
                    out * weight.groups_per_row;
                size_t r = 0;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
                if (dynamic_q8_ && isa_ == CpuIsa::Avx2 && weight.group_size == 32 &&
                    (weight.cols % 32) == 0) {
                    for (; r + 4 <= rows; r += 4) {
                        float values[4];
                        detail::q4_q8_dot4_avx2_msvc(packed, scales,
                            quantized[r], quantized[r + 1], quantized[r + 2], quantized[r + 3],
                            weight.cols, weight.group_size, weight.groups_per_row, values);
                        for (size_t lane = 0; lane < 4; ++lane) {
                            float& destination = output[(r + lane) * weight.rows + out];
                            destination = beta == 0.0f ? values[lane] :
                                values[lane] + beta * destination;
                        }
                    }
                }
#endif
                for (; r < rows; ++r) {
                    float value = 0.0f;
                    if (dynamic_q8_) {
                        const Q8GroupVector& activation = quantized[r];
                        value = q8_dot_(packed, scales, activation.values.data(),
                            activation.scales.data(), activation.sums.data(), weight.cols,
                            weight.group_size, weight.groups_per_row);
                    } else {
                        value = dot_(packed, scales, input + r * weight.cols,
                            weight.cols, weight.group_size, weight.groups_per_row);
                    }
                    float& destination = output[r * weight.rows + out];
                    destination = beta == 0.0f ? value : value + beta * destination;
                }
            }
        }
    });
}

void CpuLinearEngine::embedding(const Q4GroupMatrix& table, int32_t token,
                                float* output) const {
    table.validate();
    if (token < 0 || token >= static_cast<int32_t>(table.rows) || !output) {
        throw std::invalid_argument("invalid CPU embedding token");
    }
    dequantize_q4_row(table, static_cast<size_t>(token), output);
}

} // namespace lfm

#include "celeg/backend/cpu/linear.hpp"
#include "celeg/backend/cpu/kernel_backend.hpp"
#include "linear_dispatch.hpp"

#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include "quantized_dot_avx2_msvc.hpp"
#endif

namespace celeg {

CpuLinearEngine::CpuLinearEngine(const CpuKernelBackend& backend,
                                 CpuThreadPool& pool)
    : isa_(backend.isa), pool_(&pool) {
    if (backend.isa == CpuIsa::Auto || !backend.compiled) {
        throw std::invalid_argument("CpuLinearEngine requires a resolved CPU kernel backend");
    }
    dot_ = backend.kernels.q4_dot;
    q8_dot_ = backend.kernels.q4_q8_dot;
    gguf_dot_ = backend.kernels.gguf_dot;
    gguf_dot4_ = backend.kernels.gguf_dot4;
    dynamic_q8_ = backend.kernels.has_dynamic_q8();
}

void CpuLinearEngine::gemv(const CpuLinearWeight& weight, const float* input,
                           float* output, float beta) const {
    weight.validate();
    if (!input || !output) throw std::invalid_argument("null CPU GEMV buffer");
    const detail::LinearStorageKind kind = detail::classify_linear_weight(weight);
    if (kind == detail::LinearStorageKind::Q4) {
        size_t offset = 0;
        for (const CpuLinearMatrix& segment : weight.segments) {
            const Q4GroupMatrix& matrix = std::get<Q4GroupMatrix>(segment);
            gemv(matrix, input, output + offset, beta);
            offset += matrix.rows;
        }
        return;
    }
    if (kind == detail::LinearStorageKind::Int8) {
        size_t offset = 0;
        for (const CpuLinearMatrix& segment : weight.segments) {
            gemv_int8(std::get<CpuInt8Matrix>(segment), input, output + offset, beta);
            offset += detail::linear_segment_rows(segment);
        }
        return;
    }
    if (kind == detail::LinearStorageKind::Bf16) {
        size_t offset = 0;
        for (const CpuLinearMatrix& segment : weight.segments) {
            gemv_bf16(std::get<CpuBf16Matrix>(segment), input, output + offset, beta);
            offset += detail::linear_segment_rows(segment);
        }
        return;
    }
    const std::vector<CpuQ8KBlock> activation =
        cpu_quantize_q8k(input, weight.cols, isa_);
    size_t offset = 0;
    for (const CpuLinearMatrix& segment : weight.segments) {
        gemv_gguf(std::get<GgmlMatrixView>(segment), activation, output + offset, beta);
        offset += detail::linear_segment_rows(segment);
    }
}

void CpuLinearEngine::gemv_transpose(const CpuLinearWeight& weight,
                                     const float* input, float* output,
                                     size_t row_offset, size_t row_count) const {
    weight.validate();
    if (!input || !output || row_offset > weight.rows) {
        throw std::invalid_argument("invalid CPU transposed GEMV buffer");
    }
    if (row_count == 0) row_count = weight.rows - row_offset;
    if (row_offset + row_count > weight.rows) {
        throw std::invalid_argument("CPU transposed GEMV row range is invalid");
    }
    std::fill(output, output + weight.cols, 0.0f);
    size_t base = 0;
    for (const CpuLinearMatrix& segment : weight.segments) {
        const size_t begin = std::max(row_offset, base);
        const size_t end = std::min(row_offset + row_count, base +
                                    static_cast<size_t>(std::visit([](const auto& m) {
                                        return m.rows;
                                    }, segment)));
        if (begin < end) {
            const size_t local_begin = begin - base;
            const size_t local_end = end - base;
            std::vector<float> row(weight.cols);
            for (size_t r = local_begin; r < local_end; ++r) {
                std::visit([&](const auto& matrix) {
                    using Matrix = std::remove_cvref_t<decltype(matrix)>;
                    if constexpr (std::is_same_v<Matrix, Q4GroupMatrix>) {
                        dequantize_q4_row(matrix, r, row.data());
                    } else if constexpr (std::is_same_v<Matrix, CpuInt8Matrix>) {
                        for (size_t c = 0; c < weight.cols; ++c) {
                            row[c] = static_cast<float>(matrix.data()[r * weight.cols + c]) *
                                matrix.scales->at(r);
                        }
                    } else if constexpr (std::is_same_v<Matrix, CpuBf16Matrix>) {
                        for (size_t c = 0; c < weight.cols; ++c) {
                            row[c] = bf16_bits_to_float(
                                matrix.data()[r * weight.cols + c]);
                        }
                    } else {
                        ggml_decode_row(matrix, r, row.data());
                    }
                }, segment);
                const float scale = input[base + r - row_offset];
                for (size_t c = 0; c < weight.cols; ++c) output[c] += scale * row[c];
            }
        }
        base += std::visit([](const auto& m) { return static_cast<size_t>(m.rows); }, segment);
    }
}

void CpuLinearEngine::gemv_rows(const CpuLinearWeight& weight,
                                const float* input, float* output,
                                size_t row_offset, size_t row_count) const {
    weight.validate();
    if (!input || !output || row_offset > weight.rows ||
        row_offset + row_count > weight.rows) {
        throw std::invalid_argument("invalid CPU row-sliced GEMV arguments");
    }
    size_t global_row = 0;
    for (const CpuLinearMatrix& segment : weight.segments) {
        const size_t segment_rows =
            static_cast<size_t>(std::visit([](const auto& m) { return m.rows; }, segment));
        const size_t begin = std::max(row_offset, global_row);
        const size_t end = std::min(row_offset + row_count, global_row + segment_rows);
        if (begin < end) {
            std::vector<float> row(static_cast<size_t>(weight.cols));
            for (size_t r = begin; r < end; ++r) {
                const size_t local = r - global_row;
                std::visit([&](const auto& matrix) {
                    using Matrix = std::remove_cvref_t<decltype(matrix)>;
                    if constexpr (std::is_same_v<Matrix, Q4GroupMatrix>) {
                        dequantize_q4_row(matrix, local, row.data());
                    } else if constexpr (std::is_same_v<Matrix, CpuInt8Matrix>) {
                        for (size_t c = 0; c < static_cast<size_t>(weight.cols); ++c) {
                            row[c] = static_cast<float>(
                                         matrix.data()[local * weight.cols + c]) *
                                matrix.scales->at(local);
                        }
                    } else if constexpr (std::is_same_v<Matrix, CpuBf16Matrix>) {
                        for (size_t c = 0; c < static_cast<size_t>(weight.cols); ++c) {
                            row[c] = bf16_bits_to_float(
                                matrix.data()[local * weight.cols + c]);
                        }
                    } else {
                        ggml_decode_row(matrix, local, row.data());
                    }
                }, segment);
                float sum = 0.0f;
                for (size_t c = 0; c < static_cast<size_t>(weight.cols); ++c) {
                    sum += row[c] * input[c];
                }
                output[r - row_offset] = sum;
            }
        }
        global_row += segment_rows;
    }
}

void CpuLinearEngine::gemm(const CpuLinearWeight& weight, const float* input,
                           float* output, size_t rows, float beta) const {
    weight.validate();
    if ((!input || !output) && rows != 0) {
        throw std::invalid_argument("null CPU GEMM buffer");
    }
    if (rows == 0) return;
    const detail::LinearStorageKind kind = detail::classify_linear_weight(weight);
    if (kind == detail::LinearStorageKind::Q4) {
        size_t offset = 0;
        for (const CpuLinearMatrix& segment : weight.segments) {
            const Q4GroupMatrix& matrix = std::get<Q4GroupMatrix>(segment);
            std::vector<float> segment_output(rows * matrix.rows);
            gemm(matrix, input, segment_output.data(), rows, 0.0f);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < matrix.rows; ++column) {
                    float& destination = output[row * weight.rows + offset + column];
                    const float value = segment_output[row * matrix.rows + column];
                    destination = beta == 0.0f
                        ? value : value + beta * destination;
                }
            }
            offset += matrix.rows;
        }
        return;
    }
    if (kind == detail::LinearStorageKind::Int8) {
        size_t offset = 0;
        for (const CpuLinearMatrix& segment : weight.segments) {
            gemm_int8(std::get<CpuInt8Matrix>(segment), input, output, rows, beta,
                      weight.rows, offset);
            offset += detail::linear_segment_rows(segment);
        }
        return;
    }
    if (kind == detail::LinearStorageKind::Bf16) {
        size_t offset = 0;
        for (const CpuLinearMatrix& segment : weight.segments) {
            gemm_bf16(std::get<CpuBf16Matrix>(segment), input, output, rows, beta,
                      weight.rows, offset);
            offset += detail::linear_segment_rows(segment);
        }
        return;
    }
    std::vector<CpuQ8KBlock> activation;
    prepare_gguf_activation(input, rows, weight.cols, activation);
    gemm_gguf(activation, weight, output, rows, beta);
}

void CpuLinearEngine::embedding(const CpuLinearWeight& table, int32_t token,
                                float* output) const {
    table.validate();
    if (token < 0 || token >= static_cast<int32_t>(table.rows) || !output) {
        throw std::invalid_argument("invalid CPU embedding token");
    }
    size_t row = static_cast<size_t>(token);
    for (const CpuLinearMatrix& segment : table.segments) {
        const size_t segment_rows = std::visit(
            [](const auto& value) { return static_cast<size_t>(value.rows); },
            segment);
        if (row >= segment_rows) {
            row -= segment_rows;
            continue;
        }
        if (const auto* q4 = std::get_if<Q4GroupMatrix>(&segment)) {
            dequantize_q4_row(*q4, row, output);
        } else if (const auto* int8 = std::get_if<CpuInt8Matrix>(&segment)) {
            const int8_t* weights = int8->data() + row * int8->cols;
            for (size_t col = 0; col < int8->cols; ++col) {
                output[col] = static_cast<float>(weights[col]) * int8->scales->at(row);
            }
        } else if (const auto* bf16 = std::get_if<CpuBf16Matrix>(&segment)) {
            const uint16_t* weights = bf16->data() + row * bf16->cols;
            for (size_t col = 0; col < bf16->cols; ++col) {
                output[col] = bf16_bits_to_float(weights[col]);
            }
        } else {
            ggml_decode_row(std::get<GgmlMatrixView>(segment), row, output);
        }
        return;
    }
    throw std::logic_error("CPU embedding row was not found");
}

void CpuLinearEngine::gemv_raw(const float* weight, const float* input,
                               float* output, int n, int k) const {
    const size_t grain = std::max<size_t>(1, static_cast<size_t>(n) /
                                           std::max<size_t>(1, pool_->size() * 4));
    pool_->parallel_for(0, static_cast<size_t>(n), grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const float* w = weight + row * k;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
            if (isa_ != CpuIsa::Scalar) {
                output[row] = detail::f32_dot_avx2_msvc(w, input, static_cast<size_t>(k));
                continue;
            }
#endif
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) sum += w[i] * input[i];
            output[row] = sum;
        }
    });
}

void CpuLinearEngine::gemm_raw(const float* weight, const float* input,
                               float* output, size_t rows, int n, int k) const {
    const size_t grain = std::max<size_t>(1, rows /
                                           std::max<size_t>(1, pool_->size() * 2));
    pool_->parallel_for(0, rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const float* row_input = input + row * static_cast<size_t>(k);
            float* row_output = output + row * static_cast<size_t>(n);
            for (int out = 0; out < n; ++out) {
                const float* w = weight + static_cast<size_t>(out) * k;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
                if (isa_ != CpuIsa::Scalar) {
                    row_output[out] = detail::f32_dot_avx2_msvc(
                        w, row_input, static_cast<size_t>(k));
                    continue;
                }
#endif
                float sum = 0.0f;
                for (int col = 0; col < k; ++col) sum += w[col] * row_input[col];
                row_output[out] = sum;
            }
        }
    });
}

}

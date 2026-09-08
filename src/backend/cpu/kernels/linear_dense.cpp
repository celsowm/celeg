#include "celeg/backend/cpu/linear.hpp"

#include "celeg/model/weights/quantization.hpp"

#include <algorithm>

namespace celeg {

void CpuLinearEngine::gemv_int8(const CpuInt8Matrix& matrix, const float* input,
                                float* output, float beta) const {
    const size_t grain = std::max<size_t>(
        1, matrix.rows / std::max<size_t>(1, pool_->size() * 8));
    pool_->parallel_for(0, matrix.rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            float value = 0.0f;
            const int8_t* weights = matrix.data() + row * matrix.cols;
            for (size_t col = 0; col < matrix.cols; ++col) {
                value += static_cast<float>(weights[col]) * input[col];
            }
            value *= matrix.scales->at(row);
            float& destination = output[row];
            destination = beta == 0.0f ? value : value + beta * destination;
        }
    });
}

void CpuLinearEngine::gemv_bf16(const CpuBf16Matrix& matrix, const float* input,
                                float* output, float beta) const {
    const size_t grain = std::max<size_t>(
        1, matrix.rows / std::max<size_t>(1, pool_->size() * 8));
    pool_->parallel_for(0, matrix.rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            float value = 0.0f;
            const uint16_t* weights = matrix.data() + row * matrix.cols;
            for (size_t col = 0; col < matrix.cols; ++col) {
                value += bf16_bits_to_float(weights[col]) * input[col];
            }
            float& destination = output[row];
            destination = beta == 0.0f ? value : value + beta * destination;
        }
    });
}

void CpuLinearEngine::gemm_int8(const CpuInt8Matrix& matrix, const float* input,
                                float* output, size_t rows, float beta,
                                size_t output_stride, size_t output_base) const {
    const size_t grain = std::max<size_t>(
        1, rows / std::max<size_t>(1, pool_->size() * 4));
    pool_->parallel_for(0, rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            float* destination = output + row * output_stride + output_base;
            const float* activation = input + row * matrix.cols;
            for (size_t out = 0; out < matrix.rows; ++out) {
                float value = 0.0f;
                const int8_t* weights = matrix.data() + out * matrix.cols;
                for (size_t col = 0; col < matrix.cols; ++col) {
                    value += static_cast<float>(weights[col]) * activation[col];
                }
                const float previous = destination[out];
                destination[out] = matrix.scales->at(out) * value;
                if (beta != 0.0f) destination[out] += beta * previous;
            }
        }
    });
}

void CpuLinearEngine::gemm_bf16(const CpuBf16Matrix& matrix, const float* input,
                                float* output, size_t rows, float beta,
                                size_t output_stride, size_t output_base) const {
    const size_t grain = std::max<size_t>(
        1, rows / std::max<size_t>(1, pool_->size() * 4));
    pool_->parallel_for(0, rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            float* destination = output + row * output_stride + output_base;
            const float* activation = input + row * matrix.cols;
            for (size_t out = 0; out < matrix.rows; ++out) {
                float value = 0.0f;
                const uint16_t* weights = matrix.data() + out * matrix.cols;
                for (size_t col = 0; col < matrix.cols; ++col) {
                    value += bf16_bits_to_float(weights[col]) * activation[col];
                }
                const float previous = destination[out];
                destination[out] = value;
                if (beta != 0.0f) destination[out] += beta * previous;
            }
        }
    });
}

}

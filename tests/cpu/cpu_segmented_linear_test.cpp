#include "celeg/backend/cpu/kernels.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <vector>

int main() {
    constexpr size_t rows = 37;
    constexpr size_t cols = 128;
    constexpr size_t split_rows = 13;
    constexpr size_t batch = 5;

    std::vector<float> weights(rows * cols);
    std::vector<float> input(cols);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = std::sin(static_cast<float>(i) * 0.03f);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::cos(static_cast<float>(i) * 0.07f) * 1.3f;
    }

    const auto first = celeg::quantize_float_groupwise_q4(
        weights.data(), split_rows, cols, 32);
    const auto second = celeg::quantize_float_groupwise_q4(
        weights.data() + split_rows * cols, rows - split_rows, cols, 32);

    celeg::CpuLinearWeight segmented;
    segmented.rows = rows;
    segmented.cols = cols;
    segmented.segments.emplace_back(first);
    segmented.segments.emplace_back(second);
    segmented.validate();

    celeg::CpuThreadPool pool(6);
    celeg::CpuLinearEngine linear(
        celeg::detect_cpu_capabilities().best_isa(), pool);

    std::vector<float> actual(rows);
    std::vector<float> expected(rows);
    linear.gemv(segmented, input.data(), actual.data());
    linear.gemv(first, input.data(), expected.data());
    linear.gemv(second, input.data(), expected.data() + split_rows);
    for (size_t row = 0; row < rows; ++row) {
        CELEG_TEST_CHECK(std::abs(actual[row] - expected[row]) < 1e-5f);
    }

    std::vector<float> batch_input(batch * cols);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < cols; ++c) {
            batch_input[b * cols + c] = input[c] *
                (1.0f + 0.05f * static_cast<float>(b));
        }
    }

    std::vector<float> actual_batch(batch * rows);
    std::vector<float> expected_batch(batch * rows);
    std::vector<float> first_batch(batch * split_rows);
    std::vector<float> second_batch(batch * (rows - split_rows));
    linear.gemm(segmented, batch_input.data(), actual_batch.data(), batch);
    linear.gemm(first, batch_input.data(), first_batch.data(), batch);
    linear.gemm(second, batch_input.data(), second_batch.data(), batch);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t row = 0; row < split_rows; ++row) {
            expected_batch[b * rows + row] = first_batch[b * split_rows + row];
        }
        for (size_t row = 0; row < rows - split_rows; ++row) {
            expected_batch[b * rows + split_rows + row] =
                second_batch[b * (rows - split_rows) + row];
        }
    }
    for (size_t i = 0; i < actual_batch.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(actual_batch[i] - expected_batch[i]) < 1e-5f);
    }

    /// Row-sliced GEMV: `gemv_rows` must equal slicing a full GEMV over the
    /// same quantized segments and must cross a Q4 segment boundary (this is
    /// the MLA value-decompression primitive; a hand-rolled transpose variant
    /// once silently produced a column-space contraction instead).
    {
        const size_t slice_offset = 7;
        const size_t slice_rows = 9;
        std::vector<float> sliced(slice_rows);
        linear.gemv_rows(segmented, input.data(), sliced.data(),
                         slice_offset, slice_rows);
        for (size_t j = 0; j < slice_rows; ++j) {
            /// Q4 GEMV uses a fused quant path while gemv_rows dequantizes a
            /// row and dots in FP32; the two converge only up to Q4-row
            /// rounding, so compare under the suite's Q4 tolerance.
            CELEG_TEST_CHECK(
                std::abs(sliced[j] - actual[slice_offset + j]) < 0.08f);
        }
        /// Row slices must never bleed into `output` beyond `row_count`.
        std::vector<float> padded(slice_rows + 1, -1.0f);
        linear.gemv_rows(segmented, input.data(), padded.data(),
                         slice_offset, slice_rows);
        CELEG_TEST_CHECK(padded[slice_rows] == -1.0f);
    }
}

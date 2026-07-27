#include "lfm/backend/cpu/kernels.hpp"

#include "lfm/model/weights/quantization.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include "quantized_dot_avx2_msvc.hpp"
#endif

namespace lfm {
namespace {

// A whole batch of Q8-quantized activation rows in three contiguous slabs.
// The previous shape -- std::vector<Q8GroupVector>, i.e. three heap vectors
// per row -- cost hundreds of allocations per GEMM call and scattered the
// rows across memory that the microkernel then re-reads once per output row.
struct Q8ActivationBatch {
    std::vector<int8_t> values;
    std::vector<float> scales;
    std::vector<int32_t> sums;
    size_t cols = 0;
    size_t groups = 0;

    const int8_t* row_values(size_t row) const { return values.data() + row * cols; }
    const float* row_scales(size_t row) const { return scales.data() + row * groups; }
    const int32_t* row_sums(size_t row) const { return sums.data() + row * groups; }
};

} // namespace

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
        // Decode issues one GEMV per projection per layer per token, so the
        // activation scratch is reused instead of reallocated on every call.
        thread_local Q8ActivationBatch scratch;
        scratch.cols = weight.cols;
        scratch.groups = weight.groups_per_row;
        scratch.values.resize(scratch.cols);
        scratch.scales.resize(scratch.groups);
        scratch.sums.resize(scratch.groups);
        quantize_float_groupwise_q8_into(input, weight.cols, weight.group_size,
            scratch.values.data(), scratch.scales.data(), scratch.sums.data());
        // Bind the scratch through automatic-storage locals. A thread_local is
        // never captured by a lambda -- the body would resolve it against each
        // *worker* thread's own (empty) instance instead of this one.
        const int8_t* const activation_values = scratch.values.data();
        const float* const activation_scales = scratch.scales.data();
        const int32_t* const activation_sums = scratch.sums.data();
        pool_->parallel_for(0, weight.rows, grain, [&](size_t begin, size_t end) {
            for (size_t row = begin; row < end; ++row) {
                const float value = q8_dot_(weight.values.data() + row * row_bytes,
                    weight.scales_bf16.data() + row * weight.groups_per_row,
                    activation_values, activation_scales, activation_sums,
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
    Q8ActivationBatch batch;
    if (dynamic_q8_) {
        batch.cols = weight.cols;
        batch.groups = weight.groups_per_row;
        batch.values.resize(rows * batch.cols);
        batch.scales.resize(rows * batch.groups);
        batch.sums.resize(rows * batch.groups);
        // Quantization is per-row independent, so run it on the pool instead of
        // serially on the calling thread ahead of the parallel GEMM.
        const size_t quantize_grain =
            std::max<size_t>(1, rows / std::max<size_t>(1, pool_->size() * 4));
        pool_->parallel_for(0, rows, quantize_grain, [&](size_t begin, size_t end) {
            for (size_t r = begin; r < end; ++r) {
                quantize_float_groupwise_q8_into(input + r * batch.cols, batch.cols,
                    weight.group_size, batch.values.data() + r * batch.cols,
                    batch.scales.data() + r * batch.groups,
                    batch.sums.data() + r * batch.groups);
            }
        });
    }
    // A tile of output rows is held in L1 while the activation batch streams
    // past it once.  The activation-row loop must be the OUTER one: with the
    // output row outermost the whole activation batch is re-read once per
    // output row (tens of GB per prefill chunk, which saturates L3), whereas
    // this order re-reads it once per tile.  The packed weight tile is only
    // `output_tile * row_bytes` and stays resident across the sweep.
    constexpr size_t output_tile = 8;
    const size_t tiles = (static_cast<size_t>(weight.rows) + output_tile - 1) / output_tile;
    const size_t grain = std::max<size_t>(1, tiles / std::max<size_t>(1, pool_->size() * 4));
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    // Every MSVC x86 ISA selection maps to the AVX2 Q4xQ8 kernel, so the
    // four-lane microkernel applies whenever dynamic Q8 activations are in use.
    const bool use_dot4 =
        dynamic_q8_ && weight.group_size == 32 && (weight.cols % 32) == 0;
#endif
    pool_->parallel_for(0, tiles, grain, [&](size_t begin, size_t end) {
        for (size_t tile = begin; tile < end; ++tile) {
            const size_t output_begin = tile * output_tile;
            const size_t output_end = std::min(output_begin + output_tile,
                                               static_cast<size_t>(weight.rows));
            size_t r = 0;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
            if (use_dot4) {
                for (; r + 4 <= rows; r += 4) {
                    for (size_t out = output_begin; out < output_end; ++out) {
                        float values[4];
                        detail::q4_q8_dot4_avx2_msvc(
                            weight.values.data() + out * row_bytes,
                            weight.scales_bf16.data() + out * weight.groups_per_row,
                            batch.row_values(r), batch.cols,
                            batch.row_scales(r), batch.row_sums(r), batch.groups,
                            weight.cols, weight.group_size, weight.groups_per_row, values);
                        for (size_t lane = 0; lane < 4; ++lane) {
                            float& destination = output[(r + lane) * weight.rows + out];
                            destination = beta == 0.0f ? values[lane] :
                                values[lane] + beta * destination;
                        }
                    }
                }
            }
#endif
            for (; r < rows; ++r) {
                for (size_t out = output_begin; out < output_end; ++out) {
                    const uint8_t* packed = weight.values.data() + out * row_bytes;
                    const uint16_t* scales = weight.scales_bf16.data() +
                        out * weight.groups_per_row;
                    const float value = dynamic_q8_
                        ? q8_dot_(packed, scales, batch.row_values(r),
                              batch.row_scales(r), batch.row_sums(r), weight.cols,
                              weight.group_size, weight.groups_per_row)
                        : dot_(packed, scales, input + r * weight.cols,
                              weight.cols, weight.group_size, weight.groups_per_row);
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

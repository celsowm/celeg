#include "celeg/backend/cpu/kernels.hpp"

#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include "quantized_dot_avx2_msvc.hpp"
#endif

namespace celeg {
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

void quantize_gguf_rows(CpuThreadPool& pool, CpuIsa isa, const float* input,
                        size_t rows, size_t cols,
                        std::vector<CpuQ8KBlock>& activation) {
    const size_t blocks_per_row = cols / 256;
    activation.resize(rows * blocks_per_row);
    const size_t grain = std::max<size_t>(
        1, rows / std::max<size_t>(1, pool.size() * 4));
    pool.parallel_for(0, rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            cpu_quantize_q8k_into(input + row * cols, cols, isa,
                                  activation.data() + row * blocks_per_row);
        }
    });
}

} // namespace

CpuLinearEngine::CpuLinearEngine(CpuIsa isa, CpuThreadPool& pool)
    : isa_(isa), pool_(&pool), dot_(select_q4_dot_kernel(isa)),
      q8_dot_(select_q4_q8_dot_kernel(isa)),
      gguf_dot_(select_cpu_gguf_dot_kernel(isa)),
      gguf_dot4_(select_cpu_gguf_dot4_kernel(isa)),
      dynamic_q8_(q8_dot_ != nullptr) {
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

void CpuLinearEngine::gemv(const CpuLinearWeight& weight, const float* input,
                           float* output, float beta) const {
    weight.validate();
    if (!input || !output) throw std::invalid_argument("null CPU GEMV buffer");
    if (weight.segments.size() == 1 &&
        std::holds_alternative<Q4GroupMatrix>(weight.segments.front())) {
        gemv(std::get<Q4GroupMatrix>(weight.segments.front()), input, output, beta);
        return;
    }

    const std::vector<CpuQ8KBlock> activation =
        cpu_quantize_q8k(input, weight.cols, isa_);
    size_t output_offset = 0;
    for (const CpuLinearMatrix& segment : weight.segments) {
        if (!std::holds_alternative<CpuGgufMatrix>(segment)) {
            throw std::logic_error("mixed internal-Q4/GGUF weight is unsupported");
        }
        const CpuGgufMatrix& matrix = std::get<CpuGgufMatrix>(segment);
        const size_t grain = std::max<size_t>(
            1, matrix.rows / std::max<size_t>(1, pool_->size() * 8));
        pool_->parallel_for(0, matrix.rows, grain, [&](size_t begin, size_t end) {
            for (size_t row = begin; row < end; ++row) {
                const float value = gguf_dot_(
                    matrix.data + row * matrix.row_bytes(), matrix.type,
                    activation.data(), matrix.cols);
                float& destination = output[output_offset + row];
                destination = beta == 0.0f
                    ? value : value + beta * destination;
            }
        });
        output_offset += matrix.rows;
    }
}

void CpuLinearEngine::gemm(const CpuLinearWeight& weight, const float* input,
                           float* output, size_t rows, float beta) const {
    weight.validate();
    if ((!input || !output) && rows != 0) {
        throw std::invalid_argument("null CPU GEMM buffer");
    }
    if (rows == 0) return;
    if (weight.segments.size() == 1 &&
        std::holds_alternative<Q4GroupMatrix>(weight.segments.front())) {
        gemm(std::get<Q4GroupMatrix>(weight.segments.front()),
             input, output, rows, beta);
        return;
    }

    std::vector<CpuQ8KBlock> activation;
    prepare_gguf_activation(input, rows, weight.cols, activation);
    gemm_gguf(activation, weight, output, rows, beta);
}

void CpuLinearEngine::prepare_gguf_activation(
    const float* input, size_t rows, size_t cols,
    std::vector<CpuQ8KBlock>& activation) const {
    if ((!input && rows != 0) || cols == 0 || (cols % 256) != 0) {
        throw std::invalid_argument("invalid GGUF activation shape");
    }
    quantize_gguf_rows(*pool_, isa_, input, rows, cols, activation);
}

void CpuLinearEngine::gemm_gguf(std::span<const CpuQ8KBlock> activation,
                                const CpuLinearWeight& weight, float* output,
                                size_t rows, float beta) const {
    weight.validate();
    if (!weight.gguf_native()) {
        throw std::invalid_argument("prequantized GEMM requires native GGUF weights");
    }
    if ((!output && rows != 0) || rows > 0 &&
        activation.size() < rows * (weight.cols / 256)) {
        throw std::invalid_argument("invalid prequantized GGUF activation");
    }
    const size_t blocks_per_row = weight.cols / 256;

    size_t output_offset = 0;
    for (const CpuLinearMatrix& segment : weight.segments) {
        const CpuGgufMatrix& matrix = std::get<CpuGgufMatrix>(segment);
        // Q4_K/Q6_K rows are small enough that a 16-row tile stays resident
        // in L1 while amortizing thread-pool scheduling across prefill rows.
        constexpr size_t output_tile = 16;
        const size_t tiles =
            (static_cast<size_t>(matrix.rows) + output_tile - 1) / output_tile;
        const size_t grain = std::max<size_t>(
            1, tiles / std::max<size_t>(1, pool_->size() * 4));
        pool_->parallel_for(0, tiles, grain, [&](size_t begin, size_t end) {
            for (size_t tile = begin; tile < end; ++tile) {
                const size_t output_begin = tile * output_tile;
                const size_t output_end = std::min(
                    output_begin + output_tile,
                    static_cast<size_t>(matrix.rows));
                for (size_t out = output_begin; out < output_end; ++out) {
                    size_t activation_row = 0;
                    if (gguf_dot4_) {
                        for (; activation_row + 4 <= rows; activation_row += 4) {
                            float values[4];
                            gguf_dot4_(matrix.data + out * matrix.row_bytes(), matrix.type,
                                       activation.data() + activation_row * blocks_per_row,
                                       matrix.cols, values);
                            for (size_t lane = 0; lane < 4; ++lane) {
                                float& destination = output[(activation_row + lane) * weight.rows +
                                                             output_offset + out];
                                destination = beta == 0.0f ? values[lane] :
                                    values[lane] + beta * destination;
                            }
                        }
                    }
                    for (; activation_row < rows; ++activation_row) {
                        const float value = gguf_dot_(
                            matrix.data + out * matrix.row_bytes(), matrix.type,
                            activation.data() + activation_row * blocks_per_row,
                            matrix.cols);
                        float& destination = output[activation_row * weight.rows +
                                                     output_offset + out];
                        destination = beta == 0.0f ? value : value + beta * destination;
                    }
                }
            }
        });
        output_offset += matrix.rows;
    }
}

void CpuLinearEngine::gemm_grouped(std::span<const CpuGroupedGemmJob> jobs,
                                   const float* input, float* output) const {
    if (jobs.empty()) return;
    if (!input || !output) throw std::invalid_argument("null grouped CPU GEMM buffer");

    // Routed experts in LFM share shape and use one native GGUF segment. Keep
    // the unusual/mixed case correct through the ordinary GEMM path.
    const CpuLinearWeight* first = jobs.front().weight;
    if (!first) throw std::invalid_argument("grouped CPU GEMM has null weight");
    first->validate();
    const bool native = first->segments.size() == 1 &&
        std::holds_alternative<CpuGgufMatrix>(first->segments.front());
    const size_t input_width = first->cols;
    const size_t output_width = first->rows;
    size_t total_rows = 0;
    for (const CpuGroupedGemmJob& job : jobs) {
        if (!job.weight) throw std::invalid_argument("grouped CPU GEMM has null job weight");
        job.weight->validate();
        if (job.weight->cols != input_width || job.weight->rows != output_width ||
            job.weight->segments.size() != 1 ||
            !std::holds_alternative<CpuGgufMatrix>(job.weight->segments.front())) {
            for (const CpuGroupedGemmJob& fallback : jobs) {
                gemm(*fallback.weight, input + fallback.row_offset * input_width,
                     output + fallback.row_offset * output_width, fallback.rows);
            }
            return;
        }
        total_rows = std::max(total_rows, job.row_offset + job.rows);
    }
    if (!native || total_rows == 0) return;

    const size_t blocks_per_row = input_width / 256;
    std::vector<CpuQ8KBlock> activation;
    quantize_gguf_rows(*pool_, isa_, input, total_rows, input_width, activation);

    struct Tile { size_t job; size_t begin; size_t end; };
    constexpr size_t output_tile = 8;
    std::vector<Tile> tiles;
    for (size_t job_index = 0; job_index < jobs.size(); ++job_index) {
        const auto& matrix = std::get<CpuGgufMatrix>(jobs[job_index].weight->segments.front());
        for (size_t begin = 0; begin < matrix.rows; begin += output_tile) {
            tiles.push_back({job_index, begin, std::min(begin + output_tile,
                                                        static_cast<size_t>(matrix.rows))});
        }
    }
    pool_->parallel_for(0, tiles.size(), 1, [&](size_t begin, size_t end) {
        for (size_t task = begin; task < end; ++task) {
            const Tile& tile = tiles[task];
            const CpuGroupedGemmJob& job = jobs[tile.job];
            const auto& matrix = std::get<CpuGgufMatrix>(job.weight->segments.front());
            for (size_t out = tile.begin; out < tile.end; ++out) {
                size_t row = 0;
                if (gguf_dot4_) {
                    for (; row + 4 <= job.rows; row += 4) {
                        float values[4];
                        gguf_dot4_(matrix.data + out * matrix.row_bytes(), matrix.type,
                                   activation.data() + (job.row_offset + row) * blocks_per_row,
                                   matrix.cols, values);
                        for (size_t lane = 0; lane < 4; ++lane) {
                            output[(job.row_offset + row + lane) * output_width + out] = values[lane];
                        }
                    }
                }
                for (; row < job.rows; ++row) {
                    output[(job.row_offset + row) * output_width + out] = gguf_dot_(
                        matrix.data + out * matrix.row_bytes(), matrix.type,
                        activation.data() + (job.row_offset + row) * blocks_per_row,
                        matrix.cols);
                }
            }
        }
    });
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
        } else {
            cpu_gguf_dequantize_row(
                std::get<CpuGgufMatrix>(segment), row, output);
        }
        return;
    }
    throw std::logic_error("CPU embedding row was not found");
}

void CpuLinearEngine::gemv_raw(const float* weight, const float* input,
                               float* output, int n, int k) const {
    // Parallelize over output rows. Each row is a dot product of k elements.
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
    // The router is small in its output dimension but is evaluated for every
    // prefill row. Parallelizing rows avoids nested thread-pool calls and
    // keeps each worker's input/output contiguous.
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

} // namespace celeg

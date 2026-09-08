#include "celeg/backend/cpu/linear.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace celeg {
namespace {

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

}

void CpuLinearEngine::gemv_gguf(const GgmlMatrixView& matrix,
                                std::span<const CpuQ8KBlock> activation,
                                float* output, float beta) const {
    const size_t grain = std::max<size_t>(
        1, matrix.rows / std::max<size_t>(1, pool_->size() * 8));
    pool_->parallel_for(0, matrix.rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const float value = gguf_dot_(
                matrix.data + row * matrix.row_bytes(), matrix.type,
                activation.data(), matrix.cols);
            float& destination = output[row];
            destination = beta == 0.0f ? value : value + beta * destination;
        }
    });
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
    if ((!output && rows != 0) || (rows > 0 &&
        activation.size() < rows * (weight.cols / 256))) {
        throw std::invalid_argument("invalid prequantized GGUF activation");
    }
    const size_t blocks_per_row = weight.cols / 256;

    size_t output_offset = 0;
    for (const CpuLinearMatrix& segment : weight.segments) {
        const GgmlMatrixView& matrix = std::get<GgmlMatrixView>(segment);
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

    const CpuLinearWeight* first = jobs.front().weight;
    if (!first) throw std::invalid_argument("grouped CPU GEMM has null weight");
    first->validate();
    const bool native = first->segments.size() == 1 &&
        std::holds_alternative<GgmlMatrixView>(first->segments.front());
    const size_t input_width = first->cols;
    const size_t output_width = first->rows;
    size_t total_rows = 0;
    for (const CpuGroupedGemmJob& job : jobs) {
        if (!job.weight) throw std::invalid_argument("grouped CPU GEMM has null job weight");
        job.weight->validate();
        if (job.weight->cols != input_width || job.weight->rows != output_width ||
            job.weight->segments.size() != 1 ||
            !std::holds_alternative<GgmlMatrixView>(job.weight->segments.front())) {
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
        const auto& matrix = std::get<GgmlMatrixView>(jobs[job_index].weight->segments.front());
        for (size_t begin = 0; begin < matrix.rows; begin += output_tile) {
            tiles.push_back({job_index, begin, std::min(begin + output_tile,
                                                        static_cast<size_t>(matrix.rows))});
        }
    }
    pool_->parallel_for(0, tiles.size(), 1, [&](size_t begin, size_t end) {
        for (size_t task = begin; task < end; ++task) {
            const Tile& tile = tiles[task];
            const CpuGroupedGemmJob& job = jobs[tile.job];
            const auto& matrix = std::get<GgmlMatrixView>(job.weight->segments.front());
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

}

#include "weight_loader_internal.hpp"

#include <algorithm>

namespace celeg::cuda_loader_detail {
namespace {

constexpr int kRopeHeadDim = 128;

int rope_source_row(int output) {
    return (output % (kRopeHeadDim / 2)) * 2 + output / (kRopeHeadDim / 2);
}

} // namespace

void undo_rope_permutation(std::vector<__nv_bfloat16>& values, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || rows % kRopeHeadDim != 0 ||
        values.size() != static_cast<size_t>(rows) * cols) {
        throw std::runtime_error("invalid GGUF Q/K shape for RoPE permutation");
    }
    std::vector<__nv_bfloat16> reordered(values.size());
    const int heads = rows / kRopeHeadDim;
    for (int head = 0; head < heads; ++head) {
        for (int output = 0; output < kRopeHeadDim; ++output) {
            const int source = rope_source_row(output);
            std::copy_n(values.data() +
                            static_cast<size_t>(head * kRopeHeadDim + source) * cols,
                        cols,
                        reordered.data() +
                            static_cast<size_t>(head * kRopeHeadDim + output) * cols);
        }
    }
    values.swap(reordered);
}

void undo_rope_permutation_raw(std::vector<uint8_t>& blocks, int rows, size_t row_bytes) {
    if (rows <= 0 || rows % kRopeHeadDim != 0 ||
        blocks.size() != static_cast<size_t>(rows) * row_bytes) {
        throw std::runtime_error("invalid GGUF Q/K block shape for RoPE permutation");
    }
    std::vector<uint8_t> reordered(blocks.size());
    const int heads = rows / kRopeHeadDim;
    for (int head = 0; head < heads; ++head) {
        for (int output = 0; output < kRopeHeadDim; ++output) {
            const int source = rope_source_row(output);
            std::copy_n(blocks.data() +
                            static_cast<size_t>(head * kRopeHeadDim + source) * row_bytes,
                        row_bytes,
                        reordered.data() +
                            static_cast<size_t>(head * kRopeHeadDim + output) * row_bytes);
        }
    }
    blocks.swap(reordered);
}

} // namespace celeg::cuda_loader_detail

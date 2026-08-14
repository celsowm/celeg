#pragma once

// Device-resident linear (2D) weight views.
//
// This is the foundation header of the CUDA model implementation types: a
// component that needs `LinearWeight` includes only this file and therefore
// depends on nothing but the GGUF block-format enum and the CUDA BF16 type.
// It deliberately knows nothing about experts, layers, session state or
// weight ownership.

#include "celeg/checkpoint/formats/gguf.hpp"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace celeg {

enum class LinearStorageKind : uint8_t {
    Bf16,
    Int8,
    Int4,
    // GGUF native block-quantized formats. Weights stay packed in their on-disk
    // super-block layout on the device; the matmul kernel dequantizes on the
    // fly. `gguf_segments` points at the raw block bytes.
    Q4_K,
    Q6_K,
};

struct GgufLinearSegment {
    const uint8_t* blocks = nullptr;
    GgmlType type = GgmlType::Unknown;
    int row_offset = 0;
    int rows = 0;
    int cols = 0;
    size_t row_bytes = 0;
};

struct LinearWeight {
    LinearStorageKind kind = LinearStorageKind::Bf16;
    const __nv_bfloat16* bf16 = nullptr;
    const int8_t* int8 = nullptr;
    const uint8_t* int4 = nullptr;
    const float* scales = nullptr;
    std::vector<GgufLinearSegment> gguf_segments;
    int rows = 0;
    int cols = 0;

    bool quantized() const { return kind != LinearStorageKind::Bf16; }
    bool int4_quantized() const { return kind == LinearStorageKind::Int4; }
    bool int8_quantized() const { return kind == LinearStorageKind::Int8; }
    bool gguf_quantized() const {
        return kind == LinearStorageKind::Q4_K || kind == LinearStorageKind::Q6_K;
    }
    void validate_storage() const;
};

// Returns a view into a contiguous row range of an existing linear weight.
// The returned LinearWeight shares storage with the source.
inline LinearWeight slice_rows(const LinearWeight& weight,
                               int row_offset, int rows) {
    if (row_offset < 0 || rows <= 0 || row_offset + rows > weight.rows) {
        throw std::out_of_range("linear weight row slice is out of range");
    }
    LinearWeight result = weight;
    result.rows = rows;
    if (weight.bf16) {
        result.bf16 = weight.bf16 + static_cast<size_t>(row_offset) * weight.cols;
    }
    if (weight.int8) {
        result.int8 = weight.int8 + static_cast<size_t>(row_offset) * weight.cols;
        result.scales = weight.scales + row_offset;
    }
    if (weight.int4) {
        const size_t packed_cols =
            (static_cast<size_t>(weight.cols) + 1) / 2;
        result.int4 = weight.int4 + static_cast<size_t>(row_offset) * packed_cols;
        result.scales = weight.scales + row_offset;
    }
    if (weight.gguf_quantized()) {
        result.gguf_segments.clear();
        const int end = row_offset + rows;
        for (const GgufLinearSegment& segment : weight.gguf_segments) {
            const int segment_end = segment.row_offset + segment.rows;
            const int first = std::max(row_offset, segment.row_offset);
            const int last = std::min(end, segment_end);
            if (first >= last) continue;
            GgufLinearSegment view = segment;
            view.blocks = segment.blocks +
                static_cast<size_t>(first - segment.row_offset) * segment.row_bytes;
            view.row_offset = first - row_offset;
            view.rows = last - first;
            result.gguf_segments.push_back(view);
        }
        if (result.gguf_segments.empty()) {
            throw std::runtime_error("GGUF row slice has no segments");
        }
        result.kind = result.gguf_segments.front().type == GgmlType::Q4_K
            ? LinearStorageKind::Q4_K : LinearStorageKind::Q6_K;
    }
    return result;
}

} // namespace celeg

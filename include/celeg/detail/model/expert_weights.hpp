#pragma once

// Packed expert (3D) weight collections.
//
// Depends only on `linear_weights.hpp`, because an expert collection is
// defined as a stack of `LinearWeight` views. Nothing here knows about
// routing, offload, caches or layers.

#include "celeg/detail/model/linear_weights.hpp"

#include <cstddef>
#include <stdexcept>

namespace celeg {

// Packed expert linear weight. For recurrent-MoE layouts, expert
// collections are stored as contiguous 3D tensors
//   gate_up_proj: [num_experts, 2 * moe_intermediate, hidden]
//   down_proj:    [num_experts, hidden, moe_intermediate]
// `expert_view()` exposes a zero-copy 2D LinearWeight into one expert's
// contiguous region.
struct ExpertLinearWeight {
    LinearStorageKind kind = LinearStorageKind::Bf16;
    const __nv_bfloat16* bf16 = nullptr;
    const int8_t* int8 = nullptr;
    const uint8_t* int4 = nullptr;
    const float* scales = nullptr;
    const uint8_t* gguf_blocks = nullptr;
    GgmlType gguf_type = GgmlType::Unknown;
    size_t gguf_row_bytes = 0;
    size_t gguf_expert_stride = 0;
    int experts = 0;
    int rows_per_expert = 0;
    int cols = 0;

    LinearWeight expert_view(int expert_id) const;
};

inline LinearWeight ExpertLinearWeight::expert_view(int expert_id) const {
    if (expert_id < 0 || expert_id >= experts) {
        throw std::out_of_range("expert id out of range");
    }
    LinearWeight view;
    view.kind = kind;
    view.rows = rows_per_expert;
    view.cols = cols;
    const size_t expert_offset =
        static_cast<size_t>(expert_id) * static_cast<size_t>(rows_per_expert) *
        static_cast<size_t>(cols);
    const size_t scale_offset =
        static_cast<size_t>(expert_id) * static_cast<size_t>(rows_per_expert);
    if (kind == LinearStorageKind::Bf16) {
        view.bf16 = bf16 + expert_offset;
    } else if (kind == LinearStorageKind::Int8) {
        view.int8 = int8 + expert_offset;
        view.scales = scales + scale_offset;
    } else if (kind == LinearStorageKind::Int4) {
        const size_t packed_cols = (static_cast<size_t>(cols) + 1) / 2;
        view.int4 = int4 + expert_offset / static_cast<size_t>(cols) * packed_cols;
        view.scales = scales + scale_offset;
    } else if (kind == LinearStorageKind::Q4_K || kind == LinearStorageKind::Q6_K) {
        if (!gguf_blocks || gguf_row_bytes == 0 || gguf_expert_stride == 0) {
            throw std::logic_error("invalid GGUF expert storage");
        }
        GgufLinearSegment segment;
        segment.blocks = gguf_blocks +
            static_cast<size_t>(expert_id) * gguf_expert_stride;
        segment.type = gguf_type;
        segment.rows = rows_per_expert;
        segment.cols = cols;
        segment.row_bytes = gguf_row_bytes;
        view.gguf_segments.push_back(segment);
    }
    return view;
}

} // namespace celeg

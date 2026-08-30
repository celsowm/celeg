#pragma once


#include "celeg/backend/cuda/model/detail/linear_weights.hpp"

#include <cstddef>
#include <stdexcept>

namespace celeg {

/// Tag for the resident-expert-table storage carried by ExpertLinearWeight.
/// This is intentionally a separate type from LinearStorage: ExpertLinearWeight
/// is a flat table of `experts` identically-shaped rows-per-expert blocks
/// (a residency/layout concern), not itself a std::variant of payloads. Making
/// the per-expert *view* (expert_view()) impossible-state-free is what matters
/// here, and that view is expressed as a real LinearWeight/LinearStorage
/// below. Converting ExpertLinearWeight's own resident/offloaded table
/// storage into a variant is tracked separately (plan item 1.3).
enum class ExpertStorageKind : uint8_t {
    Bf16,
    Int8,
    Int4,
    Q4_K,
    Q6_K,
};

struct ExpertLinearWeight {
    ExpertStorageKind kind = ExpertStorageKind::Bf16;
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
    view.rows = rows_per_expert;
    view.cols = cols;
    const size_t expert_offset =
        static_cast<size_t>(expert_id) * static_cast<size_t>(rows_per_expert) *
        static_cast<size_t>(cols);
    const size_t scale_offset =
        static_cast<size_t>(expert_id) * static_cast<size_t>(rows_per_expert);
    switch (kind) {
        case ExpertStorageKind::Bf16:
            view.storage = Bf16LinearStorage{bf16 + expert_offset};
            break;
        case ExpertStorageKind::Int8:
            view.storage = Int8LinearStorage{int8 + expert_offset, scales + scale_offset};
            break;
        case ExpertStorageKind::Int4: {
            const size_t packed_cols = (static_cast<size_t>(cols) + 1) / 2;
            view.storage = Int4LinearStorage{
                int4 + expert_offset / static_cast<size_t>(cols) * packed_cols,
                scales + scale_offset};
            break;
        }
        case ExpertStorageKind::Q4_K:
        case ExpertStorageKind::Q6_K: {
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
            GgufLinearStorage storage;
            storage.segments.push_back(segment);
            view.storage = std::move(storage);
            break;
        }
    }
    return view;
}

}


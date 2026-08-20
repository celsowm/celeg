#pragma once


#include "celeg/backend/cuda/execution_plan.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace celeg {

struct GgufLinearSegment {
    const uint8_t* blocks = nullptr;
    GgmlType type = GgmlType::Unknown;
    int row_offset = 0;
    int rows = 0;
    int cols = 0;
    size_t row_bytes = 0;
};

struct Bf16LinearStorage {
    const __nv_bfloat16* data = nullptr;
};

struct Int8LinearStorage {
    const int8_t* data = nullptr;
    const float* scales = nullptr;
    // Optional dense BF16 shadow of the same weight, used by GEMM dispatch to
    // prefer a cuBLAS(Lt) BF16 kernel over the custom quantized kernel for
    // multi-row (prefill) inputs, while decode (single row) still uses the
    // quantized path. Populated only when the loader derived this storage
    // from an existing BF16 buffer.
    const __nv_bfloat16* bf16_fallback = nullptr;
};

struct Int4LinearStorage {
    const uint8_t* data = nullptr;
    const float* scales = nullptr;
    // See Int8LinearStorage::bf16_fallback.
    const __nv_bfloat16* bf16_fallback = nullptr;
};

struct GgufLinearStorage {
    std::vector<GgufLinearSegment> segments;
};

// FP8 E4M3 weight, per-channel static scale (one scale per output row) --
// see docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 3. Not yet populated by
// any loader; the storage variant and the matching LinearKernelKind::Fp8W8A8
// dispatch case exist ahead of the loader-side wiring so the kernel/dispatch
// path can be built and tested independently, the same staged approach
// Phase 1 used for the per-tensor kernel-override plumbing.
struct Fp8LinearStorage {
    const __nv_fp8_e4m3* data = nullptr;
    const float* scales = nullptr;
};

using LinearStorage = std::variant<
    Bf16LinearStorage,
    Int8LinearStorage,
    Int4LinearStorage,
    GgufLinearStorage,
    Fp8LinearStorage>;

struct LinearWeight {
    int rows = 0;
    int cols = 0;
    LinearStorage storage;
    // Per-tensor GEMM kernel override, set by the loader when a tensor's
    // storage format demands a specific kernel (e.g. a mixed-quant
    // checkpoint that isn't uniformly one format). Unset (nullopt) means
    // "use the execution plan's model-wide kernel", which is every
    // existing single-format model's behavior today.
    std::optional<LinearKernelKind> kernel;
};

inline LinearWeight slice_rows(const LinearWeight& weight,
                               int row_offset, int rows) {
    if (row_offset < 0 || rows <= 0 || row_offset + rows > weight.rows) {
        throw std::out_of_range("linear weight row slice is out of range");
    }
    LinearWeight result = weight;
    result.rows = rows;
    result.storage = std::visit(
        [&](const auto& storage) -> LinearStorage {
            using StorageT = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageT, Bf16LinearStorage>) {
                Bf16LinearStorage out = storage;
                if (out.data) {
                    out.data = storage.data + static_cast<size_t>(row_offset) * weight.cols;
                }
                return out;
            } else if constexpr (std::is_same_v<StorageT, Int8LinearStorage>) {
                Int8LinearStorage out = storage;
                if (out.data) {
                    out.data = storage.data + static_cast<size_t>(row_offset) * weight.cols;
                    out.scales = storage.scales + row_offset;
                }
                if (out.bf16_fallback) {
                    out.bf16_fallback = storage.bf16_fallback +
                        static_cast<size_t>(row_offset) * weight.cols;
                }
                return out;
            } else if constexpr (std::is_same_v<StorageT, Int4LinearStorage>) {
                Int4LinearStorage out = storage;
                if (out.data) {
                    const size_t packed_cols =
                        (static_cast<size_t>(weight.cols) + 1) / 2;
                    out.data = storage.data + static_cast<size_t>(row_offset) * packed_cols;
                    out.scales = storage.scales + row_offset;
                }
                if (out.bf16_fallback) {
                    out.bf16_fallback = storage.bf16_fallback +
                        static_cast<size_t>(row_offset) * weight.cols;
                }
                return out;
            } else if constexpr (std::is_same_v<StorageT, Fp8LinearStorage>) {
                Fp8LinearStorage out = storage;
                if (out.data) {
                    out.data = storage.data + static_cast<size_t>(row_offset) * weight.cols;
                    out.scales = storage.scales + row_offset;
                }
                return out;
            } else {
                static_assert(std::is_same_v<StorageT, GgufLinearStorage>);
                GgufLinearStorage out;
                const int end = row_offset + rows;
                for (const GgufLinearSegment& segment : storage.segments) {
                    const int segment_end = segment.row_offset + segment.rows;
                    const int first = std::max(row_offset, segment.row_offset);
                    const int last = std::min(end, segment_end);
                    if (first >= last) continue;
                    GgufLinearSegment view = segment;
                    view.blocks = segment.blocks +
                        static_cast<size_t>(first - segment.row_offset) * segment.row_bytes;
                    view.row_offset = first - row_offset;
                    view.rows = last - first;
                    out.segments.push_back(view);
                }
                if (out.segments.empty()) {
                    throw std::runtime_error("GGUF row slice has no segments");
                }
                return out;
            }
        },
        weight.storage);
    return result;
}

}

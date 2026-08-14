#pragma once

#include "celeg/backend/cuda/runtime_types.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

// CUDA attention capability policy.
//
// Which attention kernel family handles a given request is a capability
// decision, not an architecture decision.  It depends only on:
//
//   * how the KV cache stores vectors        (KvCacheMode)
//   * how the KV cache is addressed          (AttentionKvLayout)
//   * where the sequence position lives      (AttentionPositionSource)
//   * whether a position bias is applied     (AttentionPositionBias)
//   * whether the request is bulk or single  (AttentionOperation)
//   * tuning inputs (fast attention, segmented attention, head_dim, rows)
//
// The support matrix below is data.  It states, for every combination the CUDA
// backend can reach, whether a kernel plan exists and — when it does not — why.
// `resolve_attention_capability` maps a request onto a plan and is required to
// produce a plan the matrix marks supported; a request whose plan is not
// supported is rejected by name instead of falling through to a kernel that was
// written for a different combination.
//
// This header is intentionally free of CUDA types so the policy can be tested
// on a host-only build.

namespace celeg {

// Bulk (prompt) attention versus single-token attention.
enum class AttentionOperation {
    Prefill,
    Decode,
};

// How the KV cache for this launch is addressed.
enum class AttentionKvLayout {
    // One contiguous per-layer cache owned by the model/session.
    Contiguous,
    // Physical page pool addressed through a per-request page table.
    Paged,
    // Per-request contiguous caches gathered through a device pointer array
    // (packed execution over independent sessions).
    BatchPointers,
};

// Where the kernel reads the sequence position from.  Host-scalar launches bake
// the length into the launch; device-counter launches read a device-resident
// position and are therefore CUDA-graph capturable.
enum class AttentionPositionSource {
    HostScalar,
    DeviceCounter,
};

enum class AttentionPositionBias {
    None,
    Alibi,
};

// Kernel family selected for a request.  Each alternative maps to a separately
// benchmarked kernel specialization; the domain is intentionally closed.
// Adding an alternative requires updating the capability matrix below.
enum class AttentionAlgorithm {
    // Two-pass exact softmax over the full key range.
    Strict,
    // Single-pass online (streaming) softmax.
    Online,
    // Chunked online softmax with a partial-state reduction.
    Segmented,
    // Tiled flash-style kernel.
    Flash,
    // Batched GEMM scores + softmax + batched GEMM values.
    Gemm,
    // Fused ALiBi kernel family (bias applied inside the score loop).
    Alibi,
};

enum class AttentionUnsupportedReason {
    None,
    // No kernel exists for this combination.
    NoKernelForCombination,
    // A kernel exists but this dispatch path never selects it.
    KernelNotSelectedByPolicy,
    // The kernel family does not implement this KV addressing mode.
    LayoutNotImplemented,
    // The kernel family does not implement this position source.
    PositionSourceNotImplemented,
    // The kernel exists but rejects this head dimension.
    HeadDimensionUnsupported,
};

// One row of the capability matrix.
struct AttentionCapability {
    KvCacheMode kv_format = KvCacheMode::Bf16;
    AttentionPositionBias bias = AttentionPositionBias::None;
    AttentionOperation operation = AttentionOperation::Prefill;
    AttentionKvLayout layout = AttentionKvLayout::Contiguous;
    AttentionPositionSource position_source = AttentionPositionSource::HostScalar;
    AttentionAlgorithm algorithm = AttentionAlgorithm::Strict;
    bool supported = false;
    AttentionUnsupportedReason reason = AttentionUnsupportedReason::NoKernelForCombination;
};

// Everything a launch site knows before it picks a kernel.
struct AttentionRequest {
    KvCacheMode kv_format = KvCacheMode::Bf16;
    AttentionOperation operation = AttentionOperation::Decode;
    AttentionKvLayout layout = AttentionKvLayout::Contiguous;
    AttentionPositionSource position_source = AttentionPositionSource::DeviceCounter;
    AttentionPositionBias bias = AttentionPositionBias::None;
    // CudaModelOptions::fast_attention.
    bool fast_attention = false;
    // Session-resolved long-context chunking decision.
    bool segmented_attention = false;
    // Resolved CELEG_FLASH_ATTN request (prefill only).
    bool flash_attention_requested = false;
    int head_dim = 0;
    int rows = 1;
};

// Prefill tuning constants shared with the workspace allocator.
//
// The tiled prefill kernel is only valid up to this head dimension.
inline constexpr int kAttentionFlashMaxHeadDim = 128;
// Above this head dimension the batched-GEMM prefill path is not used even when
// the flash flag is absent: its strided GQA batches can corrupt the KV state for
// layouts it was not tuned for.
inline constexpr int kAttentionFlashPreferredHeadDim = 64;
// Row-count ceiling for the batched-GEMM prefill path's dense
// O(query_heads*rows^2) score/probability scratch.  Above it, prefill falls back
// to the segmented path, whose scratch is O(rows).
inline constexpr int kAttentionMaxGemmRows = 2048;
// Chunk size for segmented prefill: prefill parallelism comes from chunk
// *count*, not chunk size.
inline constexpr int kAttentionPrefillChunkTokens = 64;

namespace detail {

// The support matrix.  Rows are keyed by the addressing/format/position axes;
// combinations absent from the table are unsupported with
// `NoKernelForCombination`.
//
// Read this as the answer to "does a kernel exist for X, and if not, why not?"
// It is deliberately asymmetric: Int8 prefill has no flash/GEMM/segmented
// kernels, and the host-position contiguous decode path (used by embedding
// prefill and tokenwise chunk prefill) has neither ALiBi nor segmented kernels.
inline constexpr AttentionCapability kAttentionCapabilities[] = {
    // ---- Prefill, contiguous, host-scalar position -----------------------
    {KvCacheMode::Bf16, AttentionPositionBias::Alibi, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Alibi, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Flash, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Gemm, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Segmented, true, AttentionUnsupportedReason::None},
    // A chunk-local online prefill kernel exists, but bulk prefill reaches it
    // only through the segmented reduction, never on its own.
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Online, false,
     AttentionUnsupportedReason::KernelNotSelectedByPolicy},

    {KvCacheMode::Int8, AttentionPositionBias::Alibi, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Alibi, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    // Int8 prefill has no tiled, batched-GEMM or segmented kernel: dequantizing
    // scales per tile was never implemented for those families.
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Flash, false, AttentionUnsupportedReason::NoKernelForCombination},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Gemm, false, AttentionUnsupportedReason::NoKernelForCombination},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Segmented, false, AttentionUnsupportedReason::NoKernelForCombination},

    // ---- Decode, contiguous, device-counter position ---------------------
    {KvCacheMode::Bf16, AttentionPositionBias::Alibi, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Alibi, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Segmented, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},

    {KvCacheMode::Int8, AttentionPositionBias::Alibi, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Alibi, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Segmented, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},

    // ---- Decode, contiguous, host-scalar position ------------------------
    // Used by prompt-embedding prefill and tokenwise chunk prefill.  Only the
    // plain online/strict kernels take a host length; ALiBi and segmented were
    // never given host-scalar overloads.
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::Alibi, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Alibi, false,
     AttentionUnsupportedReason::PositionSourceNotImplemented},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Segmented, false,
     AttentionUnsupportedReason::PositionSourceNotImplemented},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::Alibi, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Alibi, false,
     AttentionUnsupportedReason::PositionSourceNotImplemented},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Segmented, false,
     AttentionUnsupportedReason::PositionSourceNotImplemented},

    // ---- Decode, paged, device-counter position --------------------------
    // Online and strict share one paged launcher that takes the fast-attention
    // decision as a kernel argument.
    {KvCacheMode::Bf16, AttentionPositionBias::Alibi, AttentionOperation::Decode,
     AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Alibi, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Segmented, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},

    {KvCacheMode::Int8, AttentionPositionBias::Alibi, AttentionOperation::Decode,
     AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Alibi, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Segmented, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},

    // ---- Decode, batch-pointer layout, device-counter position -----------
    // Packed execution over independent per-request contiguous caches.  There
    // is no segmented batch-pointer kernel; long contexts in packed mode are
    // served through the paged layout.
    {KvCacheMode::Bf16, AttentionPositionBias::Alibi, AttentionOperation::Decode,
     AttentionKvLayout::BatchPointers, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Alibi, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::BatchPointers, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::BatchPointers, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::BatchPointers, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Segmented, false,
     AttentionUnsupportedReason::NoKernelForCombination},
    {KvCacheMode::Int8, AttentionPositionBias::Alibi, AttentionOperation::Decode,
     AttentionKvLayout::BatchPointers, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Alibi, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::BatchPointers, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Online, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::BatchPointers, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Strict, true, AttentionUnsupportedReason::None},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Decode,
     AttentionKvLayout::BatchPointers, AttentionPositionSource::DeviceCounter,
     AttentionAlgorithm::Segmented, false,
     AttentionUnsupportedReason::NoKernelForCombination},
};

} // namespace detail

// Query the capability matrix for one (combination, algorithm) pair.
constexpr AttentionCapability attention_capability(
    KvCacheMode kv_format, AttentionPositionBias bias, AttentionOperation operation,
    AttentionKvLayout layout, AttentionPositionSource position_source,
    AttentionAlgorithm algorithm) {
    for (const AttentionCapability& entry : detail::kAttentionCapabilities) {
        if (entry.kv_format == kv_format && entry.bias == bias &&
            entry.operation == operation && entry.layout == layout &&
            entry.position_source == position_source &&
            entry.algorithm == algorithm) {
            return entry;
        }
    }
    AttentionCapability missing;
    missing.kv_format = kv_format;
    missing.bias = bias;
    missing.operation = operation;
    missing.layout = layout;
    missing.position_source = position_source;
    missing.algorithm = algorithm;
    missing.supported = false;
    missing.reason = operation == AttentionOperation::Prefill &&
                             layout != AttentionKvLayout::Contiguous
                         ? AttentionUnsupportedReason::LayoutNotImplemented
                         : AttentionUnsupportedReason::NoKernelForCombination;
    return missing;
}

// Pick the kernel family for a request, then answer with the matrix row for
// that choice.  A caller must not launch when `supported` is false.
//
// This is a pure function of the request: it performs no allocation and is
// evaluated once per attention launch, before entering the kernel.
constexpr AttentionCapability resolve_attention_capability(const AttentionRequest& request) {
    const auto answer = [&](AttentionAlgorithm algorithm) {
        return attention_capability(request.kv_format, request.bias, request.operation,
                                    request.layout, request.position_source, algorithm);
    };
    if (request.bias == AttentionPositionBias::Alibi) {
        return answer(AttentionAlgorithm::Alibi);
    }
    if (request.operation == AttentionOperation::Prefill) {
        if (request.layout != AttentionKvLayout::Contiguous ||
            request.position_source != AttentionPositionSource::HostScalar) {
            return answer(AttentionAlgorithm::Strict);
        }
        if (!request.fast_attention) return answer(AttentionAlgorithm::Strict);
        if (request.kv_format == KvCacheMode::Int8) return answer(AttentionAlgorithm::Online);
        const bool flash_supported = request.head_dim <= kAttentionFlashMaxHeadDim;
        // The environment flag only widens the flash range downwards: above the
        // preferred head dimension the tiled path is used regardless, because
        // the batched-GEMM path's strided GQA batches can corrupt the KV state
        // for wide heads.
        if (flash_supported && (request.flash_attention_requested ||
                                request.head_dim > kAttentionFlashPreferredHeadDim)) {
            return answer(AttentionAlgorithm::Flash);
        }
        if (request.rows <= kAttentionMaxGemmRows) {
            return answer(AttentionAlgorithm::Gemm);
        }
        return answer(AttentionAlgorithm::Segmented);
    }
    if (request.segmented_attention) return answer(AttentionAlgorithm::Segmented);
    if (request.fast_attention) return answer(AttentionAlgorithm::Online);
    return answer(AttentionAlgorithm::Strict);
}

constexpr const char* attention_operation_name(AttentionOperation operation) {
    switch (operation) {
    case AttentionOperation::Prefill: return "prefill";
    case AttentionOperation::Decode: return "decode";
    }
    return "unknown";
}

constexpr const char* attention_kv_format_name(KvCacheMode mode) {
    switch (mode) {
    case KvCacheMode::Bf16: return "bf16";
    case KvCacheMode::Int8: return "int8";
    }
    return "unknown";
}

constexpr const char* attention_kv_layout_name(AttentionKvLayout layout) {
    switch (layout) {
    case AttentionKvLayout::Contiguous: return "contiguous";
    case AttentionKvLayout::Paged: return "paged";
    case AttentionKvLayout::BatchPointers: return "batch-pointers";
    }
    return "unknown";
}

constexpr const char* attention_position_source_name(AttentionPositionSource source) {
    switch (source) {
    case AttentionPositionSource::HostScalar: return "host-position";
    case AttentionPositionSource::DeviceCounter: return "device-position";
    }
    return "unknown";
}

constexpr const char* attention_position_bias_name(AttentionPositionBias bias) {
    switch (bias) {
    case AttentionPositionBias::None: return "no-bias";
    case AttentionPositionBias::Alibi: return "alibi";
    }
    return "unknown";
}

constexpr const char* attention_algorithm_name(AttentionAlgorithm algorithm) {
    switch (algorithm) {
    case AttentionAlgorithm::Strict: return "strict";
    case AttentionAlgorithm::Online: return "online";
    case AttentionAlgorithm::Segmented: return "segmented";
    case AttentionAlgorithm::Flash: return "flash";
    case AttentionAlgorithm::Gemm: return "gemm";
    case AttentionAlgorithm::Alibi: return "alibi";
    }
    return "unknown";
}

constexpr const char* attention_unsupported_reason_name(AttentionUnsupportedReason reason) {
    switch (reason) {
    case AttentionUnsupportedReason::None: return "supported";
    case AttentionUnsupportedReason::NoKernelForCombination:
        return "no CUDA kernel implements this combination";
    case AttentionUnsupportedReason::KernelNotSelectedByPolicy:
        return "a kernel exists but this dispatch path never selects it";
    case AttentionUnsupportedReason::LayoutNotImplemented:
        return "this kernel family does not implement this KV addressing mode";
    case AttentionUnsupportedReason::PositionSourceNotImplemented:
        return "this kernel family does not implement this position source";
    case AttentionUnsupportedReason::HeadDimensionUnsupported:
        return "this kernel rejects this head dimension";
    }
    return "unknown";
}

// Name the combination well enough that the message identifies the exact policy
// row that has to change.
inline std::string describe_attention_capability(const AttentionCapability& capability) {
    std::string text = "CUDA attention combination ";
    text += attention_kv_format_name(capability.kv_format);
    text += '/';
    text += attention_operation_name(capability.operation);
    text += '/';
    text += attention_kv_layout_name(capability.layout);
    text += '/';
    text += attention_position_source_name(capability.position_source);
    text += '/';
    text += attention_position_bias_name(capability.bias);
    text += " -> ";
    text += attention_algorithm_name(capability.algorithm);
    return text;
}

class UnsupportedAttentionCapability : public std::invalid_argument {
public:
    explicit UnsupportedAttentionCapability(const AttentionCapability& capability)
        : std::invalid_argument(describe_attention_capability(capability) + " is not supported: " +
                                attention_unsupported_reason_name(capability.reason)),
          capability_(capability) {}

    const AttentionCapability& capability() const noexcept { return capability_; }
    AttentionUnsupportedReason reason() const noexcept { return capability_.reason; }
    AttentionAlgorithm algorithm() const noexcept { return capability_.algorithm; }

private:
    AttentionCapability capability_;
};

// Resolution point for every CUDA attention launch site: either a supported
// plan, or a typed rejection naming the combination and the reason.
inline AttentionCapability require_attention_capability(const AttentionRequest& request) {
    const AttentionCapability capability = resolve_attention_capability(request);
    if (!capability.supported) throw UnsupportedAttentionCapability(capability);
    return capability;
}

} // namespace celeg

#pragma once

#include "celeg/backend/cuda/runtime_types.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>


namespace celeg {

enum class AttentionOperation {
    Prefill,
    Decode,
};

enum class AttentionKvLayout {
    Contiguous,
    Paged,
    BatchPointers,
};

enum class AttentionPositionSource {
    HostScalar,
    DeviceCounter,
};

enum class AttentionPositionBias {
    None,
    Alibi,
};

enum class AttentionAlgorithm {
    Strict,
    Online,
    Segmented,
    Flash,
    Gemm,
    Alibi,
};

enum class AttentionUnsupportedReason {
    None,
    NoKernelForCombination,
    KernelNotSelectedByPolicy,
    LayoutNotImplemented,
    PositionSourceNotImplemented,
    HeadDimensionUnsupported,
};

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

struct AttentionRequest {
    KvCacheMode kv_format = KvCacheMode::Bf16;
    AttentionOperation operation = AttentionOperation::Decode;
    AttentionKvLayout layout = AttentionKvLayout::Contiguous;
    AttentionPositionSource position_source = AttentionPositionSource::DeviceCounter;
    AttentionPositionBias bias = AttentionPositionBias::None;
    bool fast_attention = false;
    bool segmented_attention = false;
    bool flash_attention_requested = false;
    int head_dim = 0;
    int rows = 1;
};

inline constexpr int kAttentionFlashMaxHeadDim = 128;
inline constexpr int kAttentionFlashPreferredHeadDim = 64;
inline constexpr int kAttentionMaxGemmRows = 2048;
inline constexpr int kAttentionPrefillChunkTokens = 64;

namespace detail {

inline constexpr AttentionCapability kAttentionCapabilities[] = {
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
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Flash, false, AttentionUnsupportedReason::NoKernelForCombination},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Gemm, false, AttentionUnsupportedReason::NoKernelForCombination},
    {KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
     AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
     AttentionAlgorithm::Segmented, false, AttentionUnsupportedReason::NoKernelForCombination},

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

}

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

inline AttentionCapability require_attention_capability(const AttentionRequest& request) {
    const AttentionCapability capability = resolve_attention_capability(request);
    if (!capability.supported) throw UnsupportedAttentionCapability(capability);
    return capability;
}

}

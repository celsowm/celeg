
#include "backend/cuda/attention_capability.hpp"
#include "backend/cuda/attention_norm.hpp"
#include "support/assertions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace celeg {
namespace {

constexpr std::array<KvCacheMode, 2> kFormats{KvCacheMode::Bf16, KvCacheMode::Int8};
constexpr std::array<AttentionOperation, 2> kOperations{
    AttentionOperation::Prefill, AttentionOperation::Decode};
constexpr std::array<AttentionKvLayout, 3> kLayouts{
    AttentionKvLayout::Contiguous, AttentionKvLayout::Paged,
    AttentionKvLayout::BatchPointers};
constexpr std::array<AttentionPositionSource, 2> kSources{
    AttentionPositionSource::HostScalar, AttentionPositionSource::DeviceCounter};
constexpr std::array<AttentionPositionBias, 3> kBiases{
    AttentionPositionBias::None, AttentionPositionBias::Alibi,
    AttentionPositionBias::Relative};
constexpr std::array<AttentionAlgorithm, 7> kAlgorithms{
    AttentionAlgorithm::Strict, AttentionAlgorithm::Online,
    AttentionAlgorithm::Segmented, AttentionAlgorithm::Flash,
    AttentionAlgorithm::Gemm, AttentionAlgorithm::Alibi,
    AttentionAlgorithm::RelativeBias};

AttentionRequest prefill_request(KvCacheMode format, bool fast, int head_dim, int rows) {
    AttentionRequest request;
    request.kv_format = format;
    request.operation = AttentionOperation::Prefill;
    request.layout = AttentionKvLayout::Contiguous;
    request.position_source = AttentionPositionSource::HostScalar;
    request.fast_attention = fast;
    request.head_dim = head_dim;
    request.rows = rows;
    return request;
}

AttentionRequest decode_request(KvCacheMode format, AttentionKvLayout layout,
                                AttentionPositionSource source) {
    AttentionRequest request;
    request.kv_format = format;
    request.operation = AttentionOperation::Decode;
    request.layout = layout;
    request.position_source = source;
    request.head_dim = 128;
    return request;
}

bool rejects(const AttentionRequest& request, AttentionUnsupportedReason reason,
             AttentionAlgorithm algorithm) {
    try {
        (void)require_attention_capability(request);
    } catch (const UnsupportedAttentionCapability& error) {
        const std::string message = error.what();
        return error.reason() == reason && error.algorithm() == algorithm &&
               message.find(attention_algorithm_name(algorithm)) != std::string::npos &&
               message.find(attention_kv_format_name(request.kv_format)) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}


void matrix_rows_are_unique() {
    for (const AttentionCapability& a : detail::kAttentionCapabilities) {
        int matches = 0;
        for (const AttentionCapability& b : detail::kAttentionCapabilities) {
            if (a.kv_format == b.kv_format && a.bias == b.bias &&
                a.operation == b.operation && a.layout == b.layout &&
                a.position_source == b.position_source && a.algorithm == b.algorithm) {
                ++matches;
            }
        }
        CELEG_TEST_CHECK(matches == 1);
        CELEG_TEST_CHECK(a.supported == (a.reason == AttentionUnsupportedReason::None));
    }
}

void resolution_never_contradicts_the_matrix() {
    const std::array<bool, 2> flags{false, true};
    const std::array<int, 4> head_dims{32, 64, 128, 192};
    const std::array<int, 3> row_counts{1, 512, 8192};
    size_t supported_count = 0;
    size_t rejected_count = 0;
    for (KvCacheMode format : kFormats) {
        for (AttentionOperation operation : kOperations) {
            for (AttentionKvLayout layout : kLayouts) {
                for (AttentionPositionSource source : kSources) {
                    for (AttentionPositionBias bias : kBiases) {
                        for (bool fast : flags) {
                            for (bool segmented : flags) {
                                for (bool flash : flags) {
                                    for (int head_dim : head_dims) {
                                        for (int rows : row_counts) {
    AttentionRequest request;
    request.kv_format = format;
    request.operation = operation;
    request.layout = layout;
    request.position_source = source;
    request.bias = bias;
    request.fast_attention = fast;
    request.segmented_attention = segmented;
    request.flash_attention_requested = flash;
    request.head_dim = head_dim;
    request.rows = rows;
    const AttentionCapability plan = resolve_attention_capability(request);
    CELEG_TEST_CHECK(plan.kv_format == format);
    CELEG_TEST_CHECK(plan.operation == operation);
    CELEG_TEST_CHECK(plan.layout == layout);
    CELEG_TEST_CHECK(plan.position_source == source);
    CELEG_TEST_CHECK(plan.bias == bias);
    CELEG_TEST_CHECK((bias == AttentionPositionBias::Alibi) ==
                     (plan.algorithm == AttentionAlgorithm::Alibi));
    CELEG_TEST_CHECK((bias == AttentionPositionBias::Relative) ==
                     (plan.algorithm == AttentionAlgorithm::RelativeBias));
    const AttentionCapability row = attention_capability(
        format, bias, operation, layout, source, plan.algorithm);
    CELEG_TEST_CHECK(row.supported == plan.supported);
    CELEG_TEST_CHECK(row.reason == plan.reason);
    if (plan.supported) {
        ++supported_count;
        CELEG_TEST_CHECK(plan.reason == AttentionUnsupportedReason::None);
        CELEG_TEST_CHECK(require_attention_capability(request).algorithm == plan.algorithm);
    } else {
        ++rejected_count;
        CELEG_TEST_CHECK(plan.reason != AttentionUnsupportedReason::None);
        CELEG_TEST_CHECK(rejects(request, plan.reason, plan.algorithm));
    }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    CELEG_TEST_CHECK(supported_count > 0);
    CELEG_TEST_CHECK(rejected_count > 0);
}


void prefill_support_matrix_is_asymmetric() {
    const auto selectable = [](KvCacheMode format, AttentionAlgorithm algorithm) {
        AttentionPositionBias bias = AttentionPositionBias::None;
        if (algorithm == AttentionAlgorithm::Alibi) {
            bias = AttentionPositionBias::Alibi;
        } else if (algorithm == AttentionAlgorithm::RelativeBias) {
            bias = AttentionPositionBias::Relative;
        }
        return attention_capability(format, bias, AttentionOperation::Prefill,
                                    AttentionKvLayout::Contiguous,
                                    AttentionPositionSource::HostScalar,
                                    algorithm).supported;
    };
    CELEG_TEST_CHECK(selectable(KvCacheMode::Int8, AttentionAlgorithm::Alibi));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Int8, AttentionAlgorithm::RelativeBias));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Int8, AttentionAlgorithm::Online));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Int8, AttentionAlgorithm::Strict));
    CELEG_TEST_CHECK(!selectable(KvCacheMode::Int8, AttentionAlgorithm::Flash));
    CELEG_TEST_CHECK(!selectable(KvCacheMode::Int8, AttentionAlgorithm::Gemm));
    CELEG_TEST_CHECK(!selectable(KvCacheMode::Int8, AttentionAlgorithm::Segmented));

    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Alibi));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::RelativeBias));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Flash));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Gemm));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Segmented));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Strict));
    CELEG_TEST_CHECK(!selectable(KvCacheMode::Bf16, AttentionAlgorithm::Online));
    CELEG_TEST_CHECK(attention_capability(
        KvCacheMode::Bf16, AttentionPositionBias::None, AttentionOperation::Prefill,
        AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
        AttentionAlgorithm::Online).reason ==
        AttentionUnsupportedReason::KernelNotSelectedByPolicy);
    CELEG_TEST_CHECK(attention_capability(
        KvCacheMode::Int8, AttentionPositionBias::None, AttentionOperation::Prefill,
        AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar,
        AttentionAlgorithm::Flash).reason ==
        AttentionUnsupportedReason::NoKernelForCombination);
}


void strict_paths() {
    for (KvCacheMode format : kFormats) {
        CELEG_TEST_CHECK(require_attention_capability(
            prefill_request(format, false, 128, 4096)).algorithm ==
            AttentionAlgorithm::Strict);
        CELEG_TEST_CHECK(require_attention_capability(
            prefill_request(format, false, 64, 1)).algorithm ==
            AttentionAlgorithm::Strict);
        for (AttentionPositionSource source : kSources) {
            CELEG_TEST_CHECK(require_attention_capability(decode_request(
                format, AttentionKvLayout::Contiguous, source)).algorithm ==
                AttentionAlgorithm::Strict);
        }
        CELEG_TEST_CHECK(require_attention_capability(decode_request(
            format, AttentionKvLayout::Paged,
            AttentionPositionSource::DeviceCounter)).algorithm ==
            AttentionAlgorithm::Strict);
        CELEG_TEST_CHECK(require_attention_capability(decode_request(
            format, AttentionKvLayout::BatchPointers,
            AttentionPositionSource::DeviceCounter)).algorithm ==
            AttentionAlgorithm::Strict);
    }
}


void prefill_fast_path_selection() {
    AttentionRequest request = prefill_request(KvCacheMode::Bf16, true, 128, 16);
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Flash);
    request.head_dim = 65;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Flash);
    request.head_dim = 64;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Gemm);
    request.flash_attention_requested = true;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Flash);
    request.head_dim = 129;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Gemm);
    request.flash_attention_requested = false;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Gemm);

    for (int head_dim : {32, 64, 128, 256}) {
        for (int rows : {1, 4096, 1 << 20}) {
            AttentionRequest int8 = prefill_request(KvCacheMode::Int8, true, head_dim, rows);
            int8.flash_attention_requested = true;
            CELEG_TEST_CHECK(require_attention_capability(int8).algorithm ==
                             AttentionAlgorithm::Online);
        }
    }
}


void long_context_fallback_selection() {
    AttentionRequest request = prefill_request(KvCacheMode::Bf16, true, 64,
                                               kAttentionMaxGemmRows);
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Gemm);
    request.rows = kAttentionMaxGemmRows + 1;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Segmented);
    request.head_dim = 128;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Flash);

    for (KvCacheMode format : kFormats) {
        for (AttentionKvLayout layout : {AttentionKvLayout::Contiguous,
                                         AttentionKvLayout::Paged}) {
            AttentionRequest decode = decode_request(
                format, layout, AttentionPositionSource::DeviceCounter);
            decode.fast_attention = true;
            CELEG_TEST_CHECK(require_attention_capability(decode).algorithm ==
                             AttentionAlgorithm::Online);
            decode.segmented_attention = true;
            CELEG_TEST_CHECK(require_attention_capability(decode).algorithm ==
                             AttentionAlgorithm::Segmented);
        }
        AttentionRequest packed = decode_request(
            format, AttentionKvLayout::BatchPointers,
            AttentionPositionSource::DeviceCounter);
        packed.segmented_attention = true;
        packed.fast_attention = true;
        CELEG_TEST_CHECK(rejects(packed, AttentionUnsupportedReason::NoKernelForCombination,
                                 AttentionAlgorithm::Segmented));
    }
}


void alibi_combinations() {
    for (KvCacheMode format : kFormats) {
        AttentionRequest prefill = prefill_request(format, true, 128, 4096);
        prefill.bias = AttentionPositionBias::Alibi;
        prefill.flash_attention_requested = true;
        CELEG_TEST_CHECK(require_attention_capability(prefill).algorithm ==
                         AttentionAlgorithm::Alibi);
        for (AttentionKvLayout layout : kLayouts) {
            AttentionRequest decode = decode_request(
                format, layout, AttentionPositionSource::DeviceCounter);
            decode.bias = AttentionPositionBias::Alibi;
            decode.fast_attention = true;
            decode.segmented_attention = true;
            CELEG_TEST_CHECK(require_attention_capability(decode).algorithm ==
                             AttentionAlgorithm::Alibi);
        }
        AttentionRequest host = decode_request(
            format, AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar);
        host.bias = AttentionPositionBias::Alibi;
        CELEG_TEST_CHECK(rejects(host,
                                 AttentionUnsupportedReason::PositionSourceNotImplemented,
                                 AttentionAlgorithm::Alibi));
        host.fast_attention = true;
        CELEG_TEST_CHECK(rejects(host,
                                 AttentionUnsupportedReason::PositionSourceNotImplemented,
                                 AttentionAlgorithm::Alibi));
    }
}

void relative_bias_combinations() {
    for (KvCacheMode format : kFormats) {
        AttentionRequest prefill = prefill_request(format, true, 128, 4096);
        prefill.bias = AttentionPositionBias::Relative;
        prefill.flash_attention_requested = true;
        CELEG_TEST_CHECK(require_attention_capability(prefill).algorithm ==
                         AttentionAlgorithm::RelativeBias);
        for (AttentionKvLayout layout : kLayouts) {
            AttentionRequest decode = decode_request(
                format, layout, AttentionPositionSource::DeviceCounter);
            decode.bias = AttentionPositionBias::Relative;
            decode.fast_attention = true;
            decode.segmented_attention = true;
            CELEG_TEST_CHECK(require_attention_capability(decode).algorithm ==
                             AttentionAlgorithm::RelativeBias);
        }
        AttentionRequest host = decode_request(
            format, AttentionKvLayout::Contiguous, AttentionPositionSource::HostScalar);
        host.bias = AttentionPositionBias::Relative;
        CELEG_TEST_CHECK(rejects(host,
                                 AttentionUnsupportedReason::PositionSourceNotImplemented,
                                 AttentionAlgorithm::RelativeBias));
        host.fast_attention = true;
        CELEG_TEST_CHECK(rejects(host,
                                 AttentionUnsupportedReason::PositionSourceNotImplemented,
                                 AttentionAlgorithm::RelativeBias));
    }
}


void paged_and_contiguous_parity() {
    for (KvCacheMode format : kFormats) {
        for (AttentionAlgorithm algorithm : {AttentionAlgorithm::Strict,
                                             AttentionAlgorithm::Online,
                                             AttentionAlgorithm::Segmented}) {
            CELEG_TEST_CHECK(attention_capability(
                format, AttentionPositionBias::None, AttentionOperation::Decode,
                AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
                algorithm).supported);
            CELEG_TEST_CHECK(attention_capability(
                format, AttentionPositionBias::None, AttentionOperation::Decode,
                AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
                algorithm).supported);
        }
        for (AttentionKvLayout layout : {AttentionKvLayout::Paged,
                                         AttentionKvLayout::BatchPointers}) {
            for (AttentionAlgorithm algorithm : kAlgorithms) {
                CELEG_TEST_CHECK(!attention_capability(
                    format, AttentionPositionBias::None, AttentionOperation::Prefill,
                    layout, AttentionPositionSource::HostScalar, algorithm).supported);
            }
            AttentionRequest prefill = prefill_request(format, true, 128, 64);
            prefill.layout = layout;
            CELEG_TEST_CHECK(rejects(prefill,
                                     AttentionUnsupportedReason::LayoutNotImplemented,
                                     AttentionAlgorithm::Strict));
        }
    }
}


constexpr AttentionRequest constexpr_prefill(KvCacheMode format, bool fast, int head_dim) {
    AttentionRequest request{};
    request.kv_format = format;
    request.operation = AttentionOperation::Prefill;
    request.layout = AttentionKvLayout::Contiguous;
    request.position_source = AttentionPositionSource::HostScalar;
    request.fast_attention = fast;
    request.head_dim = head_dim;
    request.rows = 64;
    return request;
}

static_assert(resolve_attention_capability(
    constexpr_prefill(KvCacheMode::Bf16, true, 128)).algorithm ==
    AttentionAlgorithm::Flash);
static_assert(resolve_attention_capability(
    constexpr_prefill(KvCacheMode::Bf16, true, 64)).algorithm ==
    AttentionAlgorithm::Gemm);
static_assert(resolve_attention_capability(
    constexpr_prefill(KvCacheMode::Int8, true, 128)).algorithm ==
    AttentionAlgorithm::Online);
static_assert(resolve_attention_capability(
    constexpr_prefill(KvCacheMode::Int8, false, 128)).supported);

/// The regular attention kernels bake `1/sqrt(head_dim)` into their score
/// computation, so whatever the caller pre-multiplies into Q has to cancel that
/// factor out and leave exactly AttentionSpec::query_scale behind. Granite-style
/// checkpoints, whose attention_multiplier is 1/head_dim rather than
/// 1/sqrt(head_dim), are the case where applying query_scale directly silently
/// squares the scale and destroys the softmax.
void query_prescale_cancels_the_kernel_side_factor() {
    const auto effective = [](const AttentionSpec& layout) {
        return cuda_query_prescale(layout) /
               std::sqrt(static_cast<float>(layout.head_dim));
    };
    AttentionSpec granite;
    granite.head_dim = 64;
    granite.query_scale = 0.015625f;
    CELEG_TEST_CHECK(std::abs(effective(granite) - granite.query_scale) < 1e-7f);

    AttentionSpec conventional;
    conventional.head_dim = 128;
    conventional.query_scale = 1.0f / std::sqrt(128.0f);
    CELEG_TEST_CHECK(
        std::abs(effective(conventional) - conventional.query_scale) < 1e-7f);

    AttentionSpec degenerate;
    degenerate.head_dim = 0;
    degenerate.query_scale = 0.5f;
    CELEG_TEST_CHECK(cuda_query_prescale(degenerate) == 0.5f);
}

void run_all() {
    query_prescale_cancels_the_kernel_side_factor();
    matrix_rows_are_unique();
    resolution_never_contradicts_the_matrix();
    prefill_support_matrix_is_asymmetric();
    strict_paths();
    prefill_fast_path_selection();
    long_context_fallback_selection();
    alibi_combinations();
    relative_bias_combinations();
    paged_and_contiguous_parity();
}

}
}

int main() {
    celeg::run_all();
    std::cout << "cuda_attention_capability_test: ok\n";
    return 0;
}

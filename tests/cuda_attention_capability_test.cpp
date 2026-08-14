// Policy tests for the CUDA attention capability matrix.
//
// The matrix is host-only data, so these tests run on every preset: they check
// what the backend claims to support, not what a GPU computes.

#include "celeg/backend/cuda/attention_capability.hpp"
#include "support/assertions.hpp"

#include <algorithm>
#include <array>
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
constexpr std::array<AttentionPositionBias, 2> kBiases{
    AttentionPositionBias::None, AttentionPositionBias::Alibi};
constexpr std::array<AttentionAlgorithm, 6> kAlgorithms{
    AttentionAlgorithm::Strict, AttentionAlgorithm::Online,
    AttentionAlgorithm::Segmented, AttentionAlgorithm::Flash,
    AttentionAlgorithm::Gemm, AttentionAlgorithm::Alibi};

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
        // The message must name the combination, not just "unsupported".
        const std::string message = error.what();
        return error.reason() == reason && error.algorithm() == algorithm &&
               message.find(attention_algorithm_name(algorithm)) != std::string::npos &&
               message.find(attention_kv_format_name(request.kv_format)) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

// --- the matrix is well formed -------------------------------------------

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

// Every reachable request must resolve to a plan the matrix marks supported, or
// be rejected: it must never silently name a kernel that is not there.
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
    // The answer echoes the request: a plan can never be attributed to a
    // different combination than the one asked about.
    CELEG_TEST_CHECK(plan.kv_format == format);
    CELEG_TEST_CHECK(plan.operation == operation);
    CELEG_TEST_CHECK(plan.layout == layout);
    CELEG_TEST_CHECK(plan.position_source == source);
    CELEG_TEST_CHECK(plan.bias == bias);
    // ALiBi requests may only ever produce the fused ALiBi family.
    CELEG_TEST_CHECK((bias == AttentionPositionBias::Alibi) ==
                     (plan.algorithm == AttentionAlgorithm::Alibi));
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

// --- the support matrix as it stands today --------------------------------

// Prefill: Int8 has alibi/online/strict; BF16 additionally has flash/gemm/
// segmented.  This asymmetry is intentional and is asserted, not inferred.
void prefill_support_matrix_is_asymmetric() {
    const auto selectable = [](KvCacheMode format, AttentionAlgorithm algorithm) {
        const AttentionPositionBias bias = algorithm == AttentionAlgorithm::Alibi
            ? AttentionPositionBias::Alibi : AttentionPositionBias::None;
        return attention_capability(format, bias, AttentionOperation::Prefill,
                                    AttentionKvLayout::Contiguous,
                                    AttentionPositionSource::HostScalar,
                                    algorithm).supported;
    };
    CELEG_TEST_CHECK(selectable(KvCacheMode::Int8, AttentionAlgorithm::Alibi));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Int8, AttentionAlgorithm::Online));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Int8, AttentionAlgorithm::Strict));
    CELEG_TEST_CHECK(!selectable(KvCacheMode::Int8, AttentionAlgorithm::Flash));
    CELEG_TEST_CHECK(!selectable(KvCacheMode::Int8, AttentionAlgorithm::Gemm));
    CELEG_TEST_CHECK(!selectable(KvCacheMode::Int8, AttentionAlgorithm::Segmented));

    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Alibi));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Flash));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Gemm));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Segmented));
    CELEG_TEST_CHECK(selectable(KvCacheMode::Bf16, AttentionAlgorithm::Strict));
    // A chunk-local online prefill kernel exists but bulk prefill never picks it.
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

// --- strict paths ---------------------------------------------------------

void strict_paths() {
    for (KvCacheMode format : kFormats) {
        // Prefill without fast attention is exact two-pass softmax in both formats.
        CELEG_TEST_CHECK(require_attention_capability(
            prefill_request(format, false, 128, 4096)).algorithm ==
            AttentionAlgorithm::Strict);
        // ... regardless of head dimension or row count.
        CELEG_TEST_CHECK(require_attention_capability(
            prefill_request(format, false, 64, 1)).algorithm ==
            AttentionAlgorithm::Strict);
        // Decode without fast attention and without chunking is strict too.
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

// --- fast-path selection by head dimension and row count ------------------

void prefill_fast_path_selection() {
    // BF16: the tiled path is used above the preferred head dimension even when
    // the flash flag is absent, and only up to the maximum it supports.
    AttentionRequest request = prefill_request(KvCacheMode::Bf16, true, 128, 16);
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Flash);
    request.head_dim = 65;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Flash);
    // At or below the preferred head dimension the flag decides.
    request.head_dim = 64;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Gemm);
    request.flash_attention_requested = true;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Flash);
    // Above the tiled kernel's head-dimension ceiling the flag cannot force it.
    request.head_dim = 129;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Gemm);
    request.flash_attention_requested = false;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Gemm);

    // Int8 fast prefill is online regardless of head dimension and row count:
    // there is no Int8 tiled, GEMM or segmented prefill kernel to fall into.
    for (int head_dim : {32, 64, 128, 256}) {
        for (int rows : {1, 4096, 1 << 20}) {
            AttentionRequest int8 = prefill_request(KvCacheMode::Int8, true, head_dim, rows);
            int8.flash_attention_requested = true;
            CELEG_TEST_CHECK(require_attention_capability(int8).algorithm ==
                             AttentionAlgorithm::Online);
        }
    }
}

// --- long-context fallbacks ----------------------------------------------

void long_context_fallback_selection() {
    // BF16 prefill above the dense-scratch row ceiling falls back to segmented.
    AttentionRequest request = prefill_request(KvCacheMode::Bf16, true, 64,
                                               kAttentionMaxGemmRows);
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Gemm);
    request.rows = kAttentionMaxGemmRows + 1;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Segmented);
    // A flash-eligible head dimension keeps the tiled path at any row count.
    request.head_dim = 128;
    CELEG_TEST_CHECK(require_attention_capability(request).algorithm ==
                     AttentionAlgorithm::Flash);

    // Decode: the session-resolved chunking decision outranks fast attention.
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
        // Packed batch-pointer decode has no segmented kernel: asking for one is
        // a deliberate rejection, not a silent fall back to the online kernel.
        AttentionRequest packed = decode_request(
            format, AttentionKvLayout::BatchPointers,
            AttentionPositionSource::DeviceCounter);
        packed.segmented_attention = true;
        packed.fast_attention = true;
        CELEG_TEST_CHECK(rejects(packed, AttentionUnsupportedReason::NoKernelForCombination,
                                 AttentionAlgorithm::Segmented));
    }
}

// --- ALiBi ----------------------------------------------------------------

void alibi_combinations() {
    for (KvCacheMode format : kFormats) {
        // Prefill.
        AttentionRequest prefill = prefill_request(format, true, 128, 4096);
        prefill.bias = AttentionPositionBias::Alibi;
        prefill.flash_attention_requested = true;
        CELEG_TEST_CHECK(require_attention_capability(prefill).algorithm ==
                         AttentionAlgorithm::Alibi);
        // Device-position decode, every layout that has a kernel.
        for (AttentionKvLayout layout : kLayouts) {
            AttentionRequest decode = decode_request(
                format, layout, AttentionPositionSource::DeviceCounter);
            decode.bias = AttentionPositionBias::Alibi;
            decode.fast_attention = true;
            decode.segmented_attention = true;
            CELEG_TEST_CHECK(require_attention_capability(decode).algorithm ==
                             AttentionAlgorithm::Alibi);
        }
        // Host-position decode (prompt-embedding and tokenwise chunk prefill)
        // has no ALiBi kernel: it must reject rather than run the unbiased one.
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

// --- paged versus contiguous ---------------------------------------------

void paged_and_contiguous_parity() {
    for (KvCacheMode format : kFormats) {
        for (AttentionAlgorithm algorithm : {AttentionAlgorithm::Strict,
                                             AttentionAlgorithm::Online,
                                             AttentionAlgorithm::Segmented}) {
            // Device-position decode supports the same families paged and
            // contiguous.
            CELEG_TEST_CHECK(attention_capability(
                format, AttentionPositionBias::None, AttentionOperation::Decode,
                AttentionKvLayout::Contiguous, AttentionPositionSource::DeviceCounter,
                algorithm).supported);
            CELEG_TEST_CHECK(attention_capability(
                format, AttentionPositionBias::None, AttentionOperation::Decode,
                AttentionKvLayout::Paged, AttentionPositionSource::DeviceCounter,
                algorithm).supported);
        }
        // Bulk prefill exists only for the contiguous cache; the paged and
        // packed layouts prefill tokenwise through decode.
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

// --- the policy is compile-time ------------------------------------------

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

void run_all() {
    matrix_rows_are_unique();
    resolution_never_contradicts_the_matrix();
    prefill_support_matrix_is_asymmetric();
    strict_paths();
    prefill_fast_path_selection();
    long_context_fallback_selection();
    alibi_combinations();
    paged_and_contiguous_parity();
}

} // namespace
} // namespace celeg

int main() {
    celeg::run_all();
    std::cout << "cuda_attention_capability_test: ok\n";
    return 0;
}

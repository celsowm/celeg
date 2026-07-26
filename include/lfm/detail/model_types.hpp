#pragma once

// Hoisted model implementation types. These were previously nested inside
// LfmModel::Impl and are now at namespace lfm:: scope so that WeightLoader,
// GemmDispatcher, and IPackedSession can depend on them without being
// friends of LfmModel::Impl (Interface Segregation + Dependency Inversion).
//
// All types in this header are implementation details; they are not part of
// the public API and live under lfm:: so the detail/ headers can reference
// them without leaking the Impl class.

#include "lfm/cuda_utils.cuh"
#include "lfm/expert_offload.hpp"
#include "lfm/expert_residency.hpp"
#include "lfm/moe.hpp"
#include "lfm/model_shape.hpp"
#include "lfm/safetensors.hpp"
#include "lfm/pinned_expert_cache.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace lfm {

// ---------------------------------------------------------------------------
// GEMM plan keys (cuBLASLt).
// ---------------------------------------------------------------------------

struct MatmulKey {
    int m = 0;
    int n = 0;
    int k = 0;

    bool operator==(const MatmulKey& other) const {
        return m == other.m && n == other.n && k == other.k;
    }
};

struct MatmulKeyHash {
    size_t operator()(const MatmulKey& key) const {
        size_t value = static_cast<size_t>(key.m);
        value = value * 1315423911u + static_cast<size_t>(key.n);
        value = value * 2654435761u + static_cast<size_t>(key.k);
        return value;
    }
};

// Cached cuBLASLt matmul plan for one (m, n, k) shape. Holds the operation
// descriptor, matrix layouts, selected algorithm, and workspace size.
// Destroyed via RAII in the dtor.
struct LtPlan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatrixLayout_t d = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_size = 0;
    bool available = false;

    ~LtPlan() {
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

// ---------------------------------------------------------------------------
// Linear weight storage.
// ---------------------------------------------------------------------------

enum class LinearStorageKind : uint8_t {
    Bf16,
    Int8,
    Int4,
    // GGUF native block-quantized formats. Weights stay packed in their on-disk
    // super-block layout on the device; the matmul kernel dequantizes on the
    // fly. `gguf_blocks` points at the raw block bytes.
    Q4_K,
    Q6_K,
};

struct LinearWeight {
    LinearStorageKind kind = LinearStorageKind::Bf16;
    const __nv_bfloat16* bf16 = nullptr;
    const int8_t* int8 = nullptr;
    const uint8_t* int4 = nullptr;
    const float* scales = nullptr;
    // Raw GGUF super-block payload for Q4_K / Q6_K storage kinds.
    const uint8_t* gguf_blocks = nullptr;
    // Which GGUF block format `gguf_blocks` holds. Only meaningful when
    // `gguf_quantized()` is true.
    GgmlType gguf_type = GgmlType::Unknown;
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

// Packed expert linear weight. For the LFM2 MoE architecture, expert
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
    int experts = 0;
    int rows_per_expert = 0;
    int cols = 0;

    LinearWeight expert_view(int expert_id) const;
};

// Dense (SwiGLU) feed-forward weights.
struct DenseFfnWeights {
    const LinearWeight* w13 = nullptr;
    const LinearWeight* w2 = nullptr;
};

// Mixture-of-experts feed-forward weights.
struct MoeFfnWeights {
    const LinearWeight* router = nullptr;
    const float* expert_bias = nullptr;
    const ExpertLinearWeight* gate_up = nullptr;
    const ExpertLinearWeight* down = nullptr;
    // Device-resident float copy of `router` ([num_experts * hidden]), produced
    // once at load time so the CUDA router kernel (which expects float) does
    // not re-cast every token. Owned by the session Impl, not by this view.
    const float* router_float = nullptr;

    // Expert-offload indirection tables ([num_experts] device pointer arrays).
    // Non-null only when this layer's experts are offloaded (host-backed with a
    // GPU cache). When set, gate_up/down above are null and the MoE FFN kernel
    // resolves experts through these tables. Owned by the session's per-layer
    // ExpertLayerCache, not by this view.
    const __nv_bfloat16* const* gate_up_ptrs = nullptr;
    const __nv_bfloat16* const* down_ptrs = nullptr;

    bool offloaded() const { return gate_up_ptrs != nullptr; }
};

// A layer's feed-forward block is either dense or MoE. The layer operator
// (attention or convolution) is independent of the FFN type.
using FeedForwardWeights = std::variant<DenseFfnWeights, MoeFfnWeights>;

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
    } else {
        const size_t packed_cols = (static_cast<size_t>(cols) + 1) / 2;
        view.int4 = int4 + expert_offset / static_cast<size_t>(cols) * packed_cols;
        view.scales = scales + scale_offset;
    }
    return view;
}

struct DeviceWeight {
    DeviceBuffer<__nv_bfloat16> bf16_storage;
    DeviceBuffer<int8_t> int8_storage;
    DeviceBuffer<uint8_t> int4_storage;
    DeviceBuffer<uint8_t> gguf_blocks_storage;
    DeviceBuffer<float> scales_storage;
    std::vector<int64_t> shape;
    LinearWeight linear;
};

using WeightMap = std::unordered_map<std::string, DeviceWeight>;

// Process-wide shared weight arena. Multiple LfmModel sessions on the same
// device + checkpoint + weight_mode share one instance to avoid duplicate
// GPU allocations.
struct SharedModelWeights {
    std::mutex mutex;
    WeightMap tensors;

    std::shared_ptr<IWeightRepository> repo;

    ExpertOffloadPlan expert_offload_plan;
    HostExpertStore host_expert_store;
    std::vector<std::unique_ptr<ExpertLayerCache>> expert_caches;
    std::unique_ptr<CudaStream> expert_transfer_stream;
    std::vector<std::vector<ExpertLocation>> expert_catalog;
    std::unique_ptr<PinnedExpertCache> pinned_expert_cache;
    std::unique_ptr<ExpertSidecar> expert_sidecar;
    std::unique_ptr<ExpertIoManager> expert_io_manager;
    ModelUsageStats usage_stats;
    std::string usage_profile_path;

    struct InflightTransfer {
        ExpertHostLease lease;
        std::unique_ptr<CudaEvent> event;
    };
    std::vector<InflightTransfer> inflight_transfers;
    CudaEvent ffn_done_event;
    CudaEvent promote_done_event;
    CudaEvent prefetch_done_event;
    CudaEvent router_done_event;
    std::vector<int> cold_expert_host;
    std::vector<float> cold_scores_host;
    std::vector<int> prefetch_idx;
    std::vector<int> prefetch_ranked;
    std::vector<float> prefetch_scores;

    void ensure_moe_experts_resident(
        int layer, const int* sel_dev, int rows, int K, int num_experts,
        cudaStream_t compute_stream, const float* route_scores_dev);

    size_t memory_bytes() const;
};

// ---------------------------------------------------------------------------
// Per-layer topology.
// ---------------------------------------------------------------------------

struct LayerCommon {
    const __nv_bfloat16* operator_norm = nullptr;
    const __nv_bfloat16* ffn_norm = nullptr;
    FeedForwardWeights feed_forward;
};

struct AttentionLayer {
    LayerCommon common;
    const LinearWeight* qkv = nullptr;
    const LinearWeight* out = nullptr;
    const __nv_bfloat16* q_norm = nullptr;
    const __nv_bfloat16* k_norm = nullptr;
    DeviceBuffer<__nv_bfloat16> key_cache;
    DeviceBuffer<__nv_bfloat16> value_cache;
    DeviceBuffer<int8_t> key_cache_int8;
    DeviceBuffer<int8_t> value_cache_int8;
    DeviceBuffer<float> key_cache_scales;
    DeviceBuffer<float> value_cache_scales;
};

struct ConvolutionLayer {
    LayerCommon common;
    const LinearWeight* conv_in = nullptr;
    const __nv_bfloat16* conv_weight = nullptr;
    const LinearWeight* conv_out = nullptr;
    DeviceBuffer<__nv_bfloat16> conv_state;
};

using Layer = std::variant<AttentionLayer, ConvolutionLayer>;

// Free-function visitors (replaces the old Impl::common / as_attention /
// as_convolution statics). Putting them at namespace scope means callers
// in packed.cu no longer need `friend struct PackedDecodeExecutorImpl`.
inline LayerCommon& common(Layer& layer) {
    return std::visit([](auto& value) -> LayerCommon& { return value.common; }, layer);
}
inline const LayerCommon& common(const Layer& layer) {
    return std::visit([](const auto& value) -> const LayerCommon& { return value.common; }, layer);
}
inline AttentionLayer* as_attention(Layer& layer) {
    return std::get_if<AttentionLayer>(&layer);
}
inline const AttentionLayer* as_attention(const Layer& layer) {
    return std::get_if<AttentionLayer>(&layer);
}
inline ConvolutionLayer* as_convolution(Layer& layer) {
    return std::get_if<ConvolutionLayer>(&layer);
}
inline const ConvolutionLayer* as_convolution(const Layer& layer) {
    return std::get_if<ConvolutionLayer>(&layer);
}

// Feed-forward visitors. These decouple call sites from whether a layer uses
// the dense SwiGLU FFN or the MoE FFN; dispatch is done via the variant.
inline DenseFfnWeights* as_dense_ffn(FeedForwardWeights& ff) {
    return std::get_if<DenseFfnWeights>(&ff);
}
inline const DenseFfnWeights* as_dense_ffn(const FeedForwardWeights& ff) {
    return std::get_if<DenseFfnWeights>(&ff);
}
inline MoeFfnWeights* as_moe_ffn(FeedForwardWeights& ff) {
    return std::get_if<MoeFfnWeights>(&ff);
}
inline const MoeFfnWeights* as_moe_ffn(const FeedForwardWeights& ff) {
    return std::get_if<MoeFfnWeights>(&ff);
}
inline bool is_moe_ffn(const FeedForwardWeights& ff) {
    return std::holds_alternative<MoeFfnWeights>(ff);
}

// Builds the MoE router config / FFN device descriptor from the model shape and
// MoE weights. Shared by the standalone decode/prefill paths (model.cu) and the
// packed executor (packed.cu) so both stay in lock-step with the checkpoint
// topology. Defined inline here (where MoeFfnWeights is complete) to avoid
// duplicating the descriptor construction across translation units.
inline lfm::MoeRouterConfig moe_router_config(const ModelShape& shape) {
    lfm::MoeRouterConfig cfg;
    cfg.num_experts = shape.num_experts;
    cfg.experts_per_token = shape.experts_per_token;
    cfg.normalize_topk = shape.normalize_topk;
    cfg.use_expert_bias = shape.use_expert_bias;
    cfg.routed_scaling_factor = shape.routed_scaling_factor;
    return cfg;
}

inline lfm::MoeFfnDevice moe_ffn_device(const MoeFfnWeights& moe, const ModelShape& shape) {
    lfm::MoeFfnDevice fdev;
    fdev.num_experts = shape.num_experts;
    fdev.inter = shape.moe_intermediate;
    fdev.hidden_dim = shape.hidden;
    if (moe.offloaded()) {
        // Host-backed experts with a GPU cache: resolve through the per-expert
        // pointer tables. Strides are unused in indirect mode.
        fdev.gate_up_ptrs = moe.gate_up_ptrs;
        fdev.down_ptrs = moe.down_ptrs;
    } else {
        fdev.gate_up = moe.gate_up->bf16;
        fdev.down = moe.down->bf16;
        fdev.expert_gate_up_stride =
            static_cast<size_t>(2) * shape.moe_intermediate * shape.hidden;
        fdev.expert_down_stride =
            static_cast<size_t>(shape.hidden) * shape.moe_intermediate;
    }
    return fdev;
}

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
    return result;
}

} // namespace lfm

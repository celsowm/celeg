#pragma once

// Hoisted model implementation types. These were previously nested inside
// The extracted model components are now at namespace celeg:: scope so that WeightLoader,
// GemmDispatcher and PackedSessionContext can depend on them without being
// friends of the compiled model (Interface Segregation + Dependency Inversion).
//
// All types in this header are implementation details; they are not part of
// the public API and live under celeg:: so the detail/ headers can reference
// them without leaking backend state classes.

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/backend/cuda/moe/offload.hpp"
#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/checkpoint/formats/safetensors.hpp"
#include "celeg/runtime/cache/host_expert_cache.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace celeg {

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
    const uint8_t* gguf_blocks = nullptr;
    GgmlType gguf_type = GgmlType::Unknown;
    size_t gguf_row_bytes = 0;
    size_t gguf_expert_stride = 0;
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
    // not re-cast every token. Owned by the session state, not by this view.
    const float* router_float = nullptr;

    // Expert-offload indirection tables ([num_experts] device pointer arrays).
    // Non-null only when this layer's experts are offloaded (host-backed with a
    // GPU cache). When set, gate_up/down above are null and the MoE FFN kernel
    // resolves experts through these tables. Owned by the session's per-layer
    // ExpertLayerCache, not by this view.
    const __nv_bfloat16* const* gate_up_ptrs = nullptr;
    const __nv_bfloat16* const* down_ptrs = nullptr;

    const LinearWeight* shared_w13 = nullptr;
    const LinearWeight* shared_w2 = nullptr;
    const LinearWeight* shared_gate = nullptr;

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

struct DeviceWeight {
    DeviceBuffer<__nv_bfloat16> bf16_storage;
    DeviceBuffer<int8_t> int8_storage;
    DeviceBuffer<uint8_t> int4_storage;
    DeviceBuffer<uint8_t> gguf_expert_storage;
    std::vector<DeviceBuffer<uint8_t>> gguf_segment_storage;
    DeviceBuffer<float> scales_storage;
    std::vector<int64_t> shape;
    LinearWeight linear;
};

using WeightMap = std::unordered_map<std::string, DeviceWeight>;

struct ResidencyController {
    std::mutex mutex;
    std::unique_ptr<ExpertLayerCache> cache;
    std::unique_ptr<CudaStream> transfer_stream;

    struct InflightTransfer {
        ExpertHostLease lease;
        std::unique_ptr<CudaEvent> event;
    };
    std::vector<InflightTransfer> inflight_transfers;
};

class CudaExpertSource;

// Process-wide shared weight arena. Multiple CudaModel sessions on the same
// device + checkpoint + weight_mode share one instance to avoid duplicate
// GPU allocations.
struct SharedModelWeights {
    std::mutex mutex;
    std::string residency_fingerprint;
    WeightMap tensors;

    std::shared_ptr<IWeightRepository> repo;

    ExpertOffloadPlan expert_offload_plan;
    HostExpertStore host_expert_store;
    std::vector<std::unique_ptr<ResidencyController>> expert_controllers;
    std::vector<std::vector<ExpertLocation>> expert_catalog;
    std::unique_ptr<HostExpertCache> host_expert_cache;
    std::shared_ptr<CudaExpertSource> expert_source;
    std::unique_ptr<ExpertSidecar> expert_sidecar;
    std::unique_ptr<ExpertIoManager> expert_io_manager;
    ModelUsageStats usage_stats;
    std::string usage_profile_path;

    std::shared_ptr<CudaExpertResidencyCoordinator> residency_coordinator;

    size_t memory_bytes() const;
};

// ---------------------------------------------------------------------------
// Per-layer topology.
// ---------------------------------------------------------------------------

struct LayerCommon {
    const __nv_bfloat16* operator_norm = nullptr;
    const __nv_bfloat16* post_attention_norm = nullptr;
    const __nv_bfloat16* ffn_norm = nullptr;
    const __nv_bfloat16* post_feed_forward_norm = nullptr;
    FeedForwardWeights feed_forward;
    const LinearWeight* per_layer_input_gate = nullptr;
    const LinearWeight* per_layer_projection = nullptr;
    const __nv_bfloat16* per_layer_input_norm = nullptr;
    const __nv_bfloat16* layer_scalar = nullptr;
};

struct AttentionLayer {
    LayerCommon common;
    AttentionSpec layout;
    const LinearWeight* query = nullptr;
    const LinearWeight* key = nullptr;
    const LinearWeight* value = nullptr;
    const LinearWeight* out = nullptr;
    const LinearWeight* latent_query = nullptr;
    const LinearWeight* latent_query_rope = nullptr;
    const LinearWeight* latent_key = nullptr;
    const LinearWeight* latent_value = nullptr;
    const LinearWeight* latent_key_rope = nullptr;
    const __nv_bfloat16* q_norm = nullptr;
    const __nv_bfloat16* k_norm = nullptr;
    DeviceBuffer<__nv_bfloat16> key_cache;
    DeviceBuffer<__nv_bfloat16> value_cache;
    DeviceBuffer<int8_t> key_cache_int8;
    DeviceBuffer<int8_t> value_cache_int8;
    DeviceBuffer<float> key_cache_scales;
    DeviceBuffer<float> value_cache_scales;
    DeviceBuffer<__nv_bfloat16> latent_key_cache;
    DeviceBuffer<__nv_bfloat16> latent_value_cache;
    DeviceBuffer<__nv_bfloat16> latent_key_rope_cache;
    DeviceBuffer<float> alibi_slopes;
    int kv_owner_layer = -1;
};

struct ConvolutionLayer {
    LayerCommon common;
    const LinearWeight* conv_in = nullptr;
    const __nv_bfloat16* conv_weight = nullptr;
    const LinearWeight* conv_out = nullptr;
    DeviceBuffer<__nv_bfloat16> conv_state;
};

struct Mamba2Layer {
    LayerCommon common;
    Mamba2Spec spec;
    const LinearWeight* in = nullptr;
    const __nv_bfloat16* conv_weight = nullptr;
    const __nv_bfloat16* conv_bias = nullptr;
    const __nv_bfloat16* dt_bias = nullptr;
    const __nv_bfloat16* a_log = nullptr;
    const __nv_bfloat16* d = nullptr;
    const __nv_bfloat16* norm = nullptr;
    const LinearWeight* out = nullptr;
    DeviceBuffer<__nv_bfloat16> conv_state;
    DeviceBuffer<__nv_bfloat16> ssm_state;
};

struct GatedDeltaNetLayer {
    LayerCommon common;
    GatedDeltaNetSpec spec;
    const LinearWeight* qkv = nullptr;
    const LinearWeight* z = nullptr;
    const LinearWeight* b = nullptr;
    const LinearWeight* a = nullptr;
    const __nv_bfloat16* conv_weight = nullptr;
    const __nv_bfloat16* dt_bias = nullptr;
    const __nv_bfloat16* a_log = nullptr;
    const __nv_bfloat16* norm = nullptr;
    const LinearWeight* out = nullptr;
    DeviceBuffer<__nv_bfloat16> conv_state;
    DeviceBuffer<__nv_bfloat16> recurrent_state;
};

struct MlpOnlyLayer {
    LayerCommon common;
    MlpBlockSpec spec;
    const LinearWeight* up = nullptr;
    const LinearWeight* down = nullptr;
};

using Layer = std::variant<AttentionLayer, ConvolutionLayer, GatedDeltaNetLayer,
                           Mamba2Layer, MlpOnlyLayer>;

// Free-function visitors replace the old common / as_attention /
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
inline Mamba2Layer* as_mamba2(Layer& layer) {
    return std::get_if<Mamba2Layer>(&layer);
}
inline const Mamba2Layer* as_mamba2(const Layer& layer) {
    return std::get_if<Mamba2Layer>(&layer);
}
inline GatedDeltaNetLayer* as_gated_delta_net(Layer& layer) {
    return std::get_if<GatedDeltaNetLayer>(&layer);
}
inline const GatedDeltaNetLayer* as_gated_delta_net(const Layer& layer) {
    return std::get_if<GatedDeltaNetLayer>(&layer);
}
inline MlpOnlyLayer* as_mlp_only(Layer& layer) {
    return std::get_if<MlpOnlyLayer>(&layer);
}
inline const MlpOnlyLayer* as_mlp_only(const Layer& layer) {
    return std::get_if<MlpOnlyLayer>(&layer);
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
inline celeg::MoeRouterConfig moe_router_config(const RuntimeTopology& shape) {
    celeg::MoeRouterConfig cfg;
    cfg.num_experts = shape.num_experts;
    cfg.experts_per_token = shape.experts_per_token;
    cfg.normalize_topk = shape.normalize_topk;
    cfg.softmax = shape.moe_router_softmax;
    cfg.use_expert_bias = shape.use_expert_bias;
    cfg.routed_scaling_factor = shape.routed_scaling_factor;
    return cfg;
}

inline celeg::MoeFfnDevice moe_ffn_device(const MoeFfnWeights& moe, const RuntimeTopology& shape) {
    celeg::MoeFfnDevice fdev;
    fdev.num_experts = shape.num_experts;
    fdev.inter = shape.moe_intermediate;
    fdev.hidden_dim = shape.hidden;
    if (moe.offloaded()) {
        // Host-backed experts with a GPU cache: resolve through the per-expert
        // pointer tables. Strides are unused in indirect mode.
        fdev.gate_up_ptrs = moe.gate_up_ptrs;
        fdev.down_ptrs = moe.down_ptrs;
    } else {
        if (moe.gate_up->kind == LinearStorageKind::Q4_K ||
            moe.gate_up->kind == LinearStorageKind::Q6_K) {
            fdev.gate_up_gguf = moe.gate_up->gguf_blocks;
            fdev.down_gguf = moe.down->gguf_blocks;
            fdev.gate_up_gguf_type = moe.gate_up->gguf_type;
            fdev.down_gguf_type = moe.down->gguf_type;
            fdev.expert_gate_up_row_bytes = moe.gate_up->gguf_row_bytes;
            fdev.expert_down_row_bytes = moe.down->gguf_row_bytes;
            fdev.expert_gate_up_byte_stride = moe.gate_up->gguf_expert_stride;
            fdev.expert_down_byte_stride = moe.down->gguf_expert_stride;
        } else {
            fdev.gate_up = moe.gate_up->bf16;
            fdev.down = moe.down->bf16;
            fdev.expert_gate_up_stride =
                static_cast<size_t>(2) * shape.moe_intermediate * shape.hidden;
            fdev.expert_down_stride =
                static_cast<size_t>(shape.hidden) * shape.moe_intermediate;
        }
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

#pragma once

// Per-layer execution state: the mixer alternatives, the `Layer` variant and
// its exhaustive dispatch helpers.
//
// Depends on the weight-binding headers (for `LinearWeight` /
// `FeedForwardWeights`), on `DeviceBuffer` for the per-layer KV and recurrent
// state, and on the resolved graph specs. It deliberately does NOT depend on
// weight ownership (`DeviceWeight`/`WeightMap`), expert offload/residency
// state or the cuBLASLt plan cache.

#include "celeg/detail/exhaustive_visit.hpp"
#include "celeg/detail/model/feed_forward_weights.hpp"
#include "celeg/detail/model/linear_weights.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/model/graph.hpp"

#include <cuda_bf16.h>

#include <cstdint>
#include <utility>
#include <variant>

namespace celeg {

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
    const LinearWeight* gate = nullptr;
    const LinearWeight* out = nullptr;
    const LinearWeight* latent_query = nullptr;
    const LinearWeight* latent_query_rope = nullptr;
    const LinearWeight* latent_key = nullptr;
    const LinearWeight* latent_value = nullptr;
    const LinearWeight* latent_key_rope = nullptr;
    const LinearWeight* latent_query_projection = nullptr;
    const LinearWeight* latent_query_expansion = nullptr;
    const LinearWeight* latent_key_projection = nullptr;
    const LinearWeight* latent_expansion = nullptr;
    const __nv_bfloat16* latent_query_norm = nullptr;
    const __nv_bfloat16* latent_key_norm = nullptr;
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
    const LinearWeight* q = nullptr;
    const LinearWeight* k = nullptr;
    const LinearWeight* v = nullptr;
    const LinearWeight* z = nullptr;
    const LinearWeight* b = nullptr;
    const LinearWeight* a = nullptr;
    const __nv_bfloat16* conv_weight = nullptr;
    DeviceBuffer<__nv_bfloat16> factorized_conv_weight;
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
// in packed/kernels.cu no longer need `friend struct PackedDecodeExecutorImpl`.
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

// Exhaustive mixer dispatch. Every alternative of `Layer` must be matched by a
// handler naming its concrete type; there is no generic fallback, so adding a
// mixer turns every dispatch site that has not been updated into a compile
// error. Handlers receive a non-null pointer to the active alternative.
template <typename LayerRef, typename... Handlers>
decltype(auto) visit_layer(LayerRef&& layer, Handlers&&... handlers) {
    return visit_exhaustive(std::forward<LayerRef>(layer),
                            std::forward<Handlers>(handlers)...);
}

} // namespace celeg

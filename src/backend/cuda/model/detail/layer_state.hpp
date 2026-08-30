#pragma once


#include "celeg/detail/exhaustive_visit.hpp"
#include "feed_forward_weights.hpp"
#include "linear_weights.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/model/graph.hpp"

#include <cuda_bf16.h>

#include <cstdint>
#include <utility>
#include <variant>

namespace celeg {


struct LayerCommon {
    const __nv_bfloat16* mixer_norm_before = nullptr;
    const __nv_bfloat16* mixer_norm_after = nullptr;
    const __nv_bfloat16* feed_forward_norm_before = nullptr;
    const __nv_bfloat16* feed_forward_norm_after = nullptr;
    FeedForwardWeights feed_forward;
    const LinearWeight* per_layer_input_gate = nullptr;
    const LinearWeight* per_layer_projection = nullptr;
    const __nv_bfloat16* per_layer_input_norm = nullptr;
    const __nv_bfloat16* layer_scalar = nullptr;
};

struct OrdinaryBf16KvState {
    DeviceBuffer<__nv_bfloat16> key_cache;
    DeviceBuffer<__nv_bfloat16> value_cache;
};

struct OrdinaryInt8KvState {
    DeviceBuffer<int8_t> key_cache;
    DeviceBuffer<int8_t> value_cache;
    DeviceBuffer<float> key_scales;
    DeviceBuffer<float> value_scales;
};

struct LatentAttentionRuntimeState {
    DeviceBuffer<__nv_bfloat16> latent_key_cache;
    DeviceBuffer<__nv_bfloat16> latent_value_cache;
    DeviceBuffer<__nv_bfloat16> latent_key_rope_cache;
};

using AttentionRuntimeState = std::variant<
    OrdinaryBf16KvState, OrdinaryInt8KvState, LatentAttentionRuntimeState>;

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
    AttentionRuntimeState state = OrdinaryBf16KvState{};
    DeviceBuffer<float> alibi_slopes;
    int kv_owner_layer = -1;

    OrdinaryBf16KvState* bf16_state() { return std::get_if<OrdinaryBf16KvState>(&state); }
    const OrdinaryBf16KvState* bf16_state() const { return std::get_if<OrdinaryBf16KvState>(&state); }
    OrdinaryInt8KvState* int8_state() { return std::get_if<OrdinaryInt8KvState>(&state); }
    const OrdinaryInt8KvState* int8_state() const { return std::get_if<OrdinaryInt8KvState>(&state); }
    LatentAttentionRuntimeState* latent_kv_state() {
        return std::get_if<LatentAttentionRuntimeState>(&state);
    }
    const LatentAttentionRuntimeState* latent_kv_state() const {
        return std::get_if<LatentAttentionRuntimeState>(&state);
    }

    __nv_bfloat16* key_cache_bf16() {
        auto* s = bf16_state(); return s ? s->key_cache.data() : nullptr;
    }
    const __nv_bfloat16* key_cache_bf16() const {
        auto* s = bf16_state(); return s ? s->key_cache.data() : nullptr;
    }
    __nv_bfloat16* value_cache_bf16() {
        auto* s = bf16_state(); return s ? s->value_cache.data() : nullptr;
    }
    const __nv_bfloat16* value_cache_bf16() const {
        auto* s = bf16_state(); return s ? s->value_cache.data() : nullptr;
    }

    int8_t* key_cache_int8_ptr() {
        auto* s = int8_state(); return s ? s->key_cache.data() : nullptr;
    }
    const int8_t* key_cache_int8_ptr() const {
        auto* s = int8_state(); return s ? s->key_cache.data() : nullptr;
    }
    int8_t* value_cache_int8_ptr() {
        auto* s = int8_state(); return s ? s->value_cache.data() : nullptr;
    }
    const int8_t* value_cache_int8_ptr() const {
        auto* s = int8_state(); return s ? s->value_cache.data() : nullptr;
    }
    float* key_cache_scales_ptr() {
        auto* s = int8_state(); return s ? s->key_scales.data() : nullptr;
    }
    const float* key_cache_scales_ptr() const {
        auto* s = int8_state(); return s ? s->key_scales.data() : nullptr;
    }
    float* value_cache_scales_ptr() {
        auto* s = int8_state(); return s ? s->value_scales.data() : nullptr;
    }
    const float* value_cache_scales_ptr() const {
        auto* s = int8_state(); return s ? s->value_scales.data() : nullptr;
    }

    __nv_bfloat16* latent_key_cache_ptr() {
        auto* s = latent_kv_state(); return s ? s->latent_key_cache.data() : nullptr;
    }
    const __nv_bfloat16* latent_key_cache_ptr() const {
        auto* s = latent_kv_state(); return s ? s->latent_key_cache.data() : nullptr;
    }
    __nv_bfloat16* latent_value_cache_ptr() {
        auto* s = latent_kv_state(); return s ? s->latent_value_cache.data() : nullptr;
    }
    const __nv_bfloat16* latent_value_cache_ptr() const {
        auto* s = latent_kv_state(); return s ? s->latent_value_cache.data() : nullptr;
    }
    __nv_bfloat16* latent_key_rope_cache_ptr() {
        auto* s = latent_kv_state(); return s ? s->latent_key_rope_cache.data() : nullptr;
    }
    const __nv_bfloat16* latent_key_rope_cache_ptr() const {
        auto* s = latent_kv_state(); return s ? s->latent_key_rope_cache.data() : nullptr;
    }

    size_t runtime_state_bytes() const {
        return std::visit([](const auto& s) -> size_t {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, OrdinaryBf16KvState>) {
                return s.key_cache.bytes() + s.value_cache.bytes();
            } else if constexpr (std::is_same_v<T, OrdinaryInt8KvState>) {
                return s.key_cache.bytes() + s.value_cache.bytes() +
                    s.key_scales.bytes() + s.value_scales.bytes();
            } else {
                return s.latent_key_cache.bytes() + s.latent_value_cache.bytes() +
                    s.latent_key_rope_cache.bytes();
            }
        }, state);
    }

    void release_runtime_state_buffers() {
        std::visit([](auto& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, OrdinaryBf16KvState>) {
                s.key_cache.reset(0);
                s.value_cache.reset(0);
            } else if constexpr (std::is_same_v<T, OrdinaryInt8KvState>) {
                s.key_cache.reset(0);
                s.value_cache.reset(0);
                s.key_scales.reset(0);
                s.value_scales.reset(0);
            } else {
                s.latent_key_cache.reset(0);
                s.latent_value_cache.reset(0);
                s.latent_key_rope_cache.reset(0);
            }
        }, state);
    }
};

struct ConvolutionLayer {
    LayerCommon common;
    ShortConvolutionSpec spec;
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

template <typename LayerRef, typename... Handlers>
decltype(auto) visit_layer(LayerRef&& layer, Handlers&&... handlers) {
    return visit_exhaustive(std::forward<LayerRef>(layer),
                            std::forward<Handlers>(handlers)...);
}

}

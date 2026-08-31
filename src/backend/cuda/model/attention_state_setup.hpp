#pragma once

#include "detail/layer_state.hpp"
#include "backend/cuda/runtime_types.hpp"

#include <cstddef>

namespace celeg {

inline void initialize_cuda_ordinary_attention_state(
    AttentionLayer& attention,
    const AttentionSpec& layout,
    KvCacheMode kv_cache_mode,
    int max_context,
    bool allocate_cache) {
    if (kv_cache_mode == KvCacheMode::Int8) {
        attention.state = OrdinaryInt8KvState{};
    } else {
        attention.state = OrdinaryBf16KvState{};
    }
    if (!allocate_cache) return;

    const size_t cache_elements = static_cast<size_t>(max_context) *
        static_cast<size_t>(layout.key_value_width());
    if (kv_cache_mode == KvCacheMode::Int8) {
        auto& state = std::get<OrdinaryInt8KvState>(attention.state);
        state.key_cache.reset(cache_elements);
        state.value_cache.reset(cache_elements);
        const size_t scale_elements = static_cast<size_t>(max_context) *
            static_cast<size_t>(layout.key_value_heads);
        state.key_scales.reset(scale_elements);
        state.value_scales.reset(scale_elements);
        return;
    }

    auto& state = std::get<OrdinaryBf16KvState>(attention.state);
    state.key_cache.reset(cache_elements);
    state.value_cache.reset(cache_elements);
}

}

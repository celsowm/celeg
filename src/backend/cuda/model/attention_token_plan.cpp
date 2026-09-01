#include "detail/compiled_model.hpp"
#include "attention_decode_dispatch.hpp"
#include "attention_layer_support.hpp"

namespace celeg {

AttentionCapability CudaCompiledModel::token_attention_plan(
    AttentionLayer& attention, const AttentionSpec& owner_layout,
    const TokenKvPolicy& kv) {
    const AttentionPositionBias bias =
        cuda_attention_position_bias(attention.layout);
    const AttentionPositionSource position_source =
        bias == AttentionPositionBias::None
            ? kv.position_source
            : AttentionPositionSource::DeviceCounter;
    return plan_cuda_decode_attention(
        attention.layout,
        resources_.options().kv_cache_mode,
        kv.kv_layout,
        position_source,
        resources_.options().fast_attention,
        kv.paged() && use_segmented_attention(session_.position_),
        owner_layout.head_dim).plan;
}

}

#include "detail/compiled_model.hpp"
#include "attention_decode_dispatch.hpp"

namespace celeg {

AttentionCapability CudaCompiledModel::token_attention_plan(
    AttentionLayer& attention, const AttentionSpec& owner_layout,
    const TokenKvPolicy& kv) {
    return plan_cuda_decode_attention(
        attention.layout,
        resources_.options().kv_cache_mode,
        kv.kv_layout,
        kv.position_source,
        resources_.options().fast_attention,
        kv.paged() && use_segmented_attention(session_.position_),
        attention.alibi_slopes.data() != nullptr,
        owner_layout.head_dim).plan;
}

}

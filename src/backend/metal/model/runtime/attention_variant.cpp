#include "attention_variant.hpp"

#include "celeg/model/weights/roles.hpp"

namespace celeg {

namespace {

bool is_multi(const CompiledAttentionProgram& a) {
    return a.semantics.multi_axis_position() != nullptr;
}

bool is_no_position(const CompiledAttentionProgram& a) {
    return std::holds_alternative<NoPositionEncodingSpec>(a.semantics.position);
}

bool is_split_half(const CompiledAttentionProgram& a) {
    if (auto* r = a.semantics.rope_position()) return r->pairing == RopePairingKind::SplitHalf;
    return false;
}

bool per_head(const std::optional<NormSpec>& n) {
    return n && n->granularity == NormGranularity::PerHead;
}

}

AttentionVariant resolve_attention_variant(
    const CompiledAttentionProgram& attention,
    uint32_t rows,
    uint32_t base_position,
    uint32_t head_dim,
    bool owns_kv) noexcept {
    AttentionVariant v;
    v.owns_kv = owns_kv;
    v.no_position = is_no_position(attention);
    v.split_half_rope = is_split_half(attention);
    v.multi_axis = is_multi(attention);
    v.fused_per_head = owns_kv && !v.multi_axis &&
        per_head(attention.semantics.query_norm) && per_head(attention.semantics.key_norm);
    v.qk_publishes_kv = v.fused_per_head && !v.no_position && v.split_half_rope;
    // tiled candidate mirrors attention.mm:568-573 (fast, no bias, window 0, base 0, hd 64, rows%32==0)
    // The full gate (relative/alibi/window/base) is checked by caller; here we record geometry only.
    v.tiled_candidate = head_dim == 64 && (rows % 32 == 0);
    if (v.fused_per_head) {
        if (v.no_position) v.qk_kernel = "celeg_qk_norm_batch_no_position";
        else if (v.split_half_rope) v.qk_kernel = "celeg_qk_norm_rope_batch_split_store_kv";
        else v.qk_kernel = "celeg_qk_norm_rope_batch";
    } else {
        if (v.multi_axis) v.qk_kernel = "celeg_qk_mrope_position_batch";
        else v.qk_kernel = "celeg_qk_position_batch";
    }
    return v;
}

std::string_view select_decode_attention_kernel(
    bool has_relative,
    bool has_alibi,
    uint32_t window_size) noexcept {
    if (has_relative) return "celeg_attention_relative_bias";
    if (has_alibi) return "celeg_attention_alibi";
    if (window_size > 0) return "celeg_attention_sliding";
    return "celeg_attention";
}

std::string_view select_batch_attention_kernel(
    bool has_relative,
    bool has_alibi,
    uint32_t window_size) noexcept {
    if (has_relative) return "celeg_attention_batch_relative_bias";
    if (has_alibi) return "celeg_attention_batch_alibi";
    if (window_size > 0) return "celeg_attention_batch_sliding";
    return "celeg_attention_batch";
}

}

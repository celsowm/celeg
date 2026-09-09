
/** @brief ALiBi score policy for the shared warp-online attention core. */
struct AlibiScorePolicy {
    const float* slopes;

    __device__ __forceinline__ float score(
        float dot, float scale, int head, int query_position,
        int key_position) const {
        return dot * scale - slopes[head] *
            static_cast<float>(query_position - key_position);
    }
};

void launch_gqa_decode_alibi_device(const GqaContiguousArgs& args) {
    const GqaGeometry& g = args.geometry;
    const OnlineContiguousBf16Storage storage{
        args.kv.keys, args.kv.values, g.kv_heads};
    gqa_online_attention_kernel<<<g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out,
        OnlineSinglePosition{args.extent.position},
        AlibiScorePolicy{args.alibi_slopes}, 1,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_alibi(const GqaContiguousArgs& args) {
    const GqaGeometry& g = args.geometry;
    const int rows = args.extent.rows;
    const OnlineContiguousBf16Storage storage{
        args.kv.keys, args.kv.values, g.kv_heads};
    gqa_online_attention_kernel<<<rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, OnlinePrefillPosition{},
        AlibiScorePolicy{args.alibi_slopes}, rows,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_alibi_int8_device(const GqaContiguousInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const OnlineContiguousInt8Storage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, g.kv_heads};
    gqa_online_attention_kernel<<<g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out,
        OnlineSinglePosition{args.extent.position},
        AlibiScorePolicy{args.alibi_slopes}, 1,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_alibi_int8(const GqaContiguousInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const int rows = args.extent.rows;
    const OnlineContiguousInt8Storage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, g.kv_heads};
    gqa_online_attention_kernel<<<rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, OnlinePrefillPosition{},
        AlibiScorePolicy{args.alibi_slopes}, rows,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_alibi_batch_ptrs(const GqaBatchPtrArgs& args) {
    const GqaGeometry& g = args.geometry;
    const OnlinePtrBf16Storage storage{
        args.kv.keys, args.kv.values, g.kv_heads};
    gqa_online_attention_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, OnlineBatchPositions{args.positions},
        AlibiScorePolicy{args.alibi_slopes}, args.rows,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_alibi_int8_batch_ptrs(const GqaBatchPtrInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const OnlinePtrInt8Storage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, g.kv_heads};
    gqa_online_attention_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, OnlineBatchPositions{args.positions},
        AlibiScorePolicy{args.alibi_slopes}, args.rows,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_alibi_paged_batch(const GqaPagedArgs& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const OnlinePagedBf16Storage storage{
        args.kv.keys, args.kv.values, index.page_tables,
        index.page_table_stride, index.attention_slot, index.page_tokens,
        index.page_vector_elements, index.layer_vector_offset, g.kv_heads};
    gqa_online_attention_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, OnlineBatchPositions{args.positions},
        AlibiScorePolicy{args.alibi_slopes}, args.rows,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_alibi_int8_paged_batch(const GqaPagedInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const PagedKvScaleIndex& scales = args.scale_index;
    const OnlinePagedInt8Storage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, index.page_tables, index.page_table_stride,
        index.attention_slot, index.page_tokens, index.page_vector_elements,
        index.layer_vector_offset, scales.page_scale_elements,
        scales.layer_scale_offset, g.kv_heads};
    gqa_online_attention_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, OnlineBatchPositions{args.positions},
        AlibiScorePolicy{args.alibi_slopes}, args.rows,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

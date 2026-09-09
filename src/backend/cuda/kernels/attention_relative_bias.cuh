
/** @brief Relative-position score policy for shared warp-online attention. */
struct RelativeBiasScorePolicy {
    const float* values;
    int bucket_count;
    int max_distance;
    bool bidirectional;

    __device__ __forceinline__ float score(
        float dot, float scale, int head, int query_position,
        int key_position) const {
        const int bucket = attention_semantics::relative_position_bucket(
            query_position, key_position, bucket_count,
            max_distance, bidirectional);
        return dot * scale + values[static_cast<size_t>(head) *
            static_cast<size_t>(bucket_count) + static_cast<size_t>(bucket)];
    }
};

void launch_gqa_decode_relative_device(const GqaContiguousArgs& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const ContiguousBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, g.kv_heads};
    gqa_online_attention_kernel<<<g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out,
        AttentionSinglePosition{args.extent.position},
        RelativeBiasScorePolicy{
            bias.values, bias.bucket_count, bias.max_distance,
            bias.bidirectional},
        1, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_relative(const GqaContiguousArgs& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const int rows = args.extent.rows;
    const ContiguousBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, g.kv_heads};
    gqa_online_attention_kernel<<<rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, AttentionPrefillPosition{},
        RelativeBiasScorePolicy{
            bias.values, bias.bucket_count, bias.max_distance,
            bias.bidirectional},
        rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_int8_device(const GqaContiguousInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const ContiguousInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, g.kv_heads};
    gqa_online_attention_kernel<<<g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out,
        AttentionSinglePosition{args.extent.position},
        RelativeBiasScorePolicy{
            bias.values, bias.bucket_count, bias.max_distance,
            bias.bidirectional},
        1, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_relative_int8(const GqaContiguousInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const int rows = args.extent.rows;
    const ContiguousInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, g.kv_heads};
    gqa_online_attention_kernel<<<rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, AttentionPrefillPosition{},
        RelativeBiasScorePolicy{
            bias.values, bias.bucket_count, bias.max_distance,
            bias.bidirectional},
        rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_batch_ptrs(const GqaBatchPtrArgs& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const PointerBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, g.kv_heads};
    gqa_online_attention_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, AttentionBatchPositions{args.positions},
        RelativeBiasScorePolicy{
            bias.values, bias.bucket_count, bias.max_distance,
            bias.bidirectional},
        args.rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_int8_batch_ptrs(const GqaBatchPtrInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const PointerInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, g.kv_heads};
    gqa_online_attention_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, AttentionBatchPositions{args.positions},
        RelativeBiasScorePolicy{
            bias.values, bias.bucket_count, bias.max_distance,
            bias.bidirectional},
        args.rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_paged_batch(const GqaPagedArgs& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const auto& bias = args.relative_bias;
    const PagedBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, index.page_tables,
        index.page_table_stride, index.attention_slot, index.page_tokens,
        index.page_vector_elements, index.layer_vector_offset, g.kv_heads};
    gqa_online_attention_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, AttentionBatchPositions{args.positions},
        RelativeBiasScorePolicy{
            bias.values, bias.bucket_count, bias.max_distance,
            bias.bidirectional},
        args.rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_int8_paged_batch(const GqaPagedInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const PagedKvScaleIndex& scales = args.scale_index;
    const auto& bias = args.relative_bias;
    const PagedInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, index.page_tables, index.page_table_stride,
        index.attention_slot, index.page_tokens, index.page_vector_elements,
        index.layer_vector_offset, scales.page_scale_elements,
        scales.layer_scale_offset, g.kv_heads};
    gqa_online_attention_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, storage, args.out, AttentionBatchPositions{args.positions},
        RelativeBiasScorePolicy{
            bias.values, bias.bucket_count, bias.max_distance,
            bias.bidirectional},
        args.rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
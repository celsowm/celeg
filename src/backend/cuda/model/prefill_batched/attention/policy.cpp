namespace celeg::prefill_detail {

AttentionCapability select_attention_plan(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    const AttentionSpec& owner_layout,
    int rows) {
    AttentionRequest request;
    request.kv_format = model.resources_.options().kv_cache_mode;
    request.operation = AttentionOperation::Prefill;
    request.layout = AttentionKvLayout::Contiguous;
    request.position_source = AttentionPositionSource::HostScalar;
    request.bias = attention.alibi_slopes.data()
        ? AttentionPositionBias::Alibi : AttentionPositionBias::None;
    request.fast_attention = model.resources_.options().fast_attention;
    request.flash_attention_requested = model.resources_.options().flash_attn;
    request.head_dim = owner_layout.head_dim;
    request.rows = rows;
    return require_attention_capability(request);
}

void store_and_attend(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    const AttentionCapability& plan,
    int rows) {
    const AttentionSpec& layout = attention.layout;
    const AttentionSpec& owner_layout = owner.layout;
    const bool int8_kv = plan.kv_format == KvCacheMode::Int8;

    if (attention.key && attention.value) {
        if (int8_kv) {
            launch_store_kv_int8_prefill(
                model.workspace_.prefill_k_.data(),
                model.workspace_.prefill_v_.data(),
                owner.key_cache_int8_ptr(), owner.value_cache_int8_ptr(),
                owner.key_cache_scales_ptr(), owner.value_cache_scales_ptr(),
                rows, owner_layout.key_value_heads, owner_layout.head_dim,
                model.stream_.get());
        } else {
            launch_store_kv_prefill(
                model.workspace_.prefill_k_.data(),
                model.workspace_.prefill_v_.data(),
                owner.key_cache_bf16(), owner.value_cache_bf16(),
                rows, owner_layout.key_value_width(), model.stream_.get());
        }
    }

    const GqaGeometry geometry{
        .q_heads = layout.query_heads,
        .kv_heads = owner_layout.key_value_heads,
        .head_dim = owner_layout.head_dim,
        .sliding_window = layout.sliding_window_size()};
    const AttentionExtent extent{.rows = rows};
    const Bf16KvView bf16_kv{
        .keys = owner.key_cache_bf16(),
        .values = owner.value_cache_bf16()};
    const Int8KvView int8_kv_view{
        .keys = owner.key_cache_int8_ptr(),
        .values = owner.value_cache_int8_ptr(),
        .key_scales = owner.key_cache_scales_ptr(),
        .value_scales = owner.value_cache_scales_ptr()};
    const AttentionRowStrides strides{
        .q_width = layout.query_width(),
        .kv_width = owner_layout.key_value_width(),
        .out_width = layout.query_width()};

    switch (plan.algorithm) {
    case AttentionAlgorithm::Alibi:
        if (int8_kv) {
            launch_gqa_prefill_alibi_int8({
                .query = model.workspace_.prefill_q_.data(),
                .kv = int8_kv_view,
                .out = model.workspace_.prefill_op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .alibi_slopes = attention.alibi_slopes.data(),
                .stream = model.stream_.get()});
        } else {
            launch_gqa_prefill_alibi({
                .query = model.workspace_.prefill_q_.data(),
                .kv = bf16_kv,
                .out = model.workspace_.prefill_op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .alibi_slopes = attention.alibi_slopes.data(),
                .stream = model.stream_.get()});
        }
        break;
    case AttentionAlgorithm::Online:
        launch_gqa_prefill_online_int8({
            .query = model.workspace_.prefill_q_.data(),
            .kv = int8_kv_view,
            .out = model.workspace_.prefill_op_output_.data(),
            .geometry = geometry,
            .extent = extent,
            .stream = model.stream_.get()});
        break;
    case AttentionAlgorithm::Flash:
        launch_gqa_prefill_flash({
            .query = model.workspace_.prefill_q_.data(),
            .kv = bf16_kv,
            .out = model.workspace_.prefill_op_output_.data(),
            .geometry = geometry,
            .strides = strides,
            .rows = rows,
            .stream = model.stream_.get()});
        break;
    case AttentionAlgorithm::Gemm:
        launch_gqa_prefill_gemm({
            .cublas = model.gemm_->cublas().get(),
            .query = model.workspace_.prefill_q_.data(),
            .kv = bf16_kv,
            .out = model.workspace_.prefill_op_output_.data(),
            .scores_scratch = model.workspace_.prefill_attn_scores_.data(),
            .probs_scratch = model.workspace_.prefill_attn_probs_.data(),
            .geometry = geometry,
            .strides = strides,
            .rows = rows,
            .stream = model.stream_.get()});
        break;
    case AttentionAlgorithm::Segmented: {
        const int chunks =
            (rows + kAttentionPrefillChunkTokens - 1) /
            kAttentionPrefillChunkTokens;
        launch_gqa_prefill_segmented({
            .query = model.workspace_.prefill_q_.data(),
            .kv = bf16_kv,
            .out = model.workspace_.prefill_op_output_.data(),
            .geometry = geometry,
            .extent = extent,
            .segmentation = {
                .chunk_tokens = kAttentionPrefillChunkTokens,
                .chunks = chunks,
                .partial_max = model.workspace_.prefill_attn_partial_max_.data(),
                .partial_denom = model.workspace_.prefill_attn_partial_denom_.data(),
                .partial_accum = model.workspace_.prefill_attn_partial_accum_.data()},
            .stream = model.stream_.get()});
        break;
    }
    case AttentionAlgorithm::Strict:
        if (int8_kv) {
            launch_gqa_prefill_strict_int8({
                .query = model.workspace_.prefill_q_.data(),
                .kv = int8_kv_view,
                .out = model.workspace_.prefill_op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = model.stream_.get()});
        } else {
            launch_gqa_prefill_strict({
                .query = model.workspace_.prefill_q_.data(),
                .kv = bf16_kv,
                .out = model.workspace_.prefill_op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = model.stream_.get()});
        }
        break;
    }
}

}

namespace celeg {

AttentionCapability CudaCompiledModel::prefill_attention_plan(
    AttentionLayer& attention, const AttentionSpec& owner_layout, int rows) {
    // The flash request is an execution preference, not a capability:
    // CELEG_FLASH_ATTN is resolved once into CudaModelOptions::flash_attn at
    // model-configuration construction time (see runtime_types.hpp); execution
    // code just reads the resolved option and lets the capability policy decide
    // whether it can be honoured for this head dimension.
    AttentionRequest request;
    request.kv_format = resources_.options_.kv_cache_mode;
    request.operation = AttentionOperation::Prefill;
    request.layout = AttentionKvLayout::Contiguous;
    request.position_source = AttentionPositionSource::HostScalar;
    request.bias = attention.alibi_slopes.data()
        ? AttentionPositionBias::Alibi : AttentionPositionBias::None;
    request.fast_attention = resources_.options_.fast_attention;
    request.flash_attention_requested = resources_.options_.flash_attn;
    request.head_dim = owner_layout.head_dim;
    request.rows = rows;
    return require_attention_capability(request);
}

void CudaCompiledModel::store_and_attend_prefill(
    AttentionLayer& attention, AttentionLayer& owner,
    const AttentionCapability& plan, int rows) {
    const AttentionSpec& layout = attention.layout;
    const AttentionSpec& owner_layout = owner.layout;
    const bool int8_kv = plan.kv_format == KvCacheMode::Int8;
    if (attention.key && attention.value) {
        if (int8_kv) {
            launch_store_kv_int8_prefill(
                workspace_.prefill_k_.data(), workspace_.prefill_v_.data(),
                owner.key_cache_int8.data(), owner.value_cache_int8.data(),
                owner.key_cache_scales.data(), owner.value_cache_scales.data(),
                rows, owner_layout.key_value_heads, owner_layout.head_dim,
                stream_.get());
        } else {
            launch_store_kv_prefill(
                workspace_.prefill_k_.data(), workspace_.prefill_v_.data(),
                owner.key_cache.data(), owner.value_cache.data(),
                rows, owner_layout.key_value_width(), stream_.get());
        }
    }
    const GqaGeometry geometry{
        .q_heads = layout.query_heads,
        .kv_heads = owner_layout.key_value_heads,
        .head_dim = owner_layout.head_dim,
        .sliding_window = layout.sliding_window_size()};
    const AttentionExtent extent{.rows = rows};
    const Bf16KvView bf16_kv{.keys = owner.key_cache.data(),
                             .values = owner.value_cache.data()};
    const Int8KvView int8_kv_view{
        .keys = owner.key_cache_int8.data(),
        .values = owner.value_cache_int8.data(),
        .key_scales = owner.key_cache_scales.data(),
        .value_scales = owner.value_cache_scales.data()};
    const AttentionRowStrides strides{
        .q_width = layout.query_width(),
        .kv_width = owner_layout.key_value_width(),
        .out_width = layout.query_width()};
    switch (plan.algorithm) {
    case AttentionAlgorithm::Alibi:
        if (int8_kv) {
            launch_gqa_prefill_alibi_int8({
                .query = workspace_.prefill_q_.data(),
                .kv = int8_kv_view,
                .out = workspace_.prefill_op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .alibi_slopes = attention.alibi_slopes.data(),
                .stream = stream_.get()});
        } else {
            launch_gqa_prefill_alibi({
                .query = workspace_.prefill_q_.data(),
                .kv = bf16_kv,
                .out = workspace_.prefill_op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .alibi_slopes = attention.alibi_slopes.data(),
                .stream = stream_.get()});
        }
        break;
    case AttentionAlgorithm::Online:
        launch_gqa_prefill_online_int8({
            .query = workspace_.prefill_q_.data(),
            .kv = int8_kv_view,
            .out = workspace_.prefill_op_output_.data(),
            .geometry = geometry,
            .extent = extent,
            .stream = stream_.get()});
        break;
    case AttentionAlgorithm::Flash:
        launch_gqa_prefill_flash({
            .query = workspace_.prefill_q_.data(),
            .kv = bf16_kv,
            .out = workspace_.prefill_op_output_.data(),
            .geometry = geometry,
            .strides = strides,
            .rows = rows,
            .stream = stream_.get()});
        break;
    case AttentionAlgorithm::Gemm:
        launch_gqa_prefill_gemm({
            .cublas = gemm_->cublas().get(),
            .query = workspace_.prefill_q_.data(),
            .kv = bf16_kv,
            .out = workspace_.prefill_op_output_.data(),
            .scores_scratch = workspace_.prefill_attn_scores_.data(),
            .probs_scratch = workspace_.prefill_attn_probs_.data(),
            .geometry = geometry,
            .strides = strides,
            .rows = rows,
            .stream = stream_.get()});
        break;
    case AttentionAlgorithm::Segmented: {
        const int chunks = (rows + kPrefillAttnChunkTokens - 1) /
            kPrefillAttnChunkTokens;
        launch_gqa_prefill_segmented({
            .query = workspace_.prefill_q_.data(),
            .kv = bf16_kv,
            .out = workspace_.prefill_op_output_.data(),
            .geometry = geometry,
            .extent = extent,
            .segmentation = {
                .chunk_tokens = kPrefillAttnChunkTokens,
                .chunks = chunks,
                .partial_max = workspace_.prefill_attn_partial_max_.data(),
                .partial_denom = workspace_.prefill_attn_partial_denom_.data(),
                .partial_accum = workspace_.prefill_attn_partial_accum_.data()},
            .stream = stream_.get()});
        break;
    }
    case AttentionAlgorithm::Strict:
        if (int8_kv) {
            launch_gqa_prefill_strict_int8({
                .query = workspace_.prefill_q_.data(),
                .kv = int8_kv_view,
                .out = workspace_.prefill_op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = stream_.get()});
        } else {
            launch_gqa_prefill_strict({
                .query = workspace_.prefill_q_.data(),
                .kv = bf16_kv,
                .out = workspace_.prefill_op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = stream_.get()});
        }
        break;
    }
}

void CudaCompiledModel::run_prefill_latent_attention(
    AttentionLayer& attention, AttentionLayer& owner, LayerCommon& common_layer,
    const CompiledLayerProgram& semantics, int rows) {
    auto& prof = prefill_phase_profile();
    const AttentionSpec& layout = attention.layout;
    if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument(
            "CUDA latent attention requires BF16 state storage");
    }
    const auto& latent = *layout.latent_state();
    // NOTE: pre-existing control flow, preserved verbatim. The factorized
    // branch intentionally falls through to the non-factorized path below.
    if (latent.factorized) {
        prof.begin(stream_.get());
        {
        auto native_fanout = native_fanout_scope(
            workspace_.prefill_normed_.data(), rows, resources_.program_.hidden);
        linear(workspace_.prefill_normed_.data(), *attention.latent_query_projection,
               workspace_.prefill_latent_projection_.data(), rows,
               latent.query_rank, resources_.program_.hidden);
        launch_rmsnorm(workspace_.prefill_latent_projection_.data(),
                       attention.latent_query_norm,
                       workspace_.prefill_latent_projection_.data(), rows,
                       latent.query_rank, latent.query_latent_norm.epsilon,
                       stream_.get());
        linear(workspace_.prefill_latent_projection_.data(),
               *attention.latent_query_expansion,
               workspace_.prefill_qkv_.data(), rows,
               layout.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
               latent.query_rank);
        launch_factorized_latent_query({
            .query_projection = workspace_.prefill_qkv_.data(),
            .expansion = attention.latent_expansion->bf16,
            .query_content = workspace_.prefill_latent_query_content_.data(),
            .rows = rows,
            .query_heads = layout.query_heads,
            .query_nope = latent.nope_head_dim,
            .query_rope_dim = latent.rope_head_dim,
            .latent_rank = latent.latent_rank,
            .stream = stream_.get()});
        launch_factorized_latent_rope({
            .query_projection = workspace_.prefill_qkv_.data(),
            .query_rope = workspace_.prefill_latent_query_rope_.data(),
            .rows = rows,
            .query_heads = layout.query_heads,
            .query_nope = latent.nope_head_dim,
            .query_rope_dim = latent.rope_head_dim,
            .stream = stream_.get()});
        linear(workspace_.prefill_normed_.data(), *attention.latent_key_projection,
               workspace_.prefill_qkv_.data(), rows,
               latent.latent_rank + latent.rope_head_dim,
               resources_.program_.hidden);
        launch_rmsnorm(workspace_.prefill_qkv_.data(), attention.latent_key_norm,
                       workspace_.prefill_latent_key_.data(), rows,
                       latent.latent_rank, latent.key_latent_norm.epsilon,
                       stream_.get());
        CELEG_CUDA(cudaMemcpyAsync(
            workspace_.prefill_latent_value_.data(),
            workspace_.prefill_latent_key_.data(),
            static_cast<size_t>(rows) * latent.latent_rank *
                sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_.get()));
        CELEG_CUDA(cudaMemcpy2DAsync(
            workspace_.prefill_latent_key_rope_.data(),
            static_cast<size_t>(latent.rope_head_dim) * sizeof(__nv_bfloat16),
            workspace_.prefill_qkv_.data() + latent.latent_rank,
            static_cast<size_t>(latent.latent_rank + latent.rope_head_dim) *
                sizeof(__nv_bfloat16),
            static_cast<size_t>(latent.rope_head_dim) * sizeof(__nv_bfloat16),
            static_cast<size_t>(rows), cudaMemcpyDeviceToDevice, stream_.get()));
        }
        prof.end(PrefillPhase::QkvProj, stream_.get());
        prof.begin(stream_.get());
        if (const auto* rope = layout.rope_position()) {
            launch_qk_norm_rope_positions(
                workspace_.prefill_latent_query_rope_.data(),
                workspace_.prefill_latent_key_rope_.data(), nullptr, nullptr,
                rows, layout.query_heads, 1, latent.rope_head_dim, nullptr,
                static_cast<float>(rope->theta), 1.0f,
                layout.query_norm.epsilon, false,
                rope->pairing, lower_cuda_rope_scaling(*rope), stream_.get());
        }
        prof.end(PrefillPhase::RopeKv, stream_.get());
        prof.begin(stream_.get());
        launch_store_latent_prefill(
            workspace_.prefill_latent_key_.data(),
            workspace_.prefill_latent_value_.data(),
            workspace_.prefill_latent_key_rope_.data(),
            owner.latent_key_cache.data(), owner.latent_value_cache.data(),
            owner.latent_key_rope_cache.data(), rows, latent.latent_rank,
            latent.rope_head_dim, stream_.get());
        const float score_scale = layout.query_scale;
        launch_latent_attention_prefill({
            .query = {.content = workspace_.prefill_latent_query_content_.data(),
                      .rope = workspace_.prefill_latent_query_rope_.data()},
            .kv = {.keys = owner.latent_key_cache.data(),
                   .values = owner.latent_value_cache.data(),
                   .key_rope = owner.latent_key_rope_cache.data()},
            .out = workspace_.prefill_op_output_.data(),
            .extent = {.rows = rows},
            .alibi_slopes = attention.alibi_slopes.data(),
            .geometry = {.query_heads = layout.query_heads,
                         .latent_rank = latent.latent_rank,
                         .rotary_width = latent.rope_head_dim,
                         .score_scale = score_scale,
                         .sliding_window = layout.sliding_window_size()},
            .stream = stream_.get()});
        prof.end(PrefillPhase::Attention, stream_.get());
        prof.begin(stream_.get());
        launch_factorized_latent_value({
            .latent_output = workspace_.prefill_op_output_.data(),
            .expansion = attention.latent_expansion->bf16,
            .value_output = workspace_.prefill_latent_decompressed_.data(),
            .rows = rows,
            .query_heads = layout.query_heads,
            .query_nope = latent.nope_head_dim,
            .value_dim = latent.value_head_dim,
            .latent_rank = latent.latent_rank,
            .stream = stream_.get()});
        linear(workspace_.prefill_normed_.data(), *attention.gate,
               workspace_.prefill_attention_gate_.data(), rows,
               layout.output_gate_width(), resources_.program_.hidden);
        if (layout.output_gate.granularity == AttentionGateGranularity::HeadWise) {
            launch_sigmoid_multiply_headwise(
                workspace_.prefill_latent_decompressed_.data(),
                workspace_.prefill_attention_gate_.data(),
                rows, layout.query_heads, latent.value_head_dim, stream_.get());
        } else {
            launch_sigmoid_multiply(
                workspace_.prefill_latent_decompressed_.data(),
                workspace_.prefill_attention_gate_.data(),
                rows * layout.latent_output_width(), stream_.get());
        }
        linear(workspace_.prefill_latent_decompressed_.data(), *attention.out,
               workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
               layout.latent_output_width(),
               resources_.options_.fused_residuals && !common_layer.post_attention_norm
                   ? 1.0f : 0.0f);
        launch_scale(workspace_.prefill_hidden_.data(),
                     rows * resources_.program_.hidden,
                     semantics.residual.multiplier,
                     stream_.get());
        prof.end(PrefillPhase::AttnOut, stream_.get());
    } else if (layout.output_gate.enabled() || layout.multi_axis_position()) {
        throw std::invalid_argument(
            "CUDA latent attention does not support query gates or M-RoPE yet");
    }
    prof.begin(stream_.get());
    {
    auto native_fanout = native_fanout_scope(
        workspace_.prefill_normed_.data(), rows, resources_.program_.hidden);
    linear(workspace_.prefill_normed_.data(), *attention.latent_query,
           workspace_.prefill_latent_query_content_.data(), rows,
           layout.latent_query_content_width(), resources_.program_.hidden);
    if (layout.latent_query_rope_width() != 0) {
        linear(workspace_.prefill_normed_.data(), *attention.latent_query_rope,
               workspace_.prefill_latent_query_rope_.data(), rows,
               layout.latent_query_rope_width(), resources_.program_.hidden);
    }
    if (attention.latent_key && attention.latent_value) {
        linear(workspace_.prefill_normed_.data(), *attention.latent_key,
               workspace_.prefill_latent_key_.data(), rows,
               latent.latent_rank, resources_.program_.hidden);
        linear(workspace_.prefill_normed_.data(), *attention.latent_value,
               workspace_.prefill_latent_value_.data(), rows,
               latent.latent_rank, resources_.program_.hidden);
        if (attention.latent_key_rope && latent.decoupled_rope &&
            latent.rope_head_dim != 0) {
            linear(workspace_.prefill_normed_.data(), *attention.latent_key_rope,
                   workspace_.prefill_latent_key_rope_.data(), rows,
                   latent.rope_head_dim, resources_.program_.hidden);
        }
    }
    }
    prof.end(PrefillPhase::QkvProj, stream_.get());
    prof.begin(stream_.get());
    if (const auto* rope = layout.rope_position();
        rope && attention.latent_key_rope && latent.decoupled_rope &&
        latent.rope_head_dim != 0) {
        launch_dynamic_qk_norm_rope_prefill(
            workspace_.prefill_latent_query_rope_.data(),
            attention.latent_key ? workspace_.prefill_latent_key_rope_.data() : nullptr,
            nullptr, nullptr, rows, layout.query_heads, 1,
            latent.rope_head_dim, static_cast<float>(rope->theta), 1.0f,
            layout.query_norm.epsilon, false,
            lower_cuda_rope_scaling(*rope), stream_.get());
    }
    prof.end(PrefillPhase::RopeKv, stream_.get());
    prof.begin(stream_.get());
    if (attention.latent_key && attention.latent_value) {
        launch_store_latent_prefill(
            workspace_.prefill_latent_key_.data(),
            workspace_.prefill_latent_value_.data(),
            attention.latent_key_rope && latent.decoupled_rope &&
            latent.rope_head_dim != 0
                ? workspace_.prefill_latent_key_rope_.data() : nullptr,
            owner.latent_key_cache.data(), owner.latent_value_cache.data(),
            owner.latent_key_rope_cache.data(), rows, latent.latent_rank,
            latent.decoupled_rope ? latent.rope_head_dim : 0, stream_.get());
    }
    const float score_scale = layout.query_scale;
    launch_latent_attention_prefill({
        .query = {.content = workspace_.prefill_latent_query_content_.data(),
                  .rope = layout.latent_query_rope_width() != 0
                              ? workspace_.prefill_latent_query_rope_.data()
                              : nullptr},
        .kv = {.keys = owner.latent_key_cache.data(),
               .values = owner.latent_value_cache.data(),
               .key_rope = owner.latent_key_rope_cache.data()},
        .out = workspace_.prefill_op_output_.data(),
        .extent = {.rows = rows},
        .alibi_slopes = attention.alibi_slopes.data(),
        .geometry = {.query_heads = layout.query_heads,
                     .latent_rank = latent.latent_rank,
                     .rotary_width = latent.decoupled_rope ? latent.rope_head_dim : 0,
                     .score_scale = score_scale,
                     .sliding_window = layout.sliding_window_size()},
        .stream = stream_.get()});
    prof.end(PrefillPhase::Attention, stream_.get());
    prof.begin(stream_.get());
    linear(workspace_.prefill_op_output_.data(), *attention.out,
           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
           layout.latent_query_content_width(),
           resources_.options_.fused_residuals && !common_layer.post_attention_norm
               ? 1.0f : 0.0f);
    launch_scale(workspace_.prefill_hidden_.data(),
                 rows * resources_.program_.hidden,
                 semantics.residual.multiplier,
                 stream_.get());
    prof.end(PrefillPhase::AttnOut, stream_.get());
}

void CudaCompiledModel::run_prefill_attention(
    AttentionLayer& attention, LayerCommon& common_layer,
    const CompiledLayerProgram& semantics, int rows) {
    auto& prof = prefill_phase_profile();
    const AttentionSpec& layout = attention.layout;
    AttentionLayer* owner = &attention;
    if (attention.kv_owner_layer >= 0) {
        owner = as_attention(resources_.layers_.at(
            static_cast<size_t>(attention.kv_owner_layer)));
        if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
    }
    if (layout.uses_latent_state()) {
        run_prefill_latent_attention(attention, *owner, common_layer, semantics, rows);
        return;
    }
    const AttentionSpec& owner_layout = owner->layout;

    const int query_projection_width = attention.query->rows;
    const bool output_gate = layout.output_gate.enabled();
    const bool gate_packed = output_gate && layout.output_gate.packed_with_query;
    prof.begin(stream_.get());
    {
    auto native_fanout = native_fanout_scope(
        workspace_.prefill_normed_.data(), rows, resources_.program_.hidden);
    linear(workspace_.prefill_normed_.data(), *attention.query,
           gate_packed ? workspace_.prefill_qkv_.data() : workspace_.prefill_q_.data(),
           rows, query_projection_width,
           resources_.program_.hidden);
    if (attention.key && attention.value) {
        linear(workspace_.prefill_normed_.data(), *attention.key,
               workspace_.prefill_k_.data(), rows, layout.key_value_width(),
               resources_.program_.hidden);
        linear(workspace_.prefill_normed_.data(), *attention.value,
               workspace_.prefill_v_.data(), rows, layout.key_value_width(),
               resources_.program_.hidden);
    }
    }
    prof.end(PrefillPhase::QkvProj, stream_.get());
    prof.begin(stream_.get());
    if (output_gate) {
        if (gate_packed) {
            launch_extract_attention_output_gate(workspace_.prefill_qkv_.data(),
                                      workspace_.prefill_q_.data(),
                                      workspace_.prefill_attention_gate_.data(),
                                      rows, layout.query_width(), stream_.get());
        } else {
            linear(workspace_.prefill_normed_.data(), *attention.gate,
                   workspace_.prefill_attention_gate_.data(), rows,
                   layout.query_width(), resources_.program_.hidden);
        }
    }
    if (const auto* rope = layout.rope_position()) {
        launch_dynamic_qk_norm_rope_prefill(
            workspace_.prefill_q_.data(), attention.key ? workspace_.prefill_k_.data() : nullptr,
            attention.q_norm, attention.k_norm, rows, layout.query_heads,
            layout.key_value_heads, layout.head_dim, static_cast<float>(rope->theta),
            static_cast<float>(rope->rotary_fraction), layout.query_norm.epsilon,
            layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream_.get());
    } else if (layout.has_query_key_norm()) {
        launch_dynamic_qk_norm_rope_prefill(
            workspace_.prefill_q_.data(), attention.key ? workspace_.prefill_k_.data() : nullptr,
            attention.q_norm, attention.k_norm, rows, layout.query_heads,
            layout.key_value_heads, layout.head_dim, 1.0f, 0.0f,
            layout.query_norm.epsilon, true, CudaRopeScaling{}, stream_.get());
    }
    launch_scale(workspace_.prefill_q_.data(),
                 static_cast<size_t>(rows) * layout.query_width(),
                 layout.query_scale, stream_.get());
    prof.end(PrefillPhase::RopeKv, stream_.get());

    prof.begin(stream_.get());
    const AttentionCapability plan =
        prefill_attention_plan(attention, owner_layout, rows);
    store_and_attend_prefill(attention, *owner, plan, rows);
    prof.end(PrefillPhase::Attention, stream_.get());
    if (output_gate) {
        launch_sigmoid_multiply(workspace_.prefill_op_output_.data(),
            workspace_.prefill_attention_gate_.data(),
            rows * layout.query_width(), stream_.get());
    }

    prof.begin(stream_.get());
    linear(workspace_.prefill_op_output_.data(), *attention.out,
           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
           layout.query_width(),
           resources_.options_.fused_residuals && !common_layer.post_attention_norm ? 1.0f : 0.0f);
    launch_scale(workspace_.prefill_hidden_.data(), rows * resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
    prof.end(PrefillPhase::AttnOut, stream_.get());
}

} // namespace celeg

#include "celeg/backend/cuda/packed_operators.hpp"

#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/kernels/gated_delta.hpp"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/paged_kv.hpp"

#include <stdexcept>
#include <string>

namespace celeg {

void PackedOperatorContext::linear(const __nv_bfloat16* input,
                                   const LinearWeight& weight,
                                   __nv_bfloat16* output,
                                   int rows,
                                   int output_width,
                                   int input_width,
                                   float beta) const {
    if (weight.rows != output_width || weight.cols != input_width) {
        throw std::runtime_error("packed linear shape mismatch");
    }
    gemm.linear(input, weight, output, rows, output_width, input_width, beta, plan);
}

void PackedConvolutionExecutor::run(
    PackedOperatorContext& context,
    const PackedSessionContext& reference,
    const ConvolutionLayer& convolution,
    int rows,
    int layer_index,
    int ragged_requests) {
    PackedWorkspace& w = context.workspace;
    context.linear(w.normed.data(), *convolution.conv_in, w.conv_projected.data(),
                   rows, 3 * context.shape.hidden, context.shape.hidden);
    const size_t offset = static_cast<size_t>(layer_index) * w.maximum_batch;
    if (ragged_requests != 0) {
        launch_conv_ragged_prefill(
            w.conv_projected.data(), convolution.conv_weight,
            w.d_conv_states.data() + offset, w.op_output.data(), w.positions.data(),
            w.d_span_offsets.data(), w.d_span_counts.data(), ragged_requests,
            context.shape.hidden, context.shape.conv_cache, w.stream.get());
    } else {
        launch_conv_decode_batch_ptrs(
            w.conv_projected.data(), convolution.conv_weight,
            w.d_conv_states.data() + offset, w.op_output.data(), w.positions.data(),
            rows, context.shape.hidden, context.shape.conv_cache, w.stream.get());
    }
    context.linear(w.op_output.data(), *convolution.conv_out, w.hidden.data(), rows,
                   context.shape.hidden, context.shape.hidden,
                   reference.options().fused_residuals ? 1.0f : 0.0f);
}

namespace {

void project_attention_qkv(PackedOperatorContext& context,
                           const PackedSessionContext& reference,
                           const AttentionLayer& attention,
                           int rows) {
    (void)reference;
    PackedWorkspace& w = context.workspace;
    const AttentionSpec& layout = attention.layout;
    context.linear(w.normed.data(), *attention.query, w.q.data(), rows,
                   layout.query_width(), context.shape.hidden);
    if (attention.key && attention.value) {
        context.linear(w.normed.data(), *attention.key, w.k.data(), rows,
                       layout.key_value_width(), context.shape.hidden);
        context.linear(w.normed.data(), *attention.value, w.v.data(), rows,
                       layout.key_value_width(), context.shape.hidden);
    }
    if (layout.positional_encoding == PositionalEncodingKind::Rope) {
        launch_dynamic_qk_norm_rope_device(
            w.q.data(), attention.key ? w.k.data() : nullptr, attention.q_norm,
            attention.k_norm, layout.query_heads, layout.key_value_heads,
            layout.head_dim, w.positions.data(), static_cast<float>(layout.rope_theta),
            static_cast<float>(layout.rotary_fraction), context.shape.numerical_policy.norm_eps,
            layout.query_key_norm, w.stream.get());
    }
    launch_scale(w.q.data(), static_cast<size_t>(rows) * layout.query_width(),
                 layout.query_scale, w.stream.get());
}

void run_paged_attention_cache(PackedOperatorContext& context,
                               const PackedSessionContext& reference,
                               int rows,
                               int layer_index,
                               bool segmented_attention,
                               int segmented_chunks) {
    PackedWorkspace& w = context.workspace;
    const AttentionLayer& current = *as_attention(reference.layers().at(
        static_cast<size_t>(layer_index)));
    const int cache_layer = current.kv_owner_layer >= 0
        ? current.kv_owner_layer : layer_index;
    const AttentionLayer& owner = *as_attention(reference.layers().at(
        static_cast<size_t>(cache_layer)));
    const AttentionSpec& layout = current.layout;
    const AttentionSpec& owner_layout = owner.layout;
    const int slot = w.paged_kv ? w.paged_kv->attention_slot(cache_layer) : -1;
    const int stride = w.paged_kv->max_pages_per_request();
    if (reference.options().kv_cache_mode == KvCacheMode::Int8) {
        if (current.key && current.value) launch_store_kv_int8_paged_batch(
            w.k.data(), w.v.data(), w.paged_kv->key_int8(), w.paged_kv->value_int8(),
            w.paged_kv->key_scales(), w.paged_kv->value_scales(), w.d_page_tables.data(),
            stride, w.positions.data(), rows, slot, w.paged_kv->page_tokens(),
            w.paged_kv->page_vector_elements(), w.paged_kv->layer_vector_offset(slot),
            w.paged_kv->page_scale_elements(), w.paged_kv->layer_scale_offset(slot),
            owner_layout.key_value_heads, owner_layout.head_dim, w.stream.get());
        if (segmented_attention) {
            launch_gqa_decode_int8_paged_segmented_batch(
                w.q.data(), w.paged_kv->key_int8(), w.paged_kv->value_int8(),
                w.paged_kv->key_scales(), w.paged_kv->value_scales(),
                w.d_page_tables.data(), stride, w.op_output.data(), w.positions.data(),
                rows, slot, w.paged_kv->page_tokens(), w.paged_kv->page_vector_elements(),
                w.paged_kv->layer_vector_offset(slot), w.paged_kv->page_scale_elements(),
                w.paged_kv->layer_scale_offset(slot), layout.query_heads,
                owner_layout.key_value_heads, owner_layout.head_dim,
                reference.options().attention_chunk_tokens, segmented_chunks,
                layout.sliding_window, w.segmented_partial_max.data(),
                w.segmented_partial_denom.data(), w.segmented_partial_accum.data(),
                w.stream.get());
        } else {
            launch_gqa_decode_int8_paged_batch(
                w.q.data(), w.paged_kv->key_int8(), w.paged_kv->value_int8(),
                w.paged_kv->key_scales(), w.paged_kv->value_scales(),
                w.d_page_tables.data(), stride, w.op_output.data(), w.positions.data(),
                rows, slot, w.paged_kv->page_tokens(), w.paged_kv->page_vector_elements(),
                w.paged_kv->layer_vector_offset(slot), w.paged_kv->page_scale_elements(),
                w.paged_kv->layer_scale_offset(slot), layout.query_heads,
                owner_layout.key_value_heads, owner_layout.head_dim,
                layout.sliding_window, reference.options().fast_attention, w.stream.get());
        }
        return;
    }
    if (current.key && current.value) launch_store_kv_paged_batch(
        w.k.data(), w.v.data(), w.paged_kv->key_bf16(), w.paged_kv->value_bf16(),
        w.d_page_tables.data(), stride, w.positions.data(), rows, slot,
        w.paged_kv->page_tokens(), w.paged_kv->page_vector_elements(),
        w.paged_kv->layer_vector_offset(slot), owner_layout.key_value_heads,
        owner_layout.head_dim, w.stream.get());
    if (segmented_attention) {
        launch_gqa_decode_paged_segmented_batch(
            w.q.data(), w.paged_kv->key_bf16(), w.paged_kv->value_bf16(),
            w.d_page_tables.data(), stride, w.op_output.data(), w.positions.data(),
            rows, slot, w.paged_kv->page_tokens(), w.paged_kv->page_vector_elements(),
            w.paged_kv->layer_vector_offset(slot), layout.query_heads,
            owner_layout.key_value_heads, owner_layout.head_dim,
            reference.options().attention_chunk_tokens, segmented_chunks,
            layout.sliding_window, w.segmented_partial_max.data(),
            w.segmented_partial_denom.data(), w.segmented_partial_accum.data(),
            w.stream.get());
    } else {
        launch_gqa_decode_paged_batch(
            w.q.data(), w.paged_kv->key_bf16(), w.paged_kv->value_bf16(),
            w.d_page_tables.data(), stride, w.op_output.data(), w.positions.data(),
            rows, slot, w.paged_kv->page_tokens(), w.paged_kv->page_vector_elements(),
            w.paged_kv->layer_vector_offset(slot), layout.query_heads,
            owner_layout.key_value_heads, owner_layout.head_dim,
            layout.sliding_window, reference.options().fast_attention, w.stream.get());
    }
}

void run_local_attention_cache(PackedOperatorContext& context,
                               const PackedSessionContext& reference,
                               int rows, int layer_index) {
    PackedWorkspace& w = context.workspace;
    const AttentionLayer& current = *as_attention(reference.layers().at(
        static_cast<size_t>(layer_index)));
    const int cache_layer = current.kv_owner_layer >= 0
        ? current.kv_owner_layer : layer_index;
    const AttentionLayer& owner = *as_attention(reference.layers().at(
        static_cast<size_t>(cache_layer)));
    const AttentionSpec& layout = current.layout;
    const AttentionSpec& owner_layout = owner.layout;
    const size_t offset = static_cast<size_t>(cache_layer) * w.maximum_batch;
    if (reference.options().kv_cache_mode == KvCacheMode::Int8) {
        if (current.key && current.value) launch_store_kv_int8_batch_ptrs(
            w.k.data(), w.v.data(), w.d_key_int8.data() + offset,
            w.d_value_int8.data() + offset, w.d_key_scales.data() + offset,
            w.d_value_scales.data() + offset, w.positions.data(), rows,
            owner_layout.key_value_heads, owner_layout.head_dim, w.stream.get());
        launch_gqa_decode_int8_batch_ptrs(
            w.q.data(), w.d_key_int8.data() + offset, w.d_value_int8.data() + offset,
            w.d_key_scales.data() + offset, w.d_value_scales.data() + offset,
            w.op_output.data(), w.positions.data(), rows, layout.query_heads,
            owner_layout.key_value_heads, owner_layout.head_dim, layout.sliding_window,
            reference.options().fast_attention, w.stream.get());
        return;
    }
    if (current.key && current.value) launch_store_kv_batch_ptrs(
        w.k.data(), w.v.data(), w.d_key_bf16.data() + offset,
        w.d_value_bf16.data() + offset, w.positions.data(), rows,
        owner_layout.key_value_width(), w.stream.get());
    launch_gqa_decode_batch_ptrs(
        w.q.data(), w.d_key_bf16.data() + offset, w.d_value_bf16.data() + offset,
        w.op_output.data(), w.positions.data(), rows, layout.query_heads,
        owner_layout.key_value_heads, owner_layout.head_dim, layout.sliding_window,
        reference.options().fast_attention, w.stream.get());
}

} // namespace

void PackedGatedDeltaNetExecutor::run(
    PackedOperatorContext& context,
    const PackedSessionContext& reference,
    const GatedDeltaNetLayer& gated_delta,
    int rows,
    int layer_index,
    const std::vector<PackedSessionContext>* batch_models,
    const std::vector<PackedPrefillRow>* row_descriptors) {
    if (!batch_models || batch_models->empty()) {
        throw std::invalid_argument("packed GatedDeltaNet execution needs session rows");
    }
    PackedWorkspace& w = context.workspace;
    const GatedDeltaNetSpec& spec = gated_delta.spec;
    const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
        spec.value_heads * spec.value_head_dim;
    const int value_width = spec.value_heads * spec.value_head_dim;
    context.linear(w.normed.data(), *gated_delta.qkv, w.gated_delta_qkv.data(),
                   rows, qkv_width, context.shape.hidden);
    context.linear(w.normed.data(), *gated_delta.z, w.gated_delta_z.data(),
                   rows, value_width, context.shape.hidden);
    context.linear(w.normed.data(), *gated_delta.b, w.gated_delta_b.data(),
                   rows, spec.value_heads, context.shape.hidden);
    context.linear(w.normed.data(), *gated_delta.a, w.gated_delta_a.data(),
                   rows, spec.value_heads, context.shape.hidden);

    size_t flat = 0;
    for (size_t request = 0; request < batch_models->size(); ++request) {
        const PackedSessionContext& session = batch_models->at(request);
        Layer& layer = session.layers().at(static_cast<size_t>(layer_index));
        GatedDeltaNetLayer* state_layer = as_gated_delta_net(layer);
        if (!state_layer) throw std::logic_error("packed GatedDeltaNet state binding mismatch");
        const size_t count = row_descriptors
            ? row_descriptors->at(request).token_count : 1;
        for (size_t token = 0; token < count; ++token, ++flat) {
            if (flat >= static_cast<size_t>(rows)) {
                throw std::invalid_argument("packed GatedDeltaNet row mapping exceeds input");
            }
            launch_gated_delta_net(
                w.gated_delta_qkv.data() + static_cast<size_t>(flat) * qkv_width,
                w.gated_delta_z.data() + static_cast<size_t>(flat) * value_width,
                w.gated_delta_b.data() + static_cast<size_t>(flat) * spec.value_heads,
                w.gated_delta_a.data() + static_cast<size_t>(flat) * spec.value_heads,
                gated_delta.conv_weight, gated_delta.dt_bias, gated_delta.a_log,
                gated_delta.norm, state_layer->conv_state.data(),
                state_layer->recurrent_state.data(),
                w.gated_delta_output.data() + static_cast<size_t>(flat) * value_width,
                1, spec.conv_kernel, spec.key_head_dim, spec.value_head_dim,
                spec.key_heads, spec.value_heads,
                context.shape.numerical_policy.norm_eps, w.stream.get());
        }
    }
    if (flat != static_cast<size_t>(rows)) {
        throw std::invalid_argument("packed GatedDeltaNet row mapping is incomplete");
    }
    context.linear(w.gated_delta_output.data(), *gated_delta.out, w.hidden.data(),
                   rows, context.shape.hidden, value_width,
                   reference.options().fused_residuals ? 1.0f : 0.0f);
}

void PackedAttentionExecutor::run(
    PackedOperatorContext& context,
    const PackedSessionContext& reference,
    const AttentionLayer& attention,
    int rows,
    int layer_index,
    bool segmented_attention,
    int segmented_chunks) {
    PackedWorkspace& w = context.workspace;
    project_attention_qkv(context, reference, attention, rows);
    if (w.paged_kv) {
        run_paged_attention_cache(context, reference, rows, layer_index,
                                  segmented_attention, segmented_chunks);
    } else {
        run_local_attention_cache(context, reference, rows, layer_index);
    }
    const int query_width = attention.layout.query_width();
    context.linear(w.op_output.data(), *attention.out, w.hidden.data(), rows,
                   context.shape.hidden, query_width,
                   reference.options().fused_residuals ? 1.0f : 0.0f);
    launch_scale(w.hidden.data(), rows * context.shape.hidden,
                 context.shape.numerical_policy.residual_multiplier, w.stream.get());
}

void PackedDenseFfnExecutor::run(
    PackedOperatorContext& context,
    const PackedSessionContext& reference,
    const LayerCommon& common_layer,
    int rows,
    int layer_index) {
    PackedWorkspace& w = context.workspace;
    launch_rmsnorm(w.hidden.data(), common_layer.ffn_norm, w.normed.data(),
                   rows, context.shape.hidden, context.shape.numerical_policy.norm_eps, w.stream.get());
    const auto& widths = context.shape.feed_forward_intermediates;
    const int intermediate = widths.empty()
        ? context.shape.intermediate
        : widths.at(static_cast<size_t>(layer_index));
    if (intermediate <= 0 || intermediate > context.shape.max_feed_forward_intermediate) {
        throw std::runtime_error("invalid packed dense FFN width at layer " +
                                 std::to_string(layer_index));
    }
    const auto* dense = as_dense_ffn(common_layer.feed_forward);
    if (!dense) throw std::logic_error("packed dense layer has no dense FFN binding");
    if (reference.options().fused_projections) {
        context.linear(w.normed.data(), *dense->w13, w.gate_up.data(), rows,
                       2 * intermediate, context.shape.hidden);
        launch_swiglu_interleaved(w.gate_up.data(), w.activated.data(), rows,
                                  intermediate, w.stream.get());
    } else {
        const auto w1 = slice_rows(*dense->w13, 0, intermediate);
        const auto w3 = slice_rows(*dense->w13, intermediate, intermediate);
        const size_t plane = static_cast<size_t>(rows) * intermediate;
        context.linear(w.normed.data(), w1, w.gate_up.data(), rows,
                       intermediate, context.shape.hidden);
        context.linear(w.normed.data(), w3, w.gate_up.data() + plane, rows,
                       intermediate, context.shape.hidden);
        launch_swiglu_fused(w.gate_up.data(), w.activated.data(),
                            static_cast<int>(plane), w.stream.get());
    }
    if (reference.options().fused_residuals) {
        context.linear(w.activated.data(), *dense->w2, w.hidden.data(), rows,
                       context.shape.hidden, intermediate, 1.0f);
    } else {
        context.linear(w.activated.data(), *dense->w2, w.mlp_output.data(), rows,
                       context.shape.hidden, intermediate);
        launch_scale(w.mlp_output.data(), rows * context.shape.hidden,
                     context.shape.numerical_policy.residual_multiplier, w.stream.get());
        launch_residual_add(w.hidden.data(), w.mlp_output.data(),
                            rows * context.shape.hidden, w.stream.get());
    }
}

void PackedMoeExecutor::run(
    PackedOperatorContext& context,
    const PackedSessionContext& reference,
    const LayerCommon& common_layer,
    int rows,
    const std::vector<PackedSessionContext>* batch_models,
    int layer_index) {
    PackedWorkspace& w = context.workspace;
    const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward);
    if (!moe) throw std::logic_error("packed MoE layer has no MoE FFN binding");
    launch_rmsnorm(w.hidden.data(), common_layer.ffn_norm, w.normed.data(),
                   rows, context.shape.hidden, context.shape.numerical_policy.norm_eps, w.stream.get());
    launch_cast_bf16_to_float(w.normed.data(), w.moe_hidden_float.data(),
                              rows * context.shape.hidden, w.stream.get());

    const MoeRouterConfig cfg = moe_router_config(context.shape);
    MoeRouterDevice router;
    router.router_weight = moe->router_float;
    router.expert_bias = moe->expert_bias;
    router.hidden_data = w.moe_hidden_float.data();
    router.selected_experts = w.moe_sel.data();
    router.routing_weights = w.moe_routing_w.data();
    router.rows = rows;
    router.hidden_dim = context.shape.hidden;
    launch_moe_router(router, cfg, w.moe_router_scratch.data(), w.stream.get());

    if (batch_models && !batch_models->empty() && layer_index >= 0) {
        const PackedSessionContext& session = batch_models->front();
        if (session.ensure_expert_residency) {
            session.ensure_expert_residency(
                session.owner, layer_index, w.moe_sel.data(), rows, w.stream.get(),
                moe->offloaded() ? w.moe_router_scratch.data() : nullptr);
        }
    }

    w.moe_output_accum.zero_async(w.stream.get());
    const MoeFfnDevice device = moe_ffn_device(*moe, context.shape);
    launch_moe_ffn(device, w.moe_sel.data(), w.moe_routing_w.data(),
                   w.normed.data(), w.moe_output_accum.data(), rows,
                   context.shape.experts_per_token, w.moe_gu_scratch.data(),
                   w.moe_act_scratch.data(), w.stream.get());
    launch_finalize_moe_output(w.moe_output_accum.data(), w.moe_output.data(),
                               rows * context.shape.hidden, w.stream.get());
    launch_residual_add(w.hidden.data(), w.moe_output.data(),
                        rows * context.shape.hidden, w.stream.get());
}

} // namespace celeg

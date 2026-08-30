#include "detail/compiled_model.hpp"
#include "backend/cuda/attention_norm.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

AttentionCapability CudaCompiledModel::token_attention_plan(
    AttentionLayer& attention, const AttentionSpec& owner_layout,
    const TokenKvPolicy& kv) {
    AttentionRequest request;
    request.kv_format = resources_.options_.kv_cache_mode;
    request.operation = AttentionOperation::Decode;
    request.layout = kv.kv_layout;
    request.position_source = kv.position_source;
    request.bias = attention.alibi_slopes.data()
        ? AttentionPositionBias::Alibi : AttentionPositionBias::None;
    request.fast_attention = resources_.options_.fast_attention;
    request.segmented_attention =
        kv.paged() && use_segmented_attention(session_.position_);
    request.head_dim = owner_layout.head_dim;
    return require_attention_capability(request);
}

void CudaCompiledModel::store_and_attend_token_contiguous(
    AttentionLayer& attention, AttentionLayer& owner, const AttentionCapability& plan,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v) {
    const AttentionSpec& layout = attention.layout;
    const AttentionSpec& owner_layout = owner.layout;
    const bool int8_kv = plan.kv_format == KvCacheMode::Int8;
    if (int8_kv) {
        launch_store_kv_int8(
            k, v, owner.key_cache_int8_ptr(), owner.value_cache_int8_ptr(),
            owner.key_cache_scales_ptr(), owner.value_cache_scales_ptr(),
            session_.position_, owner_layout.key_value_heads, owner_layout.head_dim,
            stream_.get());
    } else {
        launch_store_kv(k, v, owner.key_cache_bf16(), owner.value_cache_bf16(),
                        session_.position_, owner_layout.key_value_width(), stream_.get());
    }
    const GqaGeometry geometry{
        .q_heads = layout.query_heads,
        .kv_heads = owner_layout.key_value_heads,
        .head_dim = owner_layout.head_dim,
        .sliding_window = layout.sliding_window_size()};
    const AttentionExtent extent{.seq_len = session_.position_ + 1};
    switch (plan.algorithm) {
    case AttentionAlgorithm::Online:
        if (int8_kv) {
            launch_gqa_decode_online_int8({
                .query = q,
                .kv = {.keys = owner.key_cache_int8_ptr(),
                       .values = owner.value_cache_int8_ptr(),
                       .key_scales = owner.key_cache_scales_ptr(),
                       .value_scales = owner.value_cache_scales_ptr()},
                .out = workspace_.op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = stream_.get()});
        } else {
            launch_gqa_decode_online({
                .query = q,
                .kv = {.keys = owner.key_cache_bf16(),
                       .values = owner.value_cache_bf16()},
                .out = workspace_.op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = stream_.get()});
        }
        break;
    case AttentionAlgorithm::Strict:
        if (int8_kv) {
            launch_gqa_decode_strict_int8({
                .query = q,
                .kv = {.keys = owner.key_cache_int8_ptr(),
                       .values = owner.value_cache_int8_ptr(),
                       .key_scales = owner.key_cache_scales_ptr(),
                       .value_scales = owner.value_cache_scales_ptr()},
                .out = workspace_.op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = stream_.get()});
        } else {
            launch_gqa_decode_strict({
                .query = q,
                .kv = {.keys = owner.key_cache_bf16(),
                       .values = owner.value_cache_bf16()},
                .out = workspace_.op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = stream_.get()});
        }
        break;
    case AttentionAlgorithm::Alibi:
    case AttentionAlgorithm::Segmented:
    case AttentionAlgorithm::Flash:
    case AttentionAlgorithm::Gemm:
        throw UnsupportedAttentionCapability(plan);
    }
}

void CudaCompiledModel::store_and_attend_token_paged(
    AttentionLayer& attention, const AttentionSpec& owner_layout,
    const AttentionCapability& plan, int slot, __nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, const TokenKvPolicy& kv) {
    const AttentionSpec& layout = attention.layout;
    PhysicalPagedKvCache& paged_kv = *kv.paged_kv;
    const uint32_t* device_page_table = kv.device_page_table;
    const int page_table_stride = kv.page_table_stride;

    const GqaGeometry geometry{
        .q_heads = layout.query_heads,
        .kv_heads = owner_layout.key_value_heads,
        .head_dim = owner_layout.head_dim,
        .sliding_window = layout.sliding_window_size()};
    const PagedKvIndex index{
        .page_tables = device_page_table,
        .page_table_stride = page_table_stride,
        .attention_slot = slot,
        .page_tokens = paged_kv.page_tokens(),
        .page_vector_elements = paged_kv.page_vector_elements(),
        .layer_vector_offset = paged_kv.layer_vector_offset(slot)};
    const auto segmentation = [&] {
        const int chunks = (session_.position_ + 1 +
            resources_.options_.attention_chunk_tokens - 1) /
            resources_.options_.attention_chunk_tokens;
        return AttentionSegmentation{
            .chunk_tokens = resources_.options_.attention_chunk_tokens,
            .chunks = chunks,
            .partial_max = workspace_.attention_partial_max_.data(),
            .partial_denom = workspace_.attention_partial_denom_.data(),
            .partial_accum = workspace_.attention_partial_accum_.data()};
    };

    if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
        const PagedKvScaleIndex scale_index{
            .page_scale_elements = paged_kv.page_scale_elements(),
            .layer_scale_offset = paged_kv.layer_scale_offset(slot)};
        const Int8KvPoolView pool{
            .keys = paged_kv.key_int8(),
            .values = paged_kv.value_int8(),
            .key_scales = paged_kv.key_scales(),
            .value_scales = paged_kv.value_scales()};
        launch_store_kv_int8_paged_batch(
            k, v, paged_kv.key_int8(), paged_kv.value_int8(),
            paged_kv.key_scales(), paged_kv.value_scales(),
            device_page_table, page_table_stride, position_device_.data(),
            1, slot, paged_kv.page_tokens(),
            paged_kv.page_vector_elements(), paged_kv.layer_vector_offset(slot),
            paged_kv.page_scale_elements(), paged_kv.layer_scale_offset(slot),
            owner_layout.key_value_heads, owner_layout.head_dim, stream_.get());
        if (plan.algorithm == AttentionAlgorithm::Alibi) {
            launch_gqa_decode_alibi_int8_paged_batch({
                .query = q,
                .kv = pool,
                .index = index,
                .scale_index = scale_index,
                .out = workspace_.op_output_.data(),
                .positions = position_device_.data(),
                .rows = 1,
                .geometry = geometry,
                .alibi_slopes = attention.alibi_slopes.data(),
                .stream = stream_.get()});
        } else if (plan.algorithm == AttentionAlgorithm::Segmented) {
            launch_gqa_decode_int8_paged_segmented_batch({
                .query = q,
                .kv = pool,
                .index = index,
                .scale_index = scale_index,
                .out = workspace_.op_output_.data(),
                .positions = position_device_.data(),
                .rows = 1,
                .geometry = geometry,
                .segmentation = segmentation(),
                .stream = stream_.get()});
        } else {
            launch_gqa_decode_int8_paged_batch({
                .query = q,
                .kv = pool,
                .index = index,
                .scale_index = scale_index,
                .out = workspace_.op_output_.data(),
                .positions = position_device_.data(),
                .rows = 1,
                .geometry = geometry,
                .fast = plan.algorithm == AttentionAlgorithm::Online,
                .stream = stream_.get()});
        }
    } else {
        const Bf16KvPoolView pool{
            .keys = paged_kv.key_bf16(),
            .values = paged_kv.value_bf16()};
        launch_store_kv_paged_batch(
            k, v, paged_kv.key_bf16(), paged_kv.value_bf16(),
            device_page_table, page_table_stride, position_device_.data(),
            1, slot, paged_kv.page_tokens(),
            paged_kv.page_vector_elements(), paged_kv.layer_vector_offset(slot),
            owner_layout.key_value_heads, owner_layout.head_dim, stream_.get());
        if (plan.algorithm == AttentionAlgorithm::Alibi) {
            launch_gqa_decode_alibi_paged_batch({
                .query = q,
                .kv = pool,
                .index = index,
                .out = workspace_.op_output_.data(),
                .positions = position_device_.data(),
                .rows = 1,
                .geometry = geometry,
                .alibi_slopes = attention.alibi_slopes.data(),
                .stream = stream_.get()});
        } else if (plan.algorithm == AttentionAlgorithm::Segmented) {
            launch_gqa_decode_paged_segmented_batch({
                .query = q,
                .kv = pool,
                .index = index,
                .out = workspace_.op_output_.data(),
                .positions = position_device_.data(),
                .rows = 1,
                .geometry = geometry,
                .segmentation = segmentation(),
                .stream = stream_.get()});
        } else {
            launch_gqa_decode_paged_batch({
                .query = q,
                .kv = pool,
                .index = index,
                .out = workspace_.op_output_.data(),
                .positions = position_device_.data(),
                .rows = 1,
                .geometry = geometry,
                .fast = plan.algorithm == AttentionAlgorithm::Online,
                .stream = stream_.get()});
        }
    }
}

void CudaCompiledModel::run_token_latent_attention_paged(
    AttentionLayer& attention, LayerCommon& common_layer,
    const CompiledLayerProgram& semantics, int layer_index, const TokenKvPolicy& kv) {
    const AttentionSpec& layout = attention.layout;
    PhysicalPagedKvCache& paged_kv = *kv.paged_kv;
    if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument(
            "CUDA latent attention requires BF16 paged state storage");
    }
    if (layout.output_gate.has_value() || layout.multi_axis_position()) {
        throw std::invalid_argument(
            "CUDA latent attention does not support query gates or M-RoPE yet");
    }
    const auto& latent = *layout.latent_state();
    {
        auto native_fanout = native_fanout_scope(
            workspace_.normed_.data(), 1, resources_.program_.hidden);
        linear(workspace_.normed_.data(), *attention.latent_query,
               workspace_.latent_query_content_.data(), 1,
               layout.latent_query_content_width(), resources_.program_.hidden);
        if (layout.latent_query_rope_width() != 0) {
            linear(workspace_.normed_.data(), *attention.latent_query_rope,
                   workspace_.latent_query_rope_.data(), 1,
                   layout.latent_query_rope_width(), resources_.program_.hidden);
        }
        if (attention.latent_key && attention.latent_value) {
            linear(workspace_.normed_.data(), *attention.latent_key,
                   workspace_.latent_key_.data(), 1, latent.latent_rank,
                   resources_.program_.hidden);
            linear(workspace_.normed_.data(), *attention.latent_value,
                   workspace_.latent_value_.data(), 1, latent.latent_rank,
                   resources_.program_.hidden);
            if (attention.latent_key_rope && latent.decoupled_rope &&
                latent.rope_head_dim != 0) {
                linear(workspace_.normed_.data(), *attention.latent_key_rope,
                       workspace_.latent_key_rope_.data(), 1,
                       latent.rope_head_dim, resources_.program_.hidden);
            }
        }
    }
    if (const auto* rope = layout.rope_position();
        rope && attention.latent_key_rope && latent.decoupled_rope &&
        latent.rope_head_dim != 0) {
        launch_dynamic_qk_norm_rope(
            workspace_.latent_query_rope_.data(), workspace_.latent_key_rope_.data(),
            nullptr, nullptr, layout.query_heads, 1, latent.rope_head_dim,
            session_.position_, static_cast<float>(rope->theta), 1.0f,
            resources_.program_.final_norm.epsilon, false,
            lower_cuda_rope_scaling(*rope), rope->pairing, stream_.get());
    }
    const int cache_model_layer = attention.kv_owner_layer >= 0
        ? attention.kv_owner_layer : layer_index;
    const int slot = paged_kv.attention_slot(cache_model_layer);
    if (slot < 0) throw std::logic_error("latent attention has no page slot");
    AttentionLayer* owner = attention.kv_owner_layer >= 0
        ? as_attention(resources_.layers_.at(static_cast<size_t>(cache_model_layer)))
        : &attention;
    if (!owner) throw std::logic_error("CUDA latent KV owner is not attention");
    if (attention.latent_key && attention.latent_value) {
        launch_store_latent_paged_batch(
            workspace_.latent_key_.data(), workspace_.latent_value_.data(),
            attention.latent_key_rope && latent.decoupled_rope && latent.rope_head_dim != 0
                ? workspace_.latent_key_rope_.data() : nullptr,
            paged_kv.key_bf16(), paged_kv.value_bf16(), kv.device_page_table,
            kv.page_table_stride, position_device_.data(), 1, slot,
            paged_kv.page_tokens(), paged_kv.page_vector_elements(),
            paged_kv.layer_vector_offset(slot), latent.latent_rank,
            latent.decoupled_rope ? latent.rope_head_dim : 0, stream_.get());
    }
    launch_latent_attention_paged_batch({
        .query = {.content = workspace_.latent_query_content_.data(),
                  .rope = layout.latent_query_rope_width() != 0
                              ? workspace_.latent_query_rope_.data() : nullptr},
        .kv = {.keys = paged_kv.key_bf16(), .values = paged_kv.value_bf16()},
        .index = {.page_tables = kv.device_page_table,
                  .page_table_stride = kv.page_table_stride,
                  .attention_slot = slot,
                  .page_tokens = paged_kv.page_tokens(),
                  .page_vector_elements = paged_kv.page_vector_elements(),
                  .layer_vector_offset = paged_kv.layer_vector_offset(slot)},
        .out = workspace_.op_output_.data(),
        .positions = position_device_.data(),
        .rows = 1,
        .alibi_slopes = attention.alibi_slopes.data(),
        .geometry = {.query_heads = layout.query_heads,
                     .latent_rank = latent.latent_rank,
                     .rotary_width = latent.decoupled_rope ? latent.rope_head_dim : 0,
                     .score_scale = layout.query_scale,
                     .sliding_window = layout.sliding_window_size()},
        .stream = stream_.get()});
    const bool fuse_residual = resources_.options_.fused_residuals &&
        !semantics.mixer_norm.after.has_value() &&
        !std::holds_alternative<std::monostate>(semantics.feed_forward);
    linear(workspace_.op_output_.data(), *attention.out,
           workspace_.hidden_.data(), 1, resources_.program_.hidden,
           layout.latent_query_content_width(), fuse_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
}

void CudaCompiledModel::run_token_attention(
    AttentionLayer& attention, LayerCommon& common_layer,
    const CompiledLayerProgram& semantics, int layer_index, const TokenKvPolicy& kv) {
    const AttentionSpec& layout = attention.layout;
    const auto* compiled_attention =
        std::get_if<CompiledAttentionProgram>(&semantics.mixer);
    if (!compiled_attention) {
        throw std::logic_error("CUDA token attention has no compiled attention program");
    }

    if (compiled_attention->execution.kind != AttentionExecutionKind::Standard) {
        if (!kv.paged()) {
            throw std::invalid_argument(
                "CUDA latent attention is not implemented for contiguous host token execution");
        }
        run_token_latent_attention_paged(attention, common_layer, semantics,
                                         layer_index, kv);
        return;
    }

    AttentionLayer* owner = &attention;
    if (attention.kv_owner_layer >= 0) {
        owner = as_attention(resources_.layers_.at(
            static_cast<size_t>(attention.kv_owner_layer)));
        if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
    }
    const AttentionSpec& owner_layout = owner->layout;

    __nv_bfloat16* q = workspace_.qkv_output_.data();
    const int query_projection_width = attention.query->rows;
    const bool output_gate = layout.output_gate.has_value();
    const bool gate_packed = output_gate && layout.output_gate->packed_with_query;
    __nv_bfloat16* k = q + query_projection_width;
    __nv_bfloat16* v = k + layout.key_value_width();
    {
        auto native_fanout = native_fanout_scope(
            workspace_.normed_.data(), 1, resources_.program_.hidden);
        linear(workspace_.normed_.data(), *attention.query, q,
               1, query_projection_width, resources_.program_.hidden);
        if (attention.key && attention.value) {
            linear(workspace_.normed_.data(), *attention.key, k,
                   1, layout.key_value_width(), resources_.program_.hidden);
            linear(workspace_.normed_.data(), *attention.value, v,
                   1, layout.key_value_width(), resources_.program_.hidden);
        }
    }

    if (gate_packed) {
        // q currently holds `query_heads` chunks of `2 * head_dim`
        // (query, gate interleaved per head -- see
        // extract_attention_output_gate_kernel); de-interleave into
        // dedicated buffers before qk-norm/RoPE/attention touch q.
        launch_extract_attention_output_gate(
            q, workspace_.q_.data(), workspace_.attention_gate_.data(),
            1, layout.query_width(), layout.head_dim, stream_.get());
        q = workspace_.q_.data();
    }

    const float qk_epsilon = layout.query_norm
        ? layout.query_norm->epsilon
        : (layout.key_norm ? layout.key_norm->epsilon : resources_.program_.final_norm.epsilon);
    if (layout.has_query_key_norm()) {
        launch_attention_qk_norm(
            layout, q, attention.key ? k : nullptr,
            attention.q_norm, attention.k_norm, 1, stream_.get());
    }
    if (const auto* rope = layout.rope_position()) {
        const auto* multi = kv.paged() ? nullptr : layout.multi_axis_position();
        if (multi) {
            const auto& position =
                kv.rope_position ? *kv.rope_position : session_.next_rope_position_;
            CELEG_CUDA(cudaMemcpyAsync(mrope_position_device_.data(), position.data(),
                                       sizeof(position), cudaMemcpyHostToDevice,
                                       stream_.get()));
            launch_dynamic_mrope_qk_norm_rope(
                q, attention.key ? k : nullptr, nullptr, nullptr,
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                mrope_position_device_.data(), multi->sections[0], multi->sections[1],
                multi->sections[2], multi->interleaved,
                static_cast<float>(rope->theta), static_cast<float>(rope->rotary_fraction),
                qk_epsilon, false, lower_cuda_rope_scaling(*rope),
                stream_.get());
        } else {
            launch_dynamic_qk_norm_rope(
                q, attention.key ? k : nullptr, nullptr, nullptr,
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                session_.position_, static_cast<float>(rope->theta),
                static_cast<float>(rope->rotary_fraction), qk_epsilon,
                false, lower_cuda_rope_scaling(*rope), rope->pairing, stream_.get());
        }
    }
    launch_scale(q, layout.query_width(),
                 cuda_query_prescale(layout), stream_.get());

    const AttentionCapability plan = token_attention_plan(attention, owner_layout, kv);
    if (kv.paged()) {
        const int cache_model_layer = attention.kv_owner_layer >= 0
            ? attention.kv_owner_layer : layer_index;
        const int slot = kv.paged_kv->attention_slot(cache_model_layer);
        if (slot < 0) throw std::logic_error("attention layer has no page slot");
        store_and_attend_token_paged(attention, owner_layout, plan, slot, q, k, v, kv);
    } else {
        store_and_attend_token_contiguous(attention, *owner, plan, q, k, v);
    }

    if (output_gate) {
        const __nv_bfloat16* gate = workspace_.attention_gate_.data();
        if (!gate_packed) {
            linear(workspace_.normed_.data(), *attention.gate,
                   workspace_.attention_gate_.data(), 1,
                   layout.query_width(), resources_.program_.hidden);
        }
        launch_sigmoid_multiply(workspace_.op_output_.data(), gate,
                                layout.query_width(), stream_.get());
    }
    const bool fuse_residual = resources_.options_.fused_residuals &&
        !semantics.mixer_norm.after.has_value() &&
        !std::holds_alternative<std::monostate>(semantics.feed_forward);
    linear(workspace_.op_output_.data(), *attention.out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, layout.query_width(),
           fuse_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
}

}

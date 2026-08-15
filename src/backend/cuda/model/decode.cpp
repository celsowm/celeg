#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>


namespace celeg {


void CudaCompiledModel::run_token_layers(const TokenKvPolicy& kv) {
    int layer_index = 0;
    for (Layer& layer : resources_.layers_) {
        run_token_layer(layer, layer_index, kv);
        ++layer_index;
    }
}

void CudaCompiledModel::run_token_layer(Layer& layer, int layer_index,
                                        const TokenKvPolicy& kv) {
    LayerCommon& common_layer = common(layer);
    const CompiledLayerProgram& semantics =
        resources_.program_.layers.at(static_cast<size_t>(layer_index));

    if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
        CELEG_CUDA(cudaMemcpyAsync(
            workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
            cudaMemcpyDeviceToDevice, stream_.get()));
    }
    launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm,
                   workspace_.normed_.data(), 1, resources_.program_.hidden,
                   semantics.operator_norm.epsilon, stream_.get());

    run_token_mixer(layer, common_layer, semantics, layer_index, kv);

    if (common_layer.post_attention_norm) {
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.post_attention_norm,
                       workspace_.hidden_.data(), 1, resources_.program_.hidden,
                       semantics.post_attention_norm->epsilon, stream_.get());
    }
    if (!resources_.options_.fused_residuals || common_layer.post_attention_norm ||
        std::holds_alternative<std::monostate>(semantics.feed_forward)) {
        launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                            resources_.program_.hidden, stream_.get());
    }
    if (!std::holds_alternative<std::monostate>(semantics.feed_forward)) run_mlp_decode(common_layer, layer_index);
    if (std::binary_search(resources_.program_.norm_after_layers.begin(),
                           resources_.program_.norm_after_layers.end(), layer_index)) {
        launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_,
                       workspace_.hidden_.data(), 1, resources_.program_.hidden,
                       resources_.program_.final_norm.epsilon, stream_.get());
    }
}


void CudaCompiledModel::run_token_mixer(Layer& layer, LayerCommon& common_layer,
                                        const CompiledLayerProgram& semantics,
                                        int layer_index, const TokenKvPolicy& kv) {
    visit_layer(layer,
      [&](AttentionLayer* attention) {
        run_token_attention(*attention, common_layer, semantics, layer_index, kv);
      },
      [&](GatedDeltaNetLayer* gated_delta) {
        run_token_gated_delta(*gated_delta, semantics);
      },
      [&](Mamba2Layer* mamba) {
        run_token_mamba2(*mamba, semantics, kv);
      },
      [&](MlpOnlyLayer* mlp) {
        run_token_mlp_only(*mlp);
      },
      [&](ConvolutionLayer* convolution) {
        run_token_convolution(*convolution);
      });
}


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
            k, v, owner.key_cache_int8.data(), owner.value_cache_int8.data(),
            owner.key_cache_scales.data(), owner.value_cache_scales.data(),
            session_.position_, owner_layout.key_value_heads, owner_layout.head_dim,
            stream_.get());
    } else {
        launch_store_kv(k, v, owner.key_cache.data(), owner.value_cache.data(),
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
                .kv = {.keys = owner.key_cache_int8.data(),
                       .values = owner.value_cache_int8.data(),
                       .key_scales = owner.key_cache_scales.data(),
                       .value_scales = owner.value_cache_scales.data()},
                .out = workspace_.op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = stream_.get()});
        } else {
            launch_gqa_decode_online({
                .query = q,
                .kv = {.keys = owner.key_cache.data(),
                       .values = owner.value_cache.data()},
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
                .kv = {.keys = owner.key_cache_int8.data(),
                       .values = owner.value_cache_int8.data(),
                       .key_scales = owner.key_cache_scales.data(),
                       .value_scales = owner.value_cache_scales.data()},
                .out = workspace_.op_output_.data(),
                .geometry = geometry,
                .extent = extent,
                .stream = stream_.get()});
        } else {
            launch_gqa_decode_strict({
                .query = q,
                .kv = {.keys = owner.key_cache.data(),
                       .values = owner.value_cache.data()},
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
    if (layout.output_gate.enabled() || layout.multi_axis_position()) {
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
            workspace_.latent_query_rope_.data(),
            workspace_.latent_key_rope_.data(), nullptr, nullptr,
            layout.query_heads, 1, latent.rope_head_dim,
            session_.position_, static_cast<float>(rope->theta), 1.0f,
            semantics.operator_norm.epsilon, false,
            lower_cuda_rope_scaling(*rope), stream_.get());
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
            attention.latent_key_rope && latent.decoupled_rope &&
                    latent.rope_head_dim != 0
                ? workspace_.latent_key_rope_.data() : nullptr,
            paged_kv.key_bf16(), paged_kv.value_bf16(), kv.device_page_table,
            kv.page_table_stride, position_device_.data(), 1, slot,
            paged_kv.page_tokens(), paged_kv.page_vector_elements(),
            paged_kv.layer_vector_offset(slot), latent.latent_rank,
            latent.decoupled_rope ? latent.rope_head_dim : 0, stream_.get());
    }
    const float score_scale = layout.query_scale;
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
                     .score_scale = score_scale,
                     .sliding_window = layout.sliding_window_size()},
        .stream = stream_.get()});
    linear(workspace_.op_output_.data(), *attention.out,
           workspace_.hidden_.data(), 1, resources_.program_.hidden,
           layout.latent_query_content_width(),
           resources_.options_.fused_residuals && !common_layer.post_attention_norm &&
               std::holds_alternative<std::monostate>(semantics.feed_forward) ? 0.0f : 1.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier,
                 stream_.get());
}

void CudaCompiledModel::run_token_attention(
    AttentionLayer& attention, LayerCommon& common_layer,
    const CompiledLayerProgram& semantics, int layer_index, const TokenKvPolicy& kv) {
    const AttentionSpec& layout = attention.layout;

    if (layout.uses_latent_state()) {
        if (!kv.paged()) {
            throw std::invalid_argument(
                "CUDA latent attention is not implemented for contiguous host token "
                "execution");
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
    const bool output_gate = layout.output_gate.enabled();
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

    if (const auto* rope = layout.rope_position()) {
        const auto* multi = kv.paged() ? nullptr : layout.multi_axis_position();
        if (multi) {
            const auto& position =
                kv.rope_position ? *kv.rope_position : session_.next_rope_position_;
            CELEG_CUDA(cudaMemcpyAsync(mrope_position_device_.data(), position.data(),
                                       sizeof(position), cudaMemcpyHostToDevice,
                                       stream_.get()));
            launch_dynamic_mrope_qk_norm_rope(
                q, attention.key ? k : nullptr, attention.q_norm, attention.k_norm,
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                mrope_position_device_.data(), multi->sections[0],
                multi->sections[1], multi->sections[2], multi->interleaved,
                static_cast<float>(rope->theta),
                static_cast<float>(rope->rotary_fraction),
                semantics.operator_norm.epsilon, layout.has_query_key_norm(),
                lower_cuda_rope_scaling(*rope),
                stream_.get());
        } else {
            launch_dynamic_qk_norm_rope(
                q, attention.key ? k : nullptr, attention.q_norm, attention.k_norm,
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                session_.position_, static_cast<float>(rope->theta),
                static_cast<float>(rope->rotary_fraction), semantics.operator_norm.epsilon,
                layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream_.get());
        }
    } else if (kv.paged() && layout.has_query_key_norm()) {
        launch_dynamic_qk_norm_rope(
            q, attention.key ? k : nullptr, attention.q_norm, attention.k_norm,
            layout.query_heads, layout.key_value_heads, layout.head_dim,
            session_.position_, 1.0f, 0.0f, layout.query_norm->epsilon, true,
            CudaRopeScaling{}, stream_.get());
    }
    launch_scale(q, layout.query_width(), layout.query_scale, stream_.get());

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
        const __nv_bfloat16* gate = q + layout.query_width();
        if (!layout.output_gate.packed_with_query) {
            linear(workspace_.normed_.data(), *attention.gate,
                   workspace_.attention_gate_.data(), 1,
                   layout.query_width(), resources_.program_.hidden);
            gate = workspace_.attention_gate_.data();
        }
        launch_sigmoid_multiply(workspace_.op_output_.data(), gate,
                                layout.query_width(), stream_.get());
    }
    linear(workspace_.op_output_.data(), *attention.out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, layout.query_width(),
           resources_.options_.fused_residuals && !common_layer.post_attention_norm &&
               std::holds_alternative<std::monostate>(semantics.feed_forward) ? 0.0f : 1.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
}


void CudaCompiledModel::run_token_gated_delta(GatedDeltaNetLayer& gated_delta,
                                              const CompiledLayerProgram& semantics) {
    const GatedDeltaNetSpec& spec = gated_delta.spec;
    const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
        spec.value_heads * spec.value_head_dim;
    const int value_width = spec.value_heads * spec.value_head_dim;
    if (spec.factorized_projections) {
        linear(workspace_.normed_.data(), *gated_delta.q,
               workspace_.gated_delta_qkv_.data(), 1,
               spec.key_heads * spec.key_head_dim, resources_.program_.hidden);
        linear(workspace_.normed_.data(), *gated_delta.k,
               workspace_.qkv_output_.data(), 1,
               spec.key_heads * spec.key_head_dim, resources_.program_.hidden);
        linear(workspace_.normed_.data(), *gated_delta.v,
               workspace_.gated_delta_output_.data(), 1, value_width,
               resources_.program_.hidden);
        launch_interleave_gated_delta_qkv(
            workspace_.gated_delta_qkv_.data(), workspace_.qkv_output_.data(),
            workspace_.gated_delta_output_.data(), workspace_.gated_delta_qkv_.data(),
            1, spec.key_heads * spec.key_head_dim, value_width, stream_.get());
    } else {
        linear(workspace_.normed_.data(), *gated_delta.qkv,
               workspace_.gated_delta_qkv_.data(), 1, qkv_width,
               resources_.program_.hidden);
    }
    linear(workspace_.normed_.data(), *gated_delta.z,
           workspace_.gated_delta_z_.data(), 1, value_width,
           resources_.program_.hidden);
    linear(workspace_.normed_.data(), *gated_delta.b,
           workspace_.gated_delta_b_.data(), 1, spec.value_heads,
           resources_.program_.hidden);
    linear(workspace_.normed_.data(), *gated_delta.a,
           workspace_.gated_delta_a_.data(), 1, spec.decay_width(),
           resources_.program_.hidden);
    launch_gated_delta_net(workspace_.gated_delta_qkv_.data(),
        workspace_.gated_delta_z_.data(), workspace_.gated_delta_b_.data(),
        workspace_.gated_delta_a_.data(), gated_delta.conv_weight,
        gated_delta.dt_bias, gated_delta.a_log, gated_delta.norm,
        gated_delta.conv_state.data(), gated_delta.recurrent_state.data(),
        workspace_.gated_delta_output_.data(), 1, spec.conv_kernel,
        spec.key_head_dim, spec.value_head_dim, spec.key_heads,
        spec.value_heads, semantics.operator_norm.epsilon,
        spec.vector_decay, spec.safe_decay, spec.decay_lower_bound,
        spec.sigmoid_output_gate, stream_.get());
    linear(workspace_.gated_delta_output_.data(), *gated_delta.out,
           workspace_.hidden_.data(), 1, resources_.program_.hidden,
           value_width);
}

void CudaCompiledModel::run_token_mamba2(Mamba2Layer& mamba,
                                         const CompiledLayerProgram& semantics,
                                         const TokenKvPolicy& kv) {
    const Mamba2Spec& spec = mamba.spec;
    linear(workspace_.normed_.data(), *mamba.in,
           workspace_.mamba_projected_.data(), 1,
           2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
               spec.num_heads, resources_.program_.hidden);
    launch_mamba2_step(workspace_.mamba_projected_.data(), mamba.conv_weight,
                       mamba.conv_bias, mamba.dt_bias, mamba.a_log, mamba.d,
                       mamba.conv_state.data(), mamba.ssm_state.data(),
                       workspace_.mamba_inner_.data(), spec.intermediate_size,
                       spec.state_size, spec.num_heads, spec.head_dim,
                       spec.group_count, spec.conv_kernel, stream_.get());
    const float epsilon = kv.paged() ? semantics.operator_norm.epsilon
                                     : semantics.post_attention_norm->epsilon;
    launch_rmsnorm(workspace_.mamba_inner_.data(), mamba.norm,
                   workspace_.op_output_.data(), 1, spec.intermediate_size,
                   epsilon, stream_.get());
    launch_multiply(workspace_.op_output_.data(), workspace_.mamba_projected_.data(),
                    spec.intermediate_size, stream_.get());
    linear(workspace_.op_output_.data(), *mamba.out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, spec.intermediate_size);
}

void CudaCompiledModel::run_token_mlp_only(MlpOnlyLayer& mlp) {
    linear(workspace_.normed_.data(), *mlp.up, workspace_.gate_up_.data(),
           1, mlp.spec.intermediate_size, resources_.program_.hidden);
    launch_relu2(workspace_.gate_up_.data(), workspace_.activated_.data(),
                 mlp.spec.intermediate_size, stream_.get());
    linear(workspace_.activated_.data(), *mlp.down, workspace_.hidden_.data(),
           1, resources_.program_.hidden, mlp.spec.intermediate_size);
}

void CudaCompiledModel::run_token_convolution(ConvolutionLayer& convolution) {
    linear(workspace_.normed_.data(), *convolution.conv_in,
           workspace_.conv_projected_.data(),
           1, 3 * resources_.program_.hidden, resources_.program_.hidden);
    launch_conv_decode(
        workspace_.conv_projected_.data(), convolution.conv_weight,
        convolution.conv_state.data(), workspace_.op_output_.data(),
        resources_.program_.hidden, resources_.shape_.conv_cache, session_.position_,
        stream_.get());
    linear(workspace_.op_output_.data(), *convolution.conv_out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, resources_.program_.hidden,
           resources_.options_.fused_residuals ? 1.0f : 0.0f);
}


void CudaCompiledModel::run_token_logits() {
    launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_, workspace_.normed_.data(),
                   1, resources_.program_.hidden, resources_.program_.final_norm.epsilon,
                   stream_.get());
    linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
           1, resources_.dims_.vocab_size, resources_.program_.hidden);
    launch_scale(workspace_.logits_.data(), resources_.dims_.vocab_size,
                 resources_.program_.logits_multiplier /
                     resources_.program_.logits_divisor, stream_.get());
    if (resources_.program_.final_logit_softcap > 0.0f) {
        launch_tanh_softcap(workspace_.logits_.data(), resources_.dims_.vocab_size,
                            resources_.program_.final_logit_softcap, stream_.get());
    }
}


void CudaCompiledModel::forward_token_host(int32_t token, bool compute_logits,
                                           const float* raw_embedding,
                                           const std::array<int32_t, 3>* rope_position) {
    if (session_.position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (raw_embedding) {
        std::vector<__nv_bfloat16> converted(static_cast<size_t>(resources_.program_.hidden));
        for (int index = 0; index < resources_.program_.hidden; ++index) {
            converted[static_cast<size_t>(index)] = __float2bfloat16(raw_embedding[index]);
        }
        CELEG_CUDA(cudaMemcpyAsync(workspace_.hidden_.data(), converted.data(),
                                   converted.size() * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, stream_.get()));
        initialize_per_layer_input_host(resources_.dims_.token_policy.pad_token_id);
    } else {
        resources_.weight_layout_->embed_token(
            token, workspace_.hidden_.data(), resources_.program_.hidden, stream_.get());
        launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                     resources_.program_.embedding_transform.multiplier, stream_.get());
        if (resources_.program_.embedding_transform.post_norm) {
            launch_rmsnorm(workspace_.hidden_.data(), resources_.embedding_norm_,
                           workspace_.hidden_.data(), 1, resources_.program_.hidden,
                           resources_.program_.embedding_transform.post_norm->epsilon,
                           stream_.get());
        }
        initialize_per_layer_input_host(token);
    }

    TokenKvPolicy kv;
    kv.kv_layout = AttentionKvLayout::Contiguous;
    kv.position_source = AttentionPositionSource::HostScalar;
    kv.rope_position = rope_position;
    run_token_layers(kv);

    if (resources_.mtp_.available()) {
        run_mtp_forward(token, rope_position);
    }
    if (compute_logits) {
        run_token_logits();
        finalize_mtp_verification();
    }
    ++session_.position_;
    if (!rope_position) {
        for (int32_t& value : session_.next_rope_position_) ++value;
    }
}

void CudaCompiledModel::forward_token_paged_host(
    int32_t token, bool compute_logits, PhysicalPagedKvCache& paged_kv,
    const uint32_t* device_page_table, int page_table_stride) {
    if (session_.position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (paged_kv.mode() != resources_.options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    if (resources_.mtp_.available()) {
        throw std::runtime_error("MTP is incompatible with paged KV execution");
    }
    resources_.weight_layout_->embed_token(
        token, workspace_.hidden_.data(), resources_.program_.hidden, stream_.get());
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 resources_.program_.embedding_transform.multiplier, stream_.get());
    initialize_per_layer_input_host(token);

    TokenKvPolicy kv;
    kv.kv_layout = AttentionKvLayout::Paged;
    kv.position_source = AttentionPositionSource::DeviceCounter;
    kv.paged_kv = &paged_kv;
    kv.device_page_table = device_page_table;
    kv.page_table_stride = page_table_stride;
    run_token_layers(kv);

    if (compute_logits) run_token_logits();
    ++session_.position_;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_,
                               sizeof(session_.position_),
                               cudaMemcpyHostToDevice, stream_.get()));
}

}

#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/kernels/rope_pairing.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {

void CudaCompiledModel::prefill_batched(const std::vector<int32_t>& tokens) {
    validate_token_ids(tokens);
    reset();
    const int rows = static_cast<int>(tokens.size());
    allocate_prefill_workspace(rows);
    auto& prof = prefill_phase_profile();
    prof.count_step();

    CELEG_CUDA(cudaMemcpyAsync(workspace_.prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    prof.begin(stream_.get());
    launch_mark_seen_batch(workspace_.prefill_tokens_.data(), rows, sampling_.seen_tokens.data(),
                           resources_.dims_.vocab_size, stream_.get());
    resources_.weight_layout_->embed_batch(
        workspace_.prefill_tokens_.data(), rows, workspace_.prefill_hidden_.data(),
        resources_.program_.hidden, stream_.get());
    launch_scale(workspace_.prefill_hidden_.data(), rows * resources_.program_.hidden,
                 resources_.program_.embedding_transform.multiplier, stream_.get());
    if (resources_.program_.embedding_transform.post_norm) {
        launch_rmsnorm(workspace_.prefill_hidden_.data(), resources_.embedding_norm_,
                       workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                       resources_.program_.embedding_transform.post_norm->epsilon,
                       stream_.get());
    }
    initialize_per_layer_input_batch(workspace_.prefill_tokens_.data(), rows);
    prof.end(PrefillPhase::Embed, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : resources_.layers_) {
        LayerCommon& common_layer = common(layer);
        const CompiledLayerProgram& semantics = resources_.program_.layers.at(static_cast<size_t>(layer_idx));
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
            CELEG_CUDA(cudaMemcpyAsync(
                workspace_.prefill_residual_.data(), workspace_.prefill_hidden_.data(),
                workspace_.prefill_hidden_.bytes(), cudaMemcpyDeviceToDevice,
                stream_.get()));
        }
        prof.begin(stream_.get());
        launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.operator_norm,
                       workspace_.prefill_normed_.data(), rows, resources_.program_.hidden,
                       semantics.operator_norm.epsilon, stream_.get());
        prof.end(PrefillPhase::Norm, stream_.get());

        visit_layer(layer,
          [&](GatedDeltaNetLayer* gated_delta) {
            const GatedDeltaNetSpec& spec = gated_delta->spec;
            const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
                spec.value_heads * spec.value_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            prof.begin(stream_.get());
            {
            auto native_fanout = native_fanout_scope(
                workspace_.prefill_normed_.data(), rows, resources_.program_.hidden);
            if (spec.factorized_projections) {
                linear(workspace_.prefill_normed_.data(), *gated_delta->q,
                       workspace_.prefill_gated_delta_qkv_.data(), rows,
                       spec.key_heads * spec.key_head_dim, resources_.program_.hidden);
                linear(workspace_.prefill_normed_.data(), *gated_delta->k,
                       workspace_.prefill_k_.data(), rows,
                       spec.key_heads * spec.key_head_dim, resources_.program_.hidden);
                linear(workspace_.prefill_normed_.data(), *gated_delta->v,
                       workspace_.prefill_v_.data(), rows, value_width,
                       resources_.program_.hidden);
                launch_interleave_gated_delta_qkv(
                    workspace_.prefill_gated_delta_qkv_.data(), workspace_.prefill_k_.data(),
                    workspace_.prefill_v_.data(), workspace_.prefill_gated_delta_qkv_.data(),
                    rows, spec.key_heads * spec.key_head_dim, value_width, stream_.get());
            } else {
                linear(workspace_.prefill_normed_.data(), *gated_delta->qkv,
                       workspace_.prefill_gated_delta_qkv_.data(), rows, qkv_width,
                       resources_.program_.hidden);
            }
            linear(workspace_.prefill_normed_.data(), *gated_delta->z,
                   workspace_.prefill_gated_delta_z_.data(), rows, value_width,
                   resources_.program_.hidden);
            linear(workspace_.prefill_normed_.data(), *gated_delta->b,
                   workspace_.prefill_gated_delta_b_.data(), rows, spec.value_heads,
                   resources_.program_.hidden);
            linear(workspace_.prefill_normed_.data(), *gated_delta->a,
                   workspace_.prefill_gated_delta_a_.data(), rows, spec.decay_width(),
                   resources_.program_.hidden);
            }
            prof.end(PrefillPhase::QkvProj, stream_.get());
            prof.begin(stream_.get());
            launch_gated_delta_net(workspace_.prefill_gated_delta_qkv_.data(),
                workspace_.prefill_gated_delta_z_.data(),
                workspace_.prefill_gated_delta_b_.data(),
                workspace_.prefill_gated_delta_a_.data(), gated_delta->conv_weight,
                gated_delta->dt_bias, gated_delta->a_log, gated_delta->norm,
                gated_delta->conv_state.data(), gated_delta->recurrent_state.data(),
                workspace_.prefill_gated_delta_output_.data(), rows, spec.conv_kernel,
                spec.key_head_dim, spec.value_head_dim, spec.key_heads,
                spec.value_heads, semantics.operator_norm.epsilon,
                spec.vector_decay, spec.safe_decay, spec.decay_lower_bound,
                spec.sigmoid_output_gate, stream_.get());
            prof.end(PrefillPhase::Conv, stream_.get());
            prof.begin(stream_.get());
            linear(workspace_.prefill_gated_delta_output_.data(), *gated_delta->out,
                   workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                   value_width);
            launch_scale(workspace_.prefill_hidden_.data(), rows * resources_.program_.hidden,
                         semantics.residual.multiplier, stream_.get());
            prof.end(PrefillPhase::AttnOut, stream_.get());
          },
          [&](Mamba2Layer* mamba) {
            const Mamba2Spec& spec = mamba->spec;
            const int projection_width = 2 * spec.intermediate_size +
                2 * spec.group_count * spec.state_size + spec.num_heads;
            linear(workspace_.prefill_normed_.data(), *mamba->in,
                   workspace_.prefill_mamba_projected_.data(), rows, projection_width,
                   resources_.program_.hidden);
            launch_mamba2_prefill(
                workspace_.prefill_mamba_projected_.data(), mamba->conv_weight,
                mamba->conv_bias, mamba->dt_bias, mamba->a_log, mamba->d,
                mamba->conv_state.data(), mamba->ssm_state.data(),
                workspace_.prefill_mamba_inner_.data(), rows, spec.intermediate_size,
                spec.state_size, spec.num_heads, spec.head_dim, spec.group_count,
                spec.conv_kernel, stream_.get());
            launch_rmsnorm(workspace_.prefill_mamba_inner_.data(), mamba->norm,
                           workspace_.prefill_mamba_inner_.data(), rows,
                           spec.intermediate_size, semantics.operator_norm.epsilon,
                           stream_.get());
            launch_multiply(workspace_.prefill_mamba_inner_.data(),
                            workspace_.prefill_mamba_projected_.data(),
                            rows * spec.intermediate_size, stream_.get());
            linear(workspace_.prefill_mamba_inner_.data(), *mamba->out,
                   workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                   spec.intermediate_size);
          },
          [&](MlpOnlyLayer* mlp) {
            const int intermediate = mlp->spec.intermediate_size;
            linear(workspace_.prefill_normed_.data(), *mlp->up,
                   workspace_.prefill_gate_up_.data(), rows, intermediate,
                   resources_.program_.hidden);
            switch (mlp->spec.activation) {
            case ActivationKind::Relu2:
                launch_relu2(workspace_.prefill_gate_up_.data(),
                             workspace_.prefill_activated_.data(),
                             rows * intermediate, stream_.get());
                break;
            case ActivationKind::GeluTanh:
                launch_gelu_tanh(workspace_.prefill_gate_up_.data(),
                                 workspace_.prefill_activated_.data(),
                                 rows * intermediate, stream_.get());
                break;
            default:
                throw std::runtime_error("CUDA prefill does not implement MLP-only activation");
            }
            linear(workspace_.prefill_activated_.data(), *mlp->down,
                   workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                   intermediate);
          },
          [&](AttentionLayer* attention) {
            const AttentionSpec& layout = attention->layout;
            AttentionLayer* owner = attention;
            if (attention->kv_owner_layer >= 0) {
                owner = as_attention(resources_.layers_.at(
                    static_cast<size_t>(attention->kv_owner_layer)));
                if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
            }
            const AttentionSpec& owner_layout = owner->layout;
            if (layout.uses_latent_state()) {
                if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                    throw std::invalid_argument(
                        "CUDA latent attention requires BF16 state storage");
                }
                const auto& latent = *layout.latent_state();
                if (latent.factorized) {
                    prof.begin(stream_.get());
                    {
                    auto native_fanout = native_fanout_scope(
                        workspace_.prefill_normed_.data(), rows, resources_.program_.hidden);
                    linear(workspace_.prefill_normed_.data(), *attention->latent_query_projection,
                           workspace_.prefill_latent_projection_.data(), rows,
                           latent.query_rank, resources_.program_.hidden);
                    launch_rmsnorm(workspace_.prefill_latent_projection_.data(),
                                   attention->latent_query_norm,
                                   workspace_.prefill_latent_projection_.data(), rows,
                                   latent.query_rank, latent.query_latent_norm.epsilon,
                                   stream_.get());
                    linear(workspace_.prefill_latent_projection_.data(),
                           *attention->latent_query_expansion,
                           workspace_.prefill_qkv_.data(), rows,
                           layout.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
                           latent.query_rank);
                    launch_factorized_latent_query(
                        workspace_.prefill_qkv_.data(), attention->latent_expansion->bf16,
                        workspace_.prefill_latent_query_content_.data(), rows,
                        layout.query_heads, latent.nope_head_dim, latent.rope_head_dim,
                        latent.latent_rank, stream_.get());
                    launch_factorized_latent_rope(
                        workspace_.prefill_qkv_.data(),
                        workspace_.prefill_latent_query_rope_.data(), rows,
                        layout.query_heads, latent.nope_head_dim, latent.rope_head_dim,
                        stream_.get());
                    linear(workspace_.prefill_normed_.data(), *attention->latent_key_projection,
                           workspace_.prefill_qkv_.data(), rows,
                           latent.latent_rank + latent.rope_head_dim,
                           resources_.program_.hidden);
                    launch_rmsnorm(workspace_.prefill_qkv_.data(), attention->latent_key_norm,
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
                        owner->latent_key_cache.data(), owner->latent_value_cache.data(),
                        owner->latent_key_rope_cache.data(), rows, latent.latent_rank,
                        latent.rope_head_dim, stream_.get());
                    const float score_scale = layout.query_scale;
                    launch_latent_attention_prefill(
                        workspace_.prefill_latent_query_content_.data(),
                        workspace_.prefill_latent_query_rope_.data(),
                        owner->latent_key_cache.data(), owner->latent_value_cache.data(),
                        owner->latent_key_rope_cache.data(),
                        workspace_.prefill_op_output_.data(), rows,
                        attention->alibi_slopes.data(), layout.query_heads,
                        latent.latent_rank, latent.rope_head_dim, score_scale,
                        layout.sliding_window_size(), stream_.get());
                    prof.end(PrefillPhase::Attention, stream_.get());
                    prof.begin(stream_.get());
                    launch_factorized_latent_value(
                        workspace_.prefill_op_output_.data(), attention->latent_expansion->bf16,
                        workspace_.prefill_latent_decompressed_.data(), rows,
                        layout.query_heads, latent.nope_head_dim, latent.value_head_dim,
                        latent.latent_rank, stream_.get());
                    linear(workspace_.prefill_normed_.data(), *attention->gate,
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
                    linear(workspace_.prefill_latent_decompressed_.data(), *attention->out,
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
                linear(workspace_.prefill_normed_.data(), *attention->latent_query,
                       workspace_.prefill_latent_query_content_.data(), rows,
                       layout.latent_query_content_width(), resources_.program_.hidden);
                if (layout.latent_query_rope_width() != 0) {
                    linear(workspace_.prefill_normed_.data(), *attention->latent_query_rope,
                           workspace_.prefill_latent_query_rope_.data(), rows,
                           layout.latent_query_rope_width(), resources_.program_.hidden);
                }
                if (attention->latent_key && attention->latent_value) {
                    linear(workspace_.prefill_normed_.data(), *attention->latent_key,
                           workspace_.prefill_latent_key_.data(), rows,
                           latent.latent_rank, resources_.program_.hidden);
                    linear(workspace_.prefill_normed_.data(), *attention->latent_value,
                           workspace_.prefill_latent_value_.data(), rows,
                           latent.latent_rank, resources_.program_.hidden);
                    if (attention->latent_key_rope && latent.decoupled_rope &&
                        latent.rope_head_dim != 0) {
                        linear(workspace_.prefill_normed_.data(), *attention->latent_key_rope,
                               workspace_.prefill_latent_key_rope_.data(), rows,
                               latent.rope_head_dim, resources_.program_.hidden);
                    }
                }
                }
                prof.end(PrefillPhase::QkvProj, stream_.get());
                prof.begin(stream_.get());
                if (const auto* rope = layout.rope_position();
                    rope && attention->latent_key_rope && latent.decoupled_rope &&
                    latent.rope_head_dim != 0) {
                    launch_dynamic_qk_norm_rope_prefill(
                        workspace_.prefill_latent_query_rope_.data(),
                        attention->latent_key ? workspace_.prefill_latent_key_rope_.data() : nullptr,
                        nullptr, nullptr, rows, layout.query_heads, 1,
                        latent.rope_head_dim, static_cast<float>(rope->theta), 1.0f,
                        layout.query_norm.epsilon, false,
                        lower_cuda_rope_scaling(*rope), stream_.get());
                }
                prof.end(PrefillPhase::RopeKv, stream_.get());
                prof.begin(stream_.get());
                if (attention->latent_key && attention->latent_value) {
                    launch_store_latent_prefill(
                        workspace_.prefill_latent_key_.data(),
                        workspace_.prefill_latent_value_.data(),
                        attention->latent_key_rope && latent.decoupled_rope &&
                        latent.rope_head_dim != 0
                            ? workspace_.prefill_latent_key_rope_.data() : nullptr,
                        owner->latent_key_cache.data(), owner->latent_value_cache.data(),
                        owner->latent_key_rope_cache.data(), rows, latent.latent_rank,
                        latent.decoupled_rope ? latent.rope_head_dim : 0, stream_.get());
                }
                const float score_scale = layout.query_scale;
                launch_latent_attention_prefill(
                    workspace_.prefill_latent_query_content_.data(),
                    layout.latent_query_rope_width() != 0
                        ? workspace_.prefill_latent_query_rope_.data() : nullptr,
                    owner->latent_key_cache.data(), owner->latent_value_cache.data(),
                    owner->latent_key_rope_cache.data(), workspace_.prefill_op_output_.data(),
                    rows, attention->alibi_slopes.data(), layout.query_heads,
                    latent.latent_rank, latent.decoupled_rope ? latent.rope_head_dim : 0,
                    score_scale, layout.sliding_window_size(), stream_.get());
                prof.end(PrefillPhase::Attention, stream_.get());
                prof.begin(stream_.get());
                linear(workspace_.prefill_op_output_.data(), *attention->out,
                       workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                       layout.latent_query_content_width(),
                       resources_.options_.fused_residuals && !common_layer.post_attention_norm
                           ? 1.0f : 0.0f);
                launch_scale(workspace_.prefill_hidden_.data(),
                             rows * resources_.program_.hidden,
                             semantics.residual.multiplier,
                             stream_.get());
                prof.end(PrefillPhase::AttnOut, stream_.get());
            } else {
            const int query_projection_width = attention->query->rows;
            const bool output_gate = layout.output_gate.enabled();
            const bool gate_packed = output_gate && layout.output_gate.packed_with_query;
            prof.begin(stream_.get());
            {
            auto native_fanout = native_fanout_scope(
                workspace_.prefill_normed_.data(), rows, resources_.program_.hidden);
            // When the gate is packed with the query, project into the raw
            // scratch buffer: the compacted query and the extracted gate are
            // read out into disjoint destinations below, since compacting a
            // [query|gate] row in place cannot be done race-free across blocks.
            linear(workspace_.prefill_normed_.data(), *attention->query,
                   gate_packed ? workspace_.prefill_qkv_.data() : workspace_.prefill_q_.data(),
                   rows, query_projection_width,
                   resources_.program_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.prefill_normed_.data(), *attention->key,
                       workspace_.prefill_k_.data(), rows, layout.key_value_width(),
                       resources_.program_.hidden);
                linear(workspace_.prefill_normed_.data(), *attention->value,
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
                    linear(workspace_.prefill_normed_.data(), *attention->gate,
                           workspace_.prefill_attention_gate_.data(), rows,
                           layout.query_width(), resources_.program_.hidden);
                }
            }
            if (const auto* rope = layout.rope_position()) {
                launch_dynamic_qk_norm_rope_prefill(
                    workspace_.prefill_q_.data(), attention->key ? workspace_.prefill_k_.data() : nullptr,
                    attention->q_norm, attention->k_norm, rows, layout.query_heads,
                    layout.key_value_heads, layout.head_dim, static_cast<float>(rope->theta),
                    static_cast<float>(rope->rotary_fraction), layout.query_norm.epsilon,
                    layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream_.get());
            } else if (layout.has_query_key_norm()) {
                launch_dynamic_qk_norm_rope_prefill(
                    workspace_.prefill_q_.data(), attention->key ? workspace_.prefill_k_.data() : nullptr,
                    attention->q_norm, attention->k_norm, rows, layout.query_heads,
                    layout.key_value_heads, layout.head_dim, 1.0f, 0.0f,
                    layout.query_norm.epsilon, true, CudaRopeScaling{}, stream_.get());
            }
            launch_scale(workspace_.prefill_q_.data(),
                         static_cast<size_t>(rows) * layout.query_width(),
                         layout.query_scale, stream_.get());
            prof.end(PrefillPhase::RopeKv, stream_.get());

            prof.begin(stream_.get());
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                if (attention->key && attention->value) launch_store_kv_int8_prefill(
                    workspace_.prefill_k_.data(), workspace_.prefill_v_.data(),
                    owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                    owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                    rows, owner_layout.key_value_heads, owner_layout.head_dim,
                    stream_.get());
                if (attention->alibi_slopes.data()) {
                    launch_gqa_prefill_alibi_int8(
                        workspace_.prefill_q_.data(), owner->key_cache_int8.data(),
                        owner->value_cache_int8.data(), owner->key_cache_scales.data(),
                        owner->value_cache_scales.data(), workspace_.prefill_op_output_.data(),
                        rows, attention->alibi_slopes.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window_size(), stream_.get());
                } else if (resources_.options_.fast_attention) {
                    launch_gqa_prefill_online_int8(
                        workspace_.prefill_q_.data(), owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                        workspace_.prefill_op_output_.data(), rows, layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window_size(), stream_.get());
                } else {
                    launch_gqa_prefill_strict_int8(
                        workspace_.prefill_q_.data(), owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                        workspace_.prefill_op_output_.data(), rows, layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window_size(), stream_.get());
                }
            } else {
                if (attention->key && attention->value) launch_store_kv_prefill(
                    workspace_.prefill_k_.data(), workspace_.prefill_v_.data(),
                    owner->key_cache.data(), owner->value_cache.data(),
                    rows, owner_layout.key_value_width(), stream_.get());
                if (attention->alibi_slopes.data()) {
                    launch_gqa_prefill_alibi(
                        workspace_.prefill_q_.data(), owner->key_cache.data(),
                        owner->value_cache.data(), workspace_.prefill_op_output_.data(),
                        rows, attention->alibi_slopes.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window_size(), stream_.get());
                } else if (resources_.options_.fast_attention) {
                    // CELEG_FLASH_ATTN is resolved once into
                    // CudaModelOptions::flash_attn at model-configuration
                    // construction time (see runtime_types.hpp); execution
                    // code just reads the resolved option.
                    const bool use_flash = resources_.options_.flash_attn;
                    // The batched GEMM path is only valid for the narrow-head
                    // layouts it was tuned for.  With head_dim=128 its
                    // strided GQA batches can corrupt the KV state; use the
                    // tiled path automatically for wider heads so the next
                    // decode step sees a valid cache.  This is a kernel
                    // capability boundary, not an architecture dispatch.
                    const bool flash_supported = owner_layout.head_dim <= 128;
                    if (flash_supported && (use_flash || owner_layout.head_dim > 64)) {
                        launch_gqa_prefill_flash(
                            workspace_.prefill_q_.data(),
                            owner->key_cache.data(), owner->value_cache.data(),
                            workspace_.prefill_op_output_.data(), rows,
                            layout.query_heads, owner_layout.key_value_heads,
                            owner_layout.head_dim, layout.query_width(), owner_layout.key_value_width(),
                            layout.query_width(), layout.sliding_window_size(), stream_.get());
                    } else if (rows <= kMaxGemmAttentionRows) {
                        launch_gqa_prefill_gemm(
                            gemm_->cublas().get(), workspace_.prefill_q_.data(),
                            owner->key_cache.data(), owner->value_cache.data(),
                            workspace_.prefill_op_output_.data(), workspace_.prefill_attn_scores_.data(),
                            workspace_.prefill_attn_probs_.data(), rows,
                            layout.query_heads, owner_layout.key_value_heads,
                            owner_layout.head_dim, layout.query_width(), owner_layout.key_value_width(),
                            layout.query_width(), layout.sliding_window_size(), stream_.get());
                    } else {
                        const int chunks = (rows + kPrefillAttnChunkTokens - 1) /
                            kPrefillAttnChunkTokens;
                        launch_gqa_prefill_segmented(
                            workspace_.prefill_q_.data(), owner->key_cache.data(),
                            owner->value_cache.data(), workspace_.prefill_op_output_.data(),
                            rows, layout.query_heads, owner_layout.key_value_heads,
                            owner_layout.head_dim, kPrefillAttnChunkTokens, chunks,
                            layout.sliding_window_size(),
                            workspace_.prefill_attn_partial_max_.data(),
                            workspace_.prefill_attn_partial_denom_.data(),
                            workspace_.prefill_attn_partial_accum_.data(), stream_.get());
                    }
                } else {
                    launch_gqa_prefill_strict(
                        workspace_.prefill_q_.data(), owner->key_cache.data(),
                        owner->value_cache.data(), workspace_.prefill_op_output_.data(),
                        rows, layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window_size(), stream_.get());
                }
            }
            prof.end(PrefillPhase::Attention, stream_.get());
            if (output_gate) {
                launch_sigmoid_multiply(workspace_.prefill_op_output_.data(),
                    workspace_.prefill_attention_gate_.data(),
                    rows * layout.query_width(), stream_.get());
            }

            prof.begin(stream_.get());
            linear(workspace_.prefill_op_output_.data(), *attention->out,
                   workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                   layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm ? 1.0f : 0.0f);
            launch_scale(workspace_.prefill_hidden_.data(), rows * resources_.program_.hidden,
                         semantics.residual.multiplier, stream_.get());
            prof.end(PrefillPhase::AttnOut, stream_.get());
            }
          },
          [&](ConvolutionLayer* convolution) {
            prof.begin(stream_.get());
            linear(workspace_.prefill_normed_.data(), *convolution->conv_in,
                   workspace_.prefill_conv_projected_.data(), rows,
                   3 * resources_.program_.hidden, resources_.program_.hidden);
            launch_conv_prefill(
                workspace_.prefill_conv_projected_.data(), convolution->conv_weight,
                convolution->conv_state.data(), workspace_.prefill_op_output_.data(),
                rows, resources_.program_.hidden, resources_.shape_.conv_cache,
                stream_.get());
            linear(workspace_.prefill_op_output_.data(), *convolution->conv_out,
                   workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                   resources_.program_.hidden, resources_.options_.fused_residuals ? 1.0f : 0.0f);
            prof.end(PrefillPhase::Conv, stream_.get());
          });

        if (common_layer.post_attention_norm) {
            launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.post_attention_norm,
                           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                           semantics.post_attention_norm.epsilon, stream_.get());
        }
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm ||
            !semantics.execute_feed_forward) {
            prof.begin(stream_.get());
            launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.prefill_residual_.data(),
                                rows * resources_.program_.hidden, stream_.get());
            prof.end(PrefillPhase::Other, stream_.get());
        }
        prof.begin(stream_.get());
        if (semantics.execute_feed_forward) run_mlp_prefill(common_layer, rows, layer_idx);
        if (std::binary_search(resources_.program_.norm_after_layers.begin(),
                               resources_.program_.norm_after_layers.end(), layer_idx)) {
            launch_rmsnorm(workspace_.prefill_hidden_.data(), resources_.final_norm_,
                           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                           resources_.program_.final_norm.epsilon, stream_.get());
        }
        prof.end(PrefillPhase::Mlp, stream_.get());
        ++layer_idx;
    }

    prof.begin(stream_.get());
    const __nv_bfloat16* last_hidden = workspace_.prefill_hidden_.data() +
        static_cast<size_t>(rows - 1) * resources_.program_.hidden;
    launch_rmsnorm(last_hidden, resources_.final_norm_, workspace_.normed_.data(),
                   1, resources_.program_.hidden, resources_.program_.final_norm.epsilon,
                   stream_.get());
    linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
           1, resources_.dims_.vocab_size, resources_.program_.hidden);
    if (resources_.program_.logits_divisor != 1.0f ||
        resources_.program_.logits_multiplier != 1.0f) {
    launch_scale(workspace_.logits_.data(), resources_.dims_.vocab_size,
                 resources_.program_.logits_multiplier /
                     resources_.program_.logits_divisor, stream_.get());
    if (resources_.program_.final_logit_softcap > 0.0f) {
        launch_tanh_softcap(workspace_.logits_.data(), resources_.dims_.vocab_size,
                            resources_.program_.final_logit_softcap, stream_.get());
    }
    }
    prof.end(PrefillPhase::Logits, stream_.get());

    session_.position_ = rows;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_,
                             sizeof(session_.position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    // See benchmark_decode(): runner shutdown bypasses static destructors on
    // Windows, so an enabled profile must be emitted before returning.
    prof.report();
    release_prefill_workspace();
    session_.phase_ = SessionPhase::Ready;
}

} // namespace celeg


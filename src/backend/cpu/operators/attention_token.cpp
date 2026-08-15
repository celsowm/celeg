#include "../detail/model_internal.hpp"
#include "attention.hpp"

#include <algorithm>

namespace celeg {

void execute_cpu_attention_token(
    CpuExecutionContext& execution,
    CpuAttentionStateView attention_state,
    size_t index,
    const CpuCompiledModel::AttentionWeights& attention,
    const CompiledLayerProgram& semantics,
    const std::array<int32_t, 3>& rope_position) {
    const AttentionSpec& layout =
        std::get<CompiledAttentionProgram>(semantics.mixer).semantics;
            if (layout.uses_external_memory()) {
                const int q_width = layout.query_width();
                execution.shared.linear.gemv(attention.q, execution.workspace.normed.data(), execution.workspace.qkv.data());
                float* q = execution.workspace.qkv.data();
                apply_cpu_attention_qk(layout, attention, q, nullptr,
                                       execution.session.position_value, rope_position);
                const auto memory_it = execution.shared.external_attention_memory.find(
                    layout.external_memory_slot());
                if (memory_it == execution.shared.external_attention_memory.end()) {
                    throw std::logic_error("external attention memory slot is not bound");
                }
                attention_state.run_external_attention(layout, *memory_it->second, q,
                                       execution.workspace.op_output.data(),
                                       attention.relative_bias);
                if (layout.output_gate.has_value()) {
                    execution.shared.linear.gemv(attention.gate,
                                                 execution.workspace.normed.data(),
                                                 execution.workspace.attention_gate.data());
                    apply_cpu_attention_output_gate(execution.workspace.op_output.data(),
                                         execution.workspace.attention_gate.data(),
                                         static_cast<size_t>(q_width));
                }
                execution.shared.linear.gemv(attention.out, execution.workspace.op_output.data(),
                                    execution.workspace.hidden.data());
            } else if (layout.uses_latent_state()) {
                const auto& latent = *layout.latent_state();
                const int content_width = layout.latent_query_content_width();
                const int rope_width = layout.latent_query_rope_width();
                float* query_content = execution.workspace.qkv.data();
                if (latent.factorized) {
                    execution.shared.linear.gemv(attention.latent_q_projection,
                        execution.workspace.normed.data(), execution.workspace.qkv.data());
                    cpu_rmsnorm_inplace(execution.workspace.qkv.data(), attention.latent_q_norm.data(),
                                        static_cast<size_t>(latent.query_rank),
                                        latent.query_latent_norm.epsilon);
                    execution.shared.linear.gemv(attention.latent_q_expansion,
                        execution.workspace.qkv.data(), execution.workspace.latent_projection.data());
                    query_content = execution.workspace.qkv.data();
                    const int query_stride = latent.nope_head_dim + latent.rope_head_dim;
                    const int expansion_stride = latent.nope_head_dim + latent.value_head_dim;
                    for (int head = 0; head < layout.query_heads; ++head) {
                        execution.shared.linear.gemv_transpose(
                            attention.latent_expansion,
                            execution.workspace.latent_projection.data() +
                                static_cast<size_t>(head * query_stride),
                            query_content + static_cast<size_t>(head * latent.latent_rank),
                            static_cast<size_t>(head * expansion_stride),
                            static_cast<size_t>(latent.nope_head_dim));
                    }
                } else {
                    execution.shared.linear.gemv(attention.q, execution.workspace.normed.data(),
                                        execution.workspace.qkv.data());
                }
                float* query_rope = rope_width == 0 ? nullptr : execution.workspace.latent_rope.data();
                if (query_rope && latent.factorized) {
                    const int stride = latent.nope_head_dim + latent.rope_head_dim;
                    for (int head = 0; head < layout.query_heads; ++head) {
                        std::copy(execution.workspace.latent_projection.data() +
                                      static_cast<size_t>(head * stride + latent.nope_head_dim),
                                  execution.workspace.latent_projection.data() +
                                      static_cast<size_t>((head + 1) * stride),
                                  query_rope + static_cast<size_t>(head * latent.rope_head_dim));
                    }
                } else if (query_rope) {
                    execution.shared.linear.gemv(attention.latent_q_rope, execution.workspace.normed.data(),
                                        query_rope);
                }
                if (latent.factorized) {
                    execution.shared.linear.gemv(attention.latent_k_projection,
                        execution.workspace.normed.data(), execution.workspace.latent_projection.data());
                    std::copy(execution.workspace.latent_projection.data(),
                              execution.workspace.latent_projection.data() + latent.latent_rank,
                              execution.workspace.latent_key.data());
                    cpu_rmsnorm_inplace(execution.workspace.latent_key.data(), attention.latent_k_norm.data(),
                                        static_cast<size_t>(latent.latent_rank),
                                        latent.key_latent_norm.epsilon);
                    std::copy(execution.workspace.latent_key.data(),
                              execution.workspace.latent_key.data() + latent.latent_rank,
                              execution.workspace.latent_value.data());
                    std::copy(execution.workspace.latent_projection.data() + latent.latent_rank,
                              execution.workspace.latent_projection.data() + latent.latent_rank + latent.rope_head_dim,
                              execution.workspace.latent_key_rope.data());
                } else {
                    execution.shared.linear.gemv(attention.k, execution.workspace.normed.data(),
                                        execution.workspace.latent_key.data());
                    execution.shared.linear.gemv(attention.v, execution.workspace.normed.data(),
                                        execution.workspace.latent_value.data());
                }
                float* key_rope = nullptr;
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    key_rope = execution.workspace.latent_key_rope.data();
                    if (!latent.factorized) {
                        execution.shared.linear.gemv(attention.latent_k_rope,
                            execution.workspace.normed.data(), key_rope);
                    }
                }
                apply_cpu_latent_attention_positions(
                    layout, query_rope, key_rope,
                    execution.session.position_value, rope_position);
                const int owner = execution.shared.layer_to_kv_owner.at(index);
                CpuCompiledModel::AttentionState& state = attention_state.state(static_cast<size_t>(owner));
                attention_state.store_latent(state, execution.session.position_value,
                             execution.workspace.latent_key.data(),
                             execution.workspace.latent_value.data(), key_rope);
                attention_state.run_latent_attention(state, layout, query_content, query_rope,
                                     execution.workspace.op_output.data(), execution.session.position_value + 1,
                                     execution.session.position_value, attention.relative_bias);
                const float* output_input = execution.workspace.op_output.data();
                if (latent.factorized) {
                    const int expansion_stride = latent.nope_head_dim + latent.value_head_dim;
                    for (int head = 0; head < layout.query_heads; ++head) {
                        execution.shared.linear.gemv_transpose(attention.latent_expansion,
                            execution.workspace.op_output.data() +
                                static_cast<size_t>(head * latent.latent_rank),
                            execution.workspace.latent_projection.data(),
                            static_cast<size_t>(head * expansion_stride + latent.nope_head_dim),
                            static_cast<size_t>(latent.value_head_dim));
                        std::copy_n(execution.workspace.latent_projection.data(),
                                    latent.value_head_dim,
                                    execution.workspace.latent_decompressed.data() +
                                        static_cast<size_t>(head * latent.value_head_dim));
                    }
                    output_input = execution.workspace.latent_decompressed.data();
                    execution.shared.linear.gemv(attention.gate, execution.workspace.normed.data(),
                        execution.workspace.attention_gate.data());
                    apply_cpu_attention_output_gate(execution.workspace.latent_decompressed.data(),
                        execution.workspace.attention_gate.data(),
                        static_cast<size_t>(layout.latent_output_width()),
                        layout.output_gate->granularity,
                        layout.query_heads, latent.value_head_dim);
                }
                execution.shared.linear.gemv(attention.out, output_input,
                                    execution.workspace.hidden.data());
            } else {
            const int q_width = layout.query_width();
            const int q_projection_width = layout.query_projection_width();
            const int kv_width = layout.key_value_width();
            execution.shared.linear.gemv(attention.q, execution.workspace.normed.data(), execution.workspace.qkv.data());
            float* q = execution.workspace.qkv.data();
            float* k = q + q_projection_width;
            float* v = k + kv_width;
            if (!attention.k.segments.empty()) {
                execution.shared.linear.gemv(attention.k, execution.workspace.normed.data(), k);
                execution.shared.linear.gemv(attention.v, execution.workspace.normed.data(), v);
            }
            apply_cpu_attention_qk(layout, attention, q, k,
                                   execution.session.position_value, rope_position);
            const int owner = execution.shared.layer_to_kv_owner.at(index);
            CpuCompiledModel::AttentionState& state = attention_state.state(static_cast<size_t>(owner));
            if (!attention.k.segments.empty()) {
                attention_state.store_kv(state, execution.session.position_value, k, v);
            }
            attention_state.run_attention(state, layout, q, execution.workspace.op_output.data(),
                          execution.session.position_value + 1, attention.relative_bias);
            apply_cpu_attention_output_transform(
                layout, execution.workspace.op_output.data(), v);
            if (layout.output_gate.has_value()) {
                const float* gate = nullptr;
                if (layout.output_gate->packed_with_query) {
                    gate = q + q_width;
                } else {
                    execution.shared.linear.gemv(attention.gate,
                                                 execution.workspace.normed.data(),
                                                 execution.workspace.attention_gate.data());
                    gate = execution.workspace.attention_gate.data();
                }
                    apply_cpu_attention_output_gate(execution.workspace.op_output.data(),
                                     gate,
                                     static_cast<size_t>(q_width));
            }
            execution.shared.linear.gemv(attention.out, execution.workspace.op_output.data(), execution.workspace.hidden.data());
            }
}

}

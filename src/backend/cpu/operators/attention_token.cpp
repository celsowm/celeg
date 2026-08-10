#include "../detail/model_internal.hpp"
#include "attention.hpp"

namespace celeg {

void execute_cpu_attention_token(
    CpuExecutionContext& execution,
    CpuCompiledModel& model,
    size_t index,
    const CpuCompiledModel::AttentionWeights& attention,
    const CompiledLayerProgram& semantics,
    const std::array<int32_t, 3>& rope_position) {
    const AttentionSpec& layout = semantics.attention.value();
            if (layout.uses_external_memory()) {
                const int q_width = layout.query_width();
                execution.shared.linear.gemv(attention.q, execution.workspace.normed.data(), execution.workspace.qkv.data());
                float* q = execution.workspace.qkv.data();
                apply_cpu_attention_qk(execution.shared.shape, layout, attention, q, nullptr,
                                       execution.session.position_value, rope_position);
                const auto memory_it = execution.shared.external_attention_memory.find(
                    layout.sources.memory_slot);
                if (memory_it == execution.shared.external_attention_memory.end()) {
                    throw std::logic_error("external attention memory slot is not bound");
                }
                model.run_external_attention(layout, *memory_it->second, q,
                                       execution.workspace.op_output.data(),
                                       attention.relative_bias);
                if (layout.output_gate.enabled()) {
                    execution.shared.linear.gemv(attention.gate,
                                                 execution.workspace.normed.data(),
                                                 execution.workspace.attention_gate.data());
                    apply_cpu_query_gate(execution.workspace.op_output.data(),
                                         execution.workspace.attention_gate.data(),
                                         static_cast<size_t>(q_width));
                }
                execution.shared.linear.gemv(attention.out, execution.workspace.op_output.data(),
                                    execution.workspace.hidden.data());
            } else if (layout.uses_latent_state()) {
                const auto& latent = *layout.latent_state();
                if (layout.output_gate.enabled()) {
                    throw std::invalid_argument("latent attention output gating is not supported");
                }
                const int content_width = layout.latent_query_content_width();
                const int rope_width = layout.latent_query_rope_width();
                execution.shared.linear.gemv(attention.q, execution.workspace.normed.data(),
                                    execution.workspace.qkv.data());
                float* query_content = execution.workspace.qkv.data();
                float* query_rope = rope_width == 0 ? nullptr : execution.workspace.latent_rope.data();
                if (query_rope) {
                    execution.shared.linear.gemv(attention.latent_q_rope, execution.workspace.normed.data(),
                                        query_rope);
                }
                execution.shared.linear.gemv(attention.k, execution.workspace.normed.data(),
                                    execution.workspace.latent_key.data());
                execution.shared.linear.gemv(attention.v, execution.workspace.normed.data(),
                                    execution.workspace.latent_value.data());
                float* key_rope = nullptr;
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    key_rope = execution.workspace.latent_key_rope.data();
                    execution.shared.linear.gemv(attention.latent_k_rope, execution.workspace.normed.data(), key_rope);
                }
                apply_cpu_latent_attention_positions(
                    execution.shared.shape, layout, query_rope, key_rope,
                    execution.session.position_value, rope_position);
                const int owner = execution.shared.layer_to_kv_owner.at(index);
                CpuCompiledModel::AttentionState& state = model.attention_state(static_cast<size_t>(owner));
                model.store_latent(state, execution.session.position_value, execution.workspace.latent_key.data(),
                             execution.workspace.latent_value.data(), key_rope);
                model.run_latent_attention(state, layout, query_content, query_rope,
                                     execution.workspace.op_output.data(), execution.session.position_value + 1,
                                     execution.session.position_value, attention.relative_bias);
                execution.shared.linear.gemv(attention.out, execution.workspace.op_output.data(),
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
            apply_cpu_attention_qk(execution.shared.shape, layout, attention, q, k,
                                   execution.session.position_value, rope_position);
            const int owner = execution.shared.layer_to_kv_owner.at(index);
            CpuCompiledModel::AttentionState& state = model.attention_state(static_cast<size_t>(owner));
            if (!attention.k.segments.empty()) {
                model.store_kv(state, execution.session.position_value, k, v);
            }
            model.run_attention(state, layout, q, execution.workspace.op_output.data(),
                          execution.session.position_value + 1, attention.relative_bias);
            apply_cpu_attention_output_transform(
                layout, execution.workspace.op_output.data(), v);
            if (layout.output_gate.enabled()) {
                const float* gate = nullptr;
                if (layout.output_gate.packed_with_query) {
                    gate = q + q_width;
                } else {
                    execution.shared.linear.gemv(attention.gate,
                                                 execution.workspace.normed.data(),
                                                 execution.workspace.attention_gate.data());
                    gate = execution.workspace.attention_gate.data();
                }
                apply_cpu_query_gate(execution.workspace.op_output.data(),
                                     gate,
                                     static_cast<size_t>(q_width));
            }
            execution.shared.linear.gemv(attention.out, execution.workspace.op_output.data(), execution.workspace.hidden.data());
            }
}

} // namespace celeg

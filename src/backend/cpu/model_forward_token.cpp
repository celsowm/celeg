#include "detail/model_internal.hpp"
#include "operators/attention.hpp"
#include "operators/feed_forward.hpp"
#include "operators/moe.hpp"
#include "operators/recurrent.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace celeg {

void CpuCompiledModel::forward_token(int32_t token, bool compute_logits,
                                     const PromptEmbedding* embeddings) {
    CpuExecutionContext execution{*shared, workspace_, session_};
    if (session_.position_value >= shared->max_context) {
        throw std::runtime_error("CPU context limit reached");
    }
    const float* raw_embedding = embeddings
        ? embeddings->at_position(static_cast<std::size_t>(session_.position_value))
        : nullptr;
    const std::array<int32_t, 3> rope_position =
        embeddings && embeddings->rope_at_position(static_cast<std::size_t>(session_.position_value))
            ? *embeddings->rope_at_position(static_cast<std::size_t>(session_.position_value))
            : session_.next_rope_position;
    if (raw_embedding) {
        if (embeddings->width != shared->shape.hidden) {
            throw std::invalid_argument("raw embedding width does not match model hidden size");
        }
        std::copy(raw_embedding, raw_embedding + shared->shape.hidden,
                  workspace_.hidden.begin());
    } else {
        shared->linear.embedding(shared->weight_store.embedding, token, workspace_.hidden.data());
    }
    if (!raw_embedding && shared->shape.numerical_policy.embedding_multiplier != 1.0f) {
        for (float& value : workspace_.hidden) value *= shared->shape.numerical_policy.embedding_multiplier;
    }
    if (shared->program.per_layer_input.enabled) {
        const PerLayerInputPlan& plan = shared->program.per_layer_input;
        const size_t packed = plan.packed_width;
        workspace_.per_layer_input.resize(packed);
        workspace_.per_layer_context.resize(packed);
        workspace_.per_layer_gate.resize(static_cast<size_t>(plan.input_size));
        shared->linear.embedding(shared->weight_store.per_layer_embedding,
                                 raw_embedding ? shared->shape.token_policy.pad_token_id : token,
                                 workspace_.per_layer_input.data());
        for (float& value : workspace_.per_layer_input) value *= plan.token_scale;
        shared->linear.gemv(shared->weight_store.per_layer_context_projection,
                            workspace_.hidden.data(), workspace_.per_layer_context.data());
        for (float& value : workspace_.per_layer_context) value *= plan.context_scale;
        for (int layer = 0; layer < plan.layer_count; ++layer) {
            float* context = workspace_.per_layer_context.data() +
                static_cast<size_t>(layer) * static_cast<size_t>(plan.input_size);
            cpu_rmsnorm_inplace(context,
                shared->weight_store.per_layer_projection_norm.data(),
                plan.input_size, plan.norm_epsilon);
            float* token_values = workspace_.per_layer_input.data() +
                static_cast<size_t>(layer) * static_cast<size_t>(plan.input_size);
            for (int d = 0; d < plan.input_size; ++d) {
                context[d] = (context[d] + token_values[d]) * plan.residual_scale;
            }
        }
    }
    for (size_t index = 0; index < shared->weight_store.layers.size(); ++index) {
        const WeightLayer& layer_program = shared->weight_store.layers[index];
        const CommonWeights& common = common_weights(index);
        std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.residual.begin());
        cpu_rmsnorm(workspace_.hidden.data(), common.operator_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
        const CompiledLayerProgram& semantics = shared->program.layers[index];
        if (!semantics.execute_feed_forward &&
            semantics.mixer == CompiledMixer::MlpOnly) {
            const auto& mlp = std::get<CpuCompiledModel::MlpOnlyWeights>(layer_program);
            execute_cpu_mlp_only_token(execution, index, mlp);
            continue;
        }
        if (const auto* gated_delta = gated_delta_net_operator(layer_program)) {
            execute_cpu_gated_delta_token(*this, index, *gated_delta);
        } else if (const auto* mamba = mamba2_operator(layer_program)) {
            execute_cpu_mamba2_token(*this, index, *mamba);
            continue;
        }
        else if (const auto* attention = attention_operator(layer_program)) {
            const AttentionSpec& layout = semantics.attention.value();
            if (layout.uses_external_memory()) {
                const int q_width = layout.query_width();
                shared->linear.gemv(attention->q, workspace_.normed.data(), workspace_.qkv.data());
                float* q = workspace_.qkv.data();
                apply_cpu_attention_qk(shared->shape, layout, *attention, q, nullptr,
                                       session_.position_value, rope_position);
                const auto memory_it = shared->external_attention_memory.find(
                    layout.sources.memory_slot);
                if (memory_it == shared->external_attention_memory.end()) {
                    throw std::logic_error("external attention memory slot is not bound");
                }
                run_external_attention(layout, *memory_it->second, q,
                                       workspace_.op_output.data(),
                                       attention->relative_bias);
                if (layout.query_gate) {
                    apply_cpu_query_gate(workspace_.op_output.data(), q + q_width,
                                         static_cast<size_t>(q_width));
                }
                shared->linear.gemv(attention->out, workspace_.op_output.data(),
                                    workspace_.hidden.data());
            } else if (layout.uses_latent_state()) {
                const auto& latent = *layout.latent_state();
                if (layout.query_gate) {
                    throw std::invalid_argument("latent attention query gating is not supported");
                }
                const int content_width = layout.latent_query_content_width();
                const int rope_width = layout.latent_query_rope_width();
                shared->linear.gemv(attention->q, workspace_.normed.data(),
                                    workspace_.qkv.data());
                float* query_content = workspace_.qkv.data();
                float* query_rope = rope_width == 0 ? nullptr : workspace_.latent_rope.data();
                if (query_rope) {
                    shared->linear.gemv(attention->latent_q_rope, workspace_.normed.data(),
                                        query_rope);
                }
                shared->linear.gemv(attention->k, workspace_.normed.data(),
                                    workspace_.latent_key.data());
                shared->linear.gemv(attention->v, workspace_.normed.data(),
                                    workspace_.latent_value.data());
                float* key_rope = nullptr;
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    key_rope = workspace_.latent_key_rope.data();
                    shared->linear.gemv(attention->latent_k_rope, workspace_.normed.data(), key_rope);
                }
                apply_cpu_latent_attention_positions(
                    shared->shape, layout, query_rope, key_rope,
                    session_.position_value, rope_position);
                const int owner = shared->layer_to_kv_owner.at(index);
                AttentionState& state = attention_state(static_cast<size_t>(owner));
                store_latent(state, session_.position_value, workspace_.latent_key.data(),
                             workspace_.latent_value.data(), key_rope);
                run_latent_attention(state, layout, query_content, query_rope,
                                     workspace_.op_output.data(), session_.position_value + 1,
                                     session_.position_value, attention->relative_bias);
                shared->linear.gemv(attention->out, workspace_.op_output.data(),
                                    workspace_.hidden.data());
            } else {
            const int q_width = layout.query_width();
            const int q_projection_width = layout.query_projection_width();
            const int kv_width = layout.key_value_width();
            shared->linear.gemv(attention->q, workspace_.normed.data(), workspace_.qkv.data());
            float* q = workspace_.qkv.data();
            float* k = q + q_projection_width;
            float* v = k + kv_width;
            if (!attention->k.segments.empty()) {
                shared->linear.gemv(attention->k, workspace_.normed.data(), k);
                shared->linear.gemv(attention->v, workspace_.normed.data(), v);
            }
            apply_cpu_attention_qk(shared->shape, layout, *attention, q, k,
                                   session_.position_value, rope_position);
            const int owner = shared->layer_to_kv_owner.at(index);
            AttentionState& state = attention_state(static_cast<size_t>(owner));
            if (!attention->k.segments.empty()) {
                store_kv(state, session_.position_value, k, v);
            }
            run_attention(state, layout, q, workspace_.op_output.data(),
                          session_.position_value + 1, attention->relative_bias);
            if (layout.query_gate) {
                apply_cpu_query_gate(workspace_.op_output.data(), q + q_width,
                                     static_cast<size_t>(q_width));
            }
            shared->linear.gemv(attention->out, workspace_.op_output.data(), workspace_.hidden.data());
            }
        } else {
            const auto* convolution = convolution_operator(layer_program);
            if (!convolution) throw std::logic_error("CPU layer has no operator");
            execute_cpu_short_convolution_token(*this, index, *convolution);
        }
        if (shared->shape.numerical_policy.residual_multiplier != 1.0f) {
            for (float& value : workspace_.hidden) value *= shared->shape.numerical_policy.residual_multiplier;
        }
        if (shared->shape.has_split_attention_norms) {
            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.post_attention_norm.data(),
                                shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
        }
        cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->shape.hidden);

        // Mixer-only layers do not have the generic post-attention
        // normalization and dense FFN.
        if (!semantics.execute_feed_forward) continue;

        cpu_rmsnorm(workspace_.hidden.data(), common.ffn_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.numerical_policy.norm_eps);

        if (const auto* moe = std::get_if<MoeWeights>(&layer_program)) {
            const MoeLayerProgram& moe_semantics = shared->program.layers[index].moe.value();
            execute_cpu_moe_token(*this, index, *moe, moe_semantics);
            if (shared->shape.numerical_policy.residual_multiplier != 1.0f) {
                for (float& value : workspace_.mlp_output) value *= shared->shape.numerical_policy.residual_multiplier;
            }
            if (shared->shape.has_split_attention_norms) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.post_feed_forward_norm.data(),
                                    shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->shape.hidden);
        } else {
            execute_cpu_dense_feed_forward_token(execution, index, common);
            if (shared->shape.numerical_policy.residual_multiplier != 1.0f) {
                for (float& value : workspace_.mlp_output) value *= shared->shape.numerical_policy.residual_multiplier;
            }
            if (shared->shape.has_split_attention_norms) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.post_feed_forward_norm.data(),
                                    shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->shape.hidden);
        }
        if (shared->program.per_layer_input.enabled) {
            const PerLayerInputPlan& plan = shared->program.per_layer_input;
            std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.residual.begin());
            shared->linear.gemv(common.per_layer_input_gate, workspace_.hidden.data(),
                                workspace_.per_layer_gate.data());
            cpu_gelu_tanh(workspace_.per_layer_gate.data(),
                          static_cast<size_t>(plan.input_size));
            const float* layer_input = workspace_.per_layer_context.data() +
                index * static_cast<size_t>(plan.input_size);
            for (int d = 0; d < plan.input_size; ++d) {
                workspace_.per_layer_gate[static_cast<size_t>(d)] *= layer_input[d];
            }
            shared->linear.gemv(common.per_layer_projection, workspace_.per_layer_gate.data(),
                                workspace_.hidden.data());
            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.per_layer_input_norm.data(),
                                shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
            if (common.layer_scalar != 1.0f) {
                for (float& value : workspace_.hidden) value *= common.layer_scalar;
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->shape.hidden);
        }
        if (std::binary_search(shared->program.norm_after_layers.begin(),
                               shared->program.norm_after_layers.end(),
                               static_cast<int>(index))) {
            cpu_rmsnorm_inplace(workspace_.hidden.data(), shared->weight_store.final_norm.data(),
                                shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
        }
    }
    if (compute_logits) {
        cpu_rmsnorm(workspace_.hidden.data(), shared->weight_store.final_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->weight_store.embedding :
                            shared->weight_store.lm_head, workspace_.normed.data(), workspace_.logits.data());
        if (shared->shape.numerical_policy.logits_divisor != 1.0f) {
            for (float& value : workspace_.logits) value /= shared->shape.numerical_policy.logits_divisor;
        }
        if (shared->final_logit_softcap > 0.0f) {
            for (float& value : workspace_.logits) {
                value = std::tanh(value / shared->final_logit_softcap) *
                    shared->final_logit_softcap;
            }
        }
    }
    ++session_.position_value;
    if (embeddings && embeddings->has_rope_positions) {
        if (const auto* next = embeddings->rope_at_position(
                static_cast<std::size_t>(session_.position_value))) {
            session_.next_rope_position = *next;
        } else {
            session_.next_rope_position = embeddings->next_rope_position;
        }
    } else {
        for (int32_t& value : session_.next_rope_position) ++value;
    }
}

} // namespace celeg

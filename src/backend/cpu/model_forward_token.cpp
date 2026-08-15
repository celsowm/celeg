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
    CpuRecurrentStateView recurrent_state{session_};
    CpuAttentionStateView attention_state{*this};
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
        if (embeddings->width != shared->program.hidden) {
            throw std::invalid_argument("raw embedding width does not match model hidden size");
        }
        std::copy(raw_embedding, raw_embedding + shared->program.hidden,
                  workspace_.hidden.begin());
    } else {
        shared->linear.embedding(shared->weight_store.embedding, token, workspace_.hidden.data());
    }
    if (shared->program.embedding_transform.multiplier != 1.0f) {
        for (float& value : workspace_.hidden) value *= shared->program.embedding_transform.multiplier;
    }
    if (shared->program.embedding_transform.post_norm) {
        cpu_rmsnorm_inplace(workspace_.hidden.data(), shared->weight_store.embedding_norm.data(),
                            shared->program.hidden,
                            shared->program.embedding_transform.post_norm->epsilon);
    }
    if (shared->program.per_layer_input.enabled) {
        const PerLayerInputPlan& plan = shared->program.per_layer_input;
        const size_t packed = plan.packed_width;
        workspace_.per_layer_input.resize(packed);
        workspace_.per_layer_context.resize(packed);
        workspace_.per_layer_gate.resize(static_cast<size_t>(plan.input_size));
        shared->linear.embedding(shared->weight_store.per_layer_embedding,
                                 raw_embedding ? shared->dims.token_policy.pad_token_id : token,
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
        const CompiledLayerProgram& semantics = shared->program.layers[index];
        std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.residual.begin());
        cpu_rmsnorm(workspace_.hidden.data(), common.operator_norm.data(), workspace_.normed.data(),
                    shared->program.hidden, semantics.operator_norm.epsilon);
        bool mixer_owns_layer = false;
        visit_operator_weights(layer_program,
          [&](const CpuCompiledModel::AttentionWeights* attention) {
            execute_cpu_attention_token(execution, attention_state, index, *attention, semantics,
                                        rope_position);
          },
          [&](const CpuCompiledModel::ConvolutionWeights* convolution) {
            execute_cpu_short_convolution_token(execution, recurrent_state, index, *convolution);
          },
          [&](const CpuCompiledModel::GatedDeltaNetWeights* gated_delta) {
            execute_cpu_gated_delta_token(execution, recurrent_state, index, *gated_delta);
          },
          [&](const CpuCompiledModel::Mamba2Weights* mamba) {
            execute_cpu_mamba2_token(execution, recurrent_state, index, *mamba);
            mixer_owns_layer = true;
          },
          [&](const CpuCompiledModel::MlpOnlyWeights* mlp) {
            if (!std::holds_alternative<std::monostate>(semantics.feed_forward)) {
                throw std::logic_error(
                    "CPU MLP-only layer cannot also run a feed-forward block");
            }
            execute_cpu_mlp_only_token(execution, index, *mlp);
            mixer_owns_layer = true;
          });
        if (mixer_owns_layer) continue;
        if (semantics.residual.multiplier != 1.0f) {
            for (float& value : workspace_.hidden) value *= semantics.residual.multiplier;
        }
        if (semantics.post_attention_norm.has_value()) {
            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.post_attention_norm.data(),
                                shared->program.hidden, semantics.post_attention_norm->epsilon);
        }
        cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->program.hidden);

        if (std::holds_alternative<std::monostate>(semantics.feed_forward)) continue;

        cpu_rmsnorm(workspace_.hidden.data(), common.ffn_norm.data(), workspace_.normed.data(),
                    shared->program.hidden, semantics.feed_forward_norm->epsilon);

        if (const auto* moe = std::get_if<MoeWeights>(&layer_program)) {
            const MoeLayerProgram& moe_semantics =
                std::get<MoeLayerProgram>(shared->program.layers[index].feed_forward);
            execute_cpu_moe_token(execution, index, *moe, moe_semantics);
            if (semantics.residual.multiplier != 1.0f) {
                for (float& value : workspace_.mlp_output) value *= semantics.residual.multiplier;
            }
            if (semantics.post_feed_forward_norm.has_value()) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.post_feed_forward_norm.data(),
                                    shared->program.hidden, semantics.post_feed_forward_norm->epsilon);
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->program.hidden);
        } else {
            execute_cpu_dense_feed_forward_token(execution, index, common);
            if (semantics.residual.multiplier != 1.0f) {
                for (float& value : workspace_.mlp_output) value *= semantics.residual.multiplier;
            }
            if (semantics.post_feed_forward_norm.has_value()) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.post_feed_forward_norm.data(),
                                    shared->program.hidden, semantics.post_feed_forward_norm->epsilon);
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->program.hidden);
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
                                shared->program.hidden, shared->program.per_layer_input.norm_epsilon);
            if (common.layer_scalar != 1.0f) {
                for (float& value : workspace_.hidden) value *= common.layer_scalar;
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->program.hidden);
        }
        if (std::binary_search(shared->program.norm_after_layers.begin(),
                               shared->program.norm_after_layers.end(),
                               static_cast<int>(index))) {
            cpu_rmsnorm_inplace(workspace_.hidden.data(), shared->weight_store.final_norm.data(),
                                shared->program.hidden, shared->program.final_norm.epsilon);
        }
    }
    if (compute_logits) {
        cpu_rmsnorm(workspace_.hidden.data(), shared->weight_store.final_norm.data(), workspace_.normed.data(),
                    shared->program.hidden, shared->program.final_norm.epsilon);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->weight_store.embedding :
                            shared->weight_store.lm_head, workspace_.normed.data(), workspace_.logits.data());
        if (shared->program.logits_multiplier != 1.0f) {
            for (float& value : workspace_.logits) value *= shared->program.logits_multiplier;
        }
        if (shared->program.logits_divisor != 1.0f) {
            for (float& value : workspace_.logits) value /= shared->program.logits_divisor;
        }
        if (shared->program.final_logit_softcap > 0.0f) {
            for (float& value : workspace_.logits) {
                value = std::tanh(value / shared->program.final_logit_softcap) *
                    shared->program.final_logit_softcap;
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

}

#include "detail/model_internal.hpp"
#include "operators/attention.hpp"
#include "operators/feed_forward.hpp"
#include "operators/moe.hpp"
#include "operators/recurrent.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <memory>
#include <stdexcept>

namespace celeg {

namespace {
using Clock = std::chrono::steady_clock;
double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <typename Body>
void parallel_rows(CpuThreadPool& pool, size_t rows, const Body& body) {
    const size_t grain = std::max<size_t>(1, rows / std::max<size_t>(1, pool.size() * 4));
    pool.parallel_for(0, rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) body(row);
    });
}
} // namespace

void CpuCompiledModel::forward_token(int32_t token, bool compute_logits,
                                     const PromptEmbedding* embeddings) {
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
            execute_cpu_mlp_only_token(*this, index, mlp);
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
                          session_.position_value + 1);
            if (layout.query_gate) {
                apply_cpu_query_gate(workspace_.op_output.data(), q + q_width,
                                     static_cast<size_t>(q_width));
            }
            shared->linear.gemv(attention->out, workspace_.op_output.data(), workspace_.hidden.data());
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
            execute_cpu_dense_feed_forward_token(*this, index, common);
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

void CpuCompiledModel::forward_chunk(std::span<const int32_t> tokens,
                                     bool compute_logits,
                                     const PromptEmbedding* embeddings) {
    if (tokens.empty()) return;
    if (session_.position_value < 0 ||
        tokens.size() > static_cast<size_t>(shared->max_context - session_.position_value)) {
        throw std::runtime_error("CPU chunked prefill exceeds context limit");
    }
    if (embeddings && embeddings->width != shared->shape.hidden) {
        throw std::invalid_argument("raw embedding width does not match model hidden size");
    }
    const bool has_sequential_only_layer = std::any_of(
        shared->program.layers.begin(), shared->program.layers.end(),
        [](const CompiledLayerProgram& layer) {
            return layer.mixer == CompiledMixer::Mamba2 ||
                   layer.mixer == CompiledMixer::MlpOnly;
        });
    if (has_sequential_only_layer) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            forward_token(tokens[i], compute_logits && i + 1 == tokens.size(), embeddings);
        }
        return;
    }

    const size_t rows = tokens.size();
    const int base_position = session_.position_value;
    const RuntimeTopology& shape = shared->shape;
    const size_t hidden = static_cast<size_t>(shape.hidden);
    workspace_.ensure_chunk(rows, shape);

    auto scale = [&](std::vector<float>& values, size_t count, float factor) {
        if (factor == 1.0f) return;
        parallel_rows(shared->pool, rows, [&](size_t row) {
            float* data = values.data() + row * (count / rows);
            for (size_t i = 0; i < count / rows; ++i) data[i] *= factor;
        });
    };
    auto rmsnorm_rows = [&](const float* input, const std::vector<float>& weight,
                            float* output, size_t width) {
        parallel_rows(shared->pool, rows, [&](size_t row) {
            cpu_rmsnorm(input + row * width, weight.data(), output + row * width,
                        width, shape.numerical_policy.norm_eps);
        });
    };
    auto rmsnorm_rows_inplace = [&](float* data, const std::vector<float>& weight,
                                    size_t width) {
        parallel_rows(shared->pool, rows, [&](size_t row) {
            cpu_rmsnorm_inplace(data + row * width, weight.data(), width,
                                shape.numerical_policy.norm_eps);
        });
    };
    auto residual_rows = [&](float* data, const float* residual, size_t width) {
        parallel_rows(shared->pool, rows, [&](size_t row) {
            cpu_residual_add(data + row * width, residual + row * width, width);
        });
    };

    auto linear_started = Clock::now();
    parallel_rows(shared->pool, rows, [&](size_t row) {
        const float* raw = embeddings
            ? embeddings->at_position(static_cast<size_t>(base_position) + row) : nullptr;
        float* destination = workspace_.chunk_hidden.data() + row * hidden;
        if (raw) {
            std::copy(raw, raw + hidden, destination);
        } else {
            shared->linear.embedding(shared->weight_store.embedding, tokens[row], destination);
            if (shape.numerical_policy.embedding_multiplier != 1.0f) {
                for (size_t d = 0; d < hidden; ++d) {
                    destination[d] *= shape.numerical_policy.embedding_multiplier;
                }
            }
        }
    });
    session_.prefill_profile.linear_ms += milliseconds_since(linear_started);

    const PerLayerInputPlan& input_plan = shared->program.per_layer_input;
    if (input_plan.enabled) {
        const size_t packed = input_plan.packed_width;
        const size_t input_size = static_cast<size_t>(input_plan.input_size);
        workspace_.per_layer_input.resize(rows * packed);
        workspace_.per_layer_context.resize(rows * packed);
        workspace_.per_layer_gate.resize(rows * input_size);
        parallel_rows(shared->pool, rows, [&](size_t row) {
            const int32_t embedding_token = embeddings &&
                    embeddings->at_position(static_cast<size_t>(base_position) + row)
                ? shape.token_policy.pad_token_id : tokens[row];
            float* destination = workspace_.per_layer_input.data() + row * packed;
            shared->linear.embedding(shared->weight_store.per_layer_embedding,
                                     embedding_token, destination);
            for (size_t d = 0; d < packed; ++d) destination[d] *= input_plan.token_scale;
        });
        linear_started = Clock::now();
        shared->linear.gemm(shared->weight_store.per_layer_context_projection,
                            workspace_.chunk_hidden.data(), workspace_.per_layer_context.data(), rows);
        session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
        parallel_rows(shared->pool, rows, [&](size_t row) {
            float* context = workspace_.per_layer_context.data() + row * packed;
            const float* token_values = workspace_.per_layer_input.data() + row * packed;
            for (int layer = 0; layer < input_plan.layer_count; ++layer) {
                float* values = context + static_cast<size_t>(layer) * input_size;
                if (input_plan.context_scale != 1.0f) {
                    for (size_t d = 0; d < input_size; ++d) {
                        values[d] *= input_plan.context_scale;
                }
                }
                cpu_rmsnorm_inplace(values,
                                    shared->weight_store.per_layer_projection_norm.data(),
                                    input_size, input_plan.norm_epsilon);
                const float* token_input = token_values + static_cast<size_t>(layer) * input_size;
                for (size_t d = 0; d < input_size; ++d) {
                    values[d] = (values[d] + token_input[d]) * input_plan.residual_scale;
                }
            }
        });
    }

    for (size_t index = 0; index < shared->weight_store.layers.size(); ++index) {
        const WeightLayer& layer_program = shared->weight_store.layers[index];
        const CommonWeights& common = common_weights(index);
        const CompiledLayerProgram& semantics = shared->program.layers[index];
        std::copy(workspace_.chunk_hidden.begin(), workspace_.chunk_hidden.end(),
                  workspace_.chunk_residual.begin());
        rmsnorm_rows(workspace_.chunk_hidden.data(), common.operator_norm,
                     workspace_.chunk_normed.data(), hidden);
        bool normed_q8_ready = false;
        auto layer_gemm = [&](const CpuLinearWeight& weight, const float* input,
                              float* output, float beta = 0.0f) {
            const bool cacheable = weight.gguf_native() && weight.cols == hidden &&
                input == workspace_.chunk_normed.data();
            if (cacheable) {
                if (!normed_q8_ready) {
                    shared->linear.prepare_gguf_activation(
                        input, rows, hidden, workspace_.chunk_q8);
                    normed_q8_ready = true;
                }
                shared->linear.gemm_gguf(workspace_.chunk_q8, weight, output,
                                         rows, beta);
            } else {
                shared->linear.gemm(weight, input, output, rows, beta);
            }
        };

        if (const auto* gated_delta = gated_delta_net_operator(layer_program)) {
            execute_cpu_gated_delta_chunk(*this, index, *gated_delta, rows,
                                          normed_q8_ready);
        } else if (const auto* attention = attention_operator(layer_program)) {
            const AttentionSpec& layout = semantics.attention.value();
            const size_t q_width = static_cast<size_t>(layout.query_width());
            const size_t q_projection_width = static_cast<size_t>(layout.query_projection_width());
            const size_t kv_width = static_cast<size_t>(layout.key_value_width());
            linear_started = Clock::now();
            layer_gemm(attention->q, workspace_.chunk_normed.data(),
                       workspace_.chunk_qkv.data());
            if (!attention->k.segments.empty()) {
                layer_gemm(attention->k, workspace_.chunk_normed.data(),
                           workspace_.chunk_op.data());
                layer_gemm(attention->v, workspace_.chunk_normed.data(),
                           workspace_.chunk_conv.data());
            }
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            parallel_rows(shared->pool, rows, [&](size_t row) {
                float* projected_q = workspace_.chunk_qkv.data() + row * q_projection_width;
                float* q = projected_q;
                float* k = workspace_.chunk_op.data() + row * kv_width;
                const int position = base_position + static_cast<int>(row);
                const auto* explicit_rope = embeddings
                    ? embeddings->rope_at_position(static_cast<size_t>(position)) : nullptr;
                const std::array<int32_t, 3> scalar_rope = {
                    position, position, position};
                const auto& rope_position = explicit_rope ? *explicit_rope : scalar_rope;
                apply_cpu_attention_qk(shape, layout, *attention, q, k, position,
                                       rope_position);
                if (layout.query_gate) {
                    std::copy_n(projected_q + q_width, q_width,
                                workspace_.chunk_attention_gate.data() + row * q_width);
                    std::copy_n(q, q_width,
                                workspace_.chunk_qkv.data() + row * q_width);
                }
            });
            const int owner = shared->layer_to_kv_owner.at(index);
            AttentionState& state = attention_state(static_cast<size_t>(owner));
            if (!attention->k.segments.empty()) {
                for (size_t row = 0; row < rows; ++row) {
                    store_kv(state, base_position + static_cast<int>(row),
                             workspace_.chunk_op.data() + row * kv_width,
                             workspace_.chunk_conv.data() + row * kv_width);
                }
            }
            auto attention_started = Clock::now();
            const CpuKvPagePool& pool = *shared->kv_pools.at(state.pool_index);
            cpu_gqa_prefill_paged(workspace_.chunk_qkv.data(), rows, q_width, pool,
                                  state.pages, workspace_.chunk_op.data(), base_position,
                                  layout.query_heads, layout.key_value_heads, layout.head_dim,
                                  shared->pool,
                                  layout.mask == AttentionMaskKind::SlidingCausal
                                      ? layout.sliding_window : 0);
            if (layout.query_gate) {
                for (size_t row = 0; row < rows; ++row) {
                    const float* gate = workspace_.chunk_attention_gate.data() + row * q_width;
                    float* output = workspace_.chunk_op.data() + row * q_width;
                    apply_cpu_query_gate(output, gate, q_width);
                }
            }
            session_.prefill_profile.attention_ms += milliseconds_since(attention_started);
            linear_started = Clock::now();
            layer_gemm(attention->out, workspace_.chunk_op.data(),
                       workspace_.chunk_hidden.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
        } else {
            const auto* convolution = convolution_operator(layer_program);
            if (!convolution) throw std::logic_error("CPU layer has no operator");
            execute_cpu_short_convolution_chunk(*this, index, *convolution, rows,
                                                normed_q8_ready);
        }

        if (shape.numerical_policy.residual_multiplier != 1.0f) {
            scale(workspace_.chunk_hidden, rows * hidden,
                  shape.numerical_policy.residual_multiplier);
        }
        if (shape.has_split_attention_norms) {
            rmsnorm_rows_inplace(workspace_.chunk_hidden.data(), common.post_attention_norm, hidden);
        }
        residual_rows(workspace_.chunk_hidden.data(), workspace_.chunk_residual.data(), hidden);
        rmsnorm_rows(workspace_.chunk_hidden.data(), common.ffn_norm,
                     workspace_.chunk_normed.data(), hidden);
        normed_q8_ready = false;

        if (const auto* moe = std::get_if<MoeWeights>(&layer_program)) {
            const MoeLayerProgram& moe_semantics = shared->program.layers[index].moe.value();
            execute_cpu_moe_chunk(*this, index, *moe, moe_semantics, rows,
                                  normed_q8_ready);
        } else {
            execute_cpu_dense_feed_forward_chunk(*this, index, common, rows,
                                                 normed_q8_ready);
        }
        if (shape.numerical_policy.residual_multiplier != 1.0f) {
            scale(workspace_.chunk_mlp, rows * hidden,
                  shape.numerical_policy.residual_multiplier);
        }
        if (shape.has_split_attention_norms) {
            rmsnorm_rows_inplace(workspace_.chunk_mlp.data(), common.post_feed_forward_norm, hidden);
        }
        residual_rows(workspace_.chunk_hidden.data(), workspace_.chunk_mlp.data(), hidden);

        if (input_plan.enabled) {
            std::copy(workspace_.chunk_hidden.begin(), workspace_.chunk_hidden.end(),
                      workspace_.chunk_residual.begin());
            linear_started = Clock::now();
            shared->linear.gemm(common.per_layer_input_gate, workspace_.chunk_hidden.data(),
                                workspace_.per_layer_gate.data(), rows);
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            const size_t input_size = static_cast<size_t>(input_plan.input_size);
            parallel_rows(shared->pool, rows, [&](size_t row) {
                float* gate = workspace_.per_layer_gate.data() + row * input_size;
                cpu_gelu_tanh(gate, input_size);
                const float* context = workspace_.per_layer_context.data() +
                    row * input_plan.packed_width + index * input_size;
                for (size_t d = 0; d < input_size; ++d) gate[d] *= context[d];
            });
            linear_started = Clock::now();
            shared->linear.gemm(common.per_layer_projection, workspace_.per_layer_gate.data(),
                                workspace_.chunk_hidden.data(), rows);
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            rmsnorm_rows_inplace(workspace_.chunk_hidden.data(), common.per_layer_input_norm, hidden);
            if (common.layer_scalar != 1.0f) {
                scale(workspace_.chunk_hidden, rows * hidden, common.layer_scalar);
            }
            residual_rows(workspace_.chunk_hidden.data(), workspace_.chunk_residual.data(), hidden);
        }
    }

    if (compute_logits) {
        const float* last_hidden = workspace_.chunk_hidden.data() + (rows - 1) * hidden;
        cpu_rmsnorm(last_hidden, shared->weight_store.final_norm.data(), workspace_.final_normed.data(),
                    hidden, shape.numerical_policy.norm_eps);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->weight_store.embedding :
                            shared->weight_store.lm_head, workspace_.final_normed.data(),
                            workspace_.logits.data());
        if (shape.numerical_policy.logits_divisor != 1.0f) {
            for (float& value : workspace_.logits) value /= shape.numerical_policy.logits_divisor;
        }
        if (shared->final_logit_softcap > 0.0f) {
            for (float& value : workspace_.logits) {
                value = std::tanh(value / shared->final_logit_softcap) * shared->final_logit_softcap;
            }
        }
    }
    session_.position_value += static_cast<int>(rows);
    if (embeddings && embeddings->has_rope_positions) {
        if (const auto* next = embeddings->rope_at_position(
                static_cast<size_t>(session_.position_value))) {
            session_.next_rope_position = *next;
        } else {
            session_.next_rope_position = embeddings->next_rope_position;
        }
    } else {
        for (int32_t& value : session_.next_rope_position) {
            value += static_cast<int32_t>(rows);
        }
    }
}

} // namespace celeg

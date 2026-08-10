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
            return layer.chunk_capability != CompiledChunkCapability::Native;
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
    CpuExecutionContext execution{*shared, workspace_, session_};
    workspace_.ensure_chunk(rows, shared->workspace_plan);

    auto scale = [&](std::vector<float>& values, size_t count, float factor) {
        if (factor == 1.0f) return;
        parallel_rows(shared->pool, rows, [&](size_t row) {
            float* data = values.data() + row * (count / rows);
            for (size_t i = 0; i < count / rows; ++i) data[i] *= factor;
        });
    };
    auto rmsnorm_rows = [&](const float* input, const std::vector<float>& weight,
                            float* output, size_t width, float epsilon = -1.0f) {
        if (epsilon < 0.0f) epsilon = shape.numerical_policy.norm_eps;
        parallel_rows(shared->pool, rows, [&](size_t row) {
            cpu_rmsnorm(input + row * width, weight.data(), output + row * width,
                        width, epsilon);
        });
    };
    auto rmsnorm_rows_inplace = [&](float* data, const std::vector<float>& weight,
                                    size_t width, float epsilon = -1.0f) {
        if (epsilon < 0.0f) epsilon = shape.numerical_policy.norm_eps;
        parallel_rows(shared->pool, rows, [&](size_t row) {
            cpu_rmsnorm_inplace(data + row * width, weight.data(), width,
                                epsilon);
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
            if (shared->program.embedding_transform.multiplier != 1.0f) {
                for (size_t d = 0; d < hidden; ++d) {
                    destination[d] *= shared->program.embedding_transform.multiplier;
                }
            }
        }
    });
    session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
    if (shared->program.embedding_transform.post_norm) {
        rmsnorm_rows(workspace_.chunk_hidden.data(), shared->weight_store.embedding_norm,
                     workspace_.chunk_hidden.data(), hidden,
                     shared->program.embedding_transform.post_norm->epsilon);
    }

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
                     workspace_.chunk_normed.data(), hidden,
                     semantics.operator_norm.epsilon);
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
            if (layout.uses_external_memory()) {
                const size_t q_width = static_cast<size_t>(layout.query_width());
                linear_started = Clock::now();
                layer_gemm(attention->q, workspace_.chunk_normed.data(),
                           workspace_.chunk_qkv.data());
                session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
                parallel_rows(shared->pool, rows, [&](size_t row) {
                    const int position = base_position + static_cast<int>(row);
                    const auto* explicit_rope = embeddings
                        ? embeddings->rope_at_position(static_cast<size_t>(position)) : nullptr;
                    const std::array<int32_t, 3> scalar_rope = {
                        position, position, position};
                    const auto& rope_position = explicit_rope ? *explicit_rope : scalar_rope;
                    apply_cpu_attention_qk(
                        shape, layout, *attention,
                        workspace_.chunk_qkv.data() + row * layout.query_projection_width(),
                        nullptr, position, rope_position);
                });
                const auto memory_it = shared->external_attention_memory.find(
                    layout.sources.memory_slot);
                if (memory_it == shared->external_attention_memory.end()) {
                    throw std::logic_error("external attention memory slot is not bound");
                }
                const auto& memory = *memory_it->second;
                parallel_rows(shared->pool, rows, [&](size_t row) {
                    run_external_attention(
                        layout, memory,
                        workspace_.chunk_qkv.data() + row * layout.query_projection_width(),
                        workspace_.chunk_op.data() + row * q_width,
                        attention->relative_bias);
                });
                if (layout.output_gate.enabled()) {
                    for (size_t row = 0; row < rows; ++row) {
                        const float* gate = nullptr;
                        if (layout.output_gate.packed_with_query) {
                            gate = workspace_.chunk_qkv.data() + row * layout.query_projection_width() + q_width;
                        } else {
                            if (row == 0) {
                                linear_started = Clock::now();
                                layer_gemm(attention->gate, workspace_.chunk_normed.data(),
                                           workspace_.chunk_attention_gate.data());
                                session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
                            }
                            gate = workspace_.chunk_attention_gate.data() + row * q_width;
                        }
                        apply_cpu_attention_output_gate(workspace_.chunk_op.data() + row * q_width,
                                             gate,
                                             q_width);
                    }
                }
                linear_started = Clock::now();
                layer_gemm(attention->out, workspace_.chunk_op.data(),
                           workspace_.chunk_hidden.data());
                session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            } else if (layout.uses_latent_state()) {
                const auto& latent = *layout.latent_state();
                if (layout.output_gate.enabled()) {
                    throw std::invalid_argument("latent attention query gating is not supported");
                }
                const size_t content_width = static_cast<size_t>(layout.latent_query_content_width());
                const size_t rope_width = static_cast<size_t>(layout.latent_query_rope_width());
                linear_started = Clock::now();
                layer_gemm(attention->q, workspace_.chunk_normed.data(),
                           workspace_.chunk_qkv.data());
                if (rope_width != 0) {
                    layer_gemm(attention->latent_q_rope, workspace_.chunk_normed.data(),
                               workspace_.chunk_latent_rope.data());
                }
                layer_gemm(attention->k, workspace_.chunk_normed.data(),
                           workspace_.chunk_latent_key.data());
                layer_gemm(attention->v, workspace_.chunk_normed.data(),
                           workspace_.chunk_latent_value.data());
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    layer_gemm(attention->latent_k_rope, workspace_.chunk_normed.data(),
                               workspace_.chunk_latent_key_rope.data());
                }
                session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
                parallel_rows(shared->pool, rows, [&](size_t row) {
                    const int position = base_position + static_cast<int>(row);
                    const auto* explicit_rope = embeddings
                        ? embeddings->rope_at_position(static_cast<size_t>(position)) : nullptr;
                    const std::array<int32_t, 3> scalar_rope = {
                        position, position, position};
                    const auto& rope_position = explicit_rope ? *explicit_rope : scalar_rope;
                    float* query_rope = rope_width == 0 ? nullptr :
                        workspace_.chunk_latent_rope.data() + row * rope_width;
                    float* key_rope = (latent.decoupled_rope && latent.rope_head_dim != 0)
                        ? workspace_.chunk_latent_key_rope.data() +
                            row * static_cast<size_t>(latent.rope_head_dim) : nullptr;
                    apply_cpu_latent_attention_positions(shape, layout, query_rope, key_rope,
                                                         position, rope_position);
                });
                const int owner = shared->layer_to_kv_owner.at(index);
                AttentionState& state = attention_state(static_cast<size_t>(owner));
                for (size_t row = 0; row < rows; ++row) {
                    store_latent(state, base_position + static_cast<int>(row),
                                 workspace_.chunk_latent_key.data() + row * latent.latent_rank,
                                 workspace_.chunk_latent_value.data() + row * latent.latent_rank,
                                 latent.decoupled_rope && latent.rope_head_dim != 0
                                     ? workspace_.chunk_latent_key_rope.data() +
                                         row * static_cast<size_t>(latent.rope_head_dim)
                                     : nullptr);
                }
                auto attention_started = Clock::now();
                const int committed_length = base_position + static_cast<int>(rows);
                parallel_rows(shared->pool, rows, [&](size_t row) {
                    const int position = base_position + static_cast<int>(row);
                    run_latent_attention(
                        state, layout,
                        workspace_.chunk_qkv.data() + row * content_width,
                        rope_width == 0 ? nullptr : workspace_.chunk_latent_rope.data() + row * rope_width,
                        workspace_.chunk_op.data() + row * content_width,
                        committed_length, position, attention->relative_bias);
                });
                session_.prefill_profile.attention_ms += milliseconds_since(attention_started);
                linear_started = Clock::now();
                layer_gemm(attention->out, workspace_.chunk_op.data(),
                           workspace_.chunk_hidden.data());
                session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            } else {
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
                (void)projected_q;
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
                                  CpuAttentionPattern::lower(layout.pattern),
                                  CpuAttentionBias::lower(layout.bias, attention->relative_bias,
                                                          layout.query_heads));
            if (layout.output_gate.enabled()) {
                if (!layout.output_gate.packed_with_query) {
                    layer_gemm(attention->gate, workspace_.chunk_normed.data(),
                               workspace_.chunk_attention_gate.data());
                }
                for (size_t row = 0; row < rows; ++row) {
                    const float* gate = layout.output_gate.packed_with_query
                        ? workspace_.chunk_qkv.data() + row * q_projection_width + q_width
                        : workspace_.chunk_attention_gate.data() + row * q_width;
                    float* output = workspace_.chunk_op.data() + row * q_width;
                    apply_cpu_attention_output_gate(output, gate, q_width);
                }
            }
            session_.prefill_profile.attention_ms += milliseconds_since(attention_started);
            linear_started = Clock::now();
            layer_gemm(attention->out, workspace_.chunk_op.data(),
                       workspace_.chunk_hidden.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            }
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
            rmsnorm_rows_inplace(workspace_.chunk_hidden.data(), common.post_attention_norm, hidden,
                                 semantics.post_attention_norm.epsilon);
        }
        residual_rows(workspace_.chunk_hidden.data(), workspace_.chunk_residual.data(), hidden);
        rmsnorm_rows(workspace_.chunk_hidden.data(), common.ffn_norm,
                     workspace_.chunk_normed.data(), hidden,
                     semantics.feed_forward_norm.epsilon);
        normed_q8_ready = false;

        if (const auto* moe = std::get_if<MoeWeights>(&layer_program)) {
            const MoeLayerProgram& moe_semantics = shared->program.layers[index].moe.value();
            execute_cpu_moe_chunk(*this, index, *moe, moe_semantics, rows,
                                  normed_q8_ready);
        } else {
            execute_cpu_dense_feed_forward_chunk(execution, index, common, rows,
                                                 normed_q8_ready);
        }
        if (shape.numerical_policy.residual_multiplier != 1.0f) {
            scale(workspace_.chunk_mlp, rows * hidden,
                  shape.numerical_policy.residual_multiplier);
        }
        if (shape.has_split_attention_norms) {
            rmsnorm_rows_inplace(workspace_.chunk_mlp.data(), common.post_feed_forward_norm, hidden,
                                 semantics.post_feed_forward_norm.epsilon);
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
        if (std::binary_search(shared->program.norm_after_layers.begin(),
                               shared->program.norm_after_layers.end(),
                               static_cast<int>(index))) {
            rmsnorm_rows_inplace(workspace_.chunk_hidden.data(),
                                 shared->weight_store.final_norm, hidden,
                                 shared->program.final_norm.epsilon);
        }
    }

    if (compute_logits) {
        const float* last_hidden = workspace_.chunk_hidden.data() + (rows - 1) * hidden;
        cpu_rmsnorm(last_hidden, shared->weight_store.final_norm.data(), workspace_.final_normed.data(),
                    hidden, shared->program.final_norm.epsilon);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->weight_store.embedding :
                            shared->weight_store.lm_head, workspace_.final_normed.data(),
                            workspace_.logits.data());
        if (shape.numerical_policy.logits_multiplier != 1.0f) {
            for (float& value : workspace_.logits) value *= shape.numerical_policy.logits_multiplier;
        }
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

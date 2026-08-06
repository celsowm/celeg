#include "detail/model_internal.hpp"
#include "celeg/backend/cpu/sampler.hpp"
#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {

struct CpuCompiledModel::BatchScratch {
    CpuWorkspace workspace_;
    using State = CpuCompiledModel;
    using SharedWeights = State::Shared;
    using LayerWeights = State::WeightLayer;
    using CommonWeights = State::CommonWeights;
    using AttentionWeights = State::AttentionWeights;
    using ConvolutionWeights = State::ConvolutionWeights;
    using MoeWeights = State::MoeWeights;
    static void validate_shared(std::span<State* const> sessions) {
        if (sessions.empty()) throw std::invalid_argument("packed CPU batch is empty");
        const auto shared = sessions.front()->shared;
        for (State* session : sessions) {
            if (!session) throw std::invalid_argument("packed CPU session is null");
            State& impl = *session;
            if (impl.shared.get() != shared.get()) {
                throw std::invalid_argument(
                    "packed CPU sessions must share the same model weights");
            }
        }
    }

    void forward(std::span<State* const> sessions,
                 std::span<const int32_t> tokens,
                 std::span<const uint8_t> compute_logits) {
        if (sessions.size() != tokens.size() ||
            sessions.size() != compute_logits.size()) {
            throw std::invalid_argument("packed CPU metadata size mismatch");
        }
        validate_shared(sessions);
        SharedWeights& shared = *sessions.front()->shared;
        const size_t rows = sessions.size();
        const RuntimeTopology& shape = shared.shape;
        const size_t hidden = static_cast<size_t>(shape.hidden);
        workspace_.ensure(rows, shape);
        auto parallel_for = [&](size_t count, const auto& body) {
            const size_t grain = std::max<size_t>(1, count / std::max<size_t>(1, shared.pool.size() * 4));
            shared.pool.parallel_for(0, count, grain, [&](size_t begin, size_t end) {
                for (size_t row = begin; row < end; ++row) body(row);
            });
        };
        auto rows_for = [&](const auto& body) { parallel_for(rows, body); };
        auto rmsnorm_rows = [&](const float* input, const std::vector<float>& weight,
                                float* output, size_t width) {
            rows_for([&](size_t row) {
                cpu_rmsnorm(input + row * width, weight.data(), output + row * width,
                            width, shape.numerical_policy.norm_eps);
            });
        };
        auto rmsnorm_rows_inplace = [&](float* values, const std::vector<float>& weight,
                                        size_t width) {
            rows_for([&](size_t row) {
                cpu_rmsnorm_inplace(values + row * width, weight.data(), width,
                                    shape.numerical_policy.norm_eps);
            });
        };
        auto residual_rows = [&](float* values, const float* residual, size_t width) {
            rows_for([&](size_t row) {
                cpu_residual_add(values + row * width, residual + row * width, width);
            });
        };

        rows_for([&](size_t row) {
            shared.linear.embedding(shared.weight_store.embedding, tokens[row],
                                    workspace_.hidden.data() + row * hidden);
            if (shape.numerical_policy.embedding_multiplier != 1.0f) {
                float* destination = workspace_.hidden.data() + row * hidden;
                for (size_t d = 0; d < hidden; ++d) {
                    destination[d] *= shape.numerical_policy.embedding_multiplier;
                }
            }
        });

        const PerLayerInputPlan& input_plan = shared.program.per_layer_input;
        if (input_plan.enabled) {
            const size_t packed = input_plan.packed_width;
            const size_t input_size = static_cast<size_t>(input_plan.input_size);
            workspace_.per_layer_input.resize(rows * packed);
            workspace_.per_layer_context.resize(rows * packed);
            workspace_.per_layer_gate.resize(rows * input_size);
            rows_for([&](size_t row) {
                float* destination = workspace_.per_layer_input.data() + row * packed;
                shared.linear.embedding(shared.weight_store.per_layer_embedding, tokens[row], destination);
                for (size_t d = 0; d < packed; ++d) destination[d] *= input_plan.token_scale;
            });
            shared.linear.gemm(shared.weight_store.per_layer_context_projection,
                               workspace_.hidden.data(), workspace_.per_layer_context.data(), rows);
            rows_for([&](size_t row) {
                float* context = workspace_.per_layer_context.data() + row * packed;
                const float* token_values = workspace_.per_layer_input.data() + row * packed;
                for (int layer = 0; layer < input_plan.layer_count; ++layer) {
                    float* values = context + static_cast<size_t>(layer) * input_size;
                    if (input_plan.context_scale != 1.0f) {
                        for (size_t d = 0; d < input_size; ++d) values[d] *= input_plan.context_scale;
                    }
                    cpu_rmsnorm_inplace(values,
                                        shared.weight_store.per_layer_projection_norm.data(),
                                        input_size, input_plan.norm_epsilon);
                    const float* token_input = token_values + static_cast<size_t>(layer) * input_size;
                    for (size_t d = 0; d < input_size; ++d) {
                        values[d] = (values[d] + token_input[d]) * input_plan.residual_scale;
                    }
                }
            });
        }

        for (size_t index = 0; index < shared.weight_store.layers.size(); ++index) {
            const LayerWeights& layer_program = shared.weight_store.layers[index];
            const CommonWeights& common = sessions.front()->common_weights(index);
            std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.residual.begin());
            rmsnorm_rows(workspace_.hidden.data(), common.operator_norm,
                         workspace_.normed.data(), hidden);
            bool normed_q8_ready = false;
            auto layer_gemm = [&](const CpuLinearWeight& weight, const float* input,
                                  float* output, float beta = 0.0f) {
                const bool cacheable = weight.gguf_native() && weight.cols == hidden &&
                    input == workspace_.normed.data();
                if (cacheable) {
                    if (!normed_q8_ready) {
                        shared.linear.prepare_gguf_activation(
                            input, rows, hidden, workspace_.chunk_q8);
                        normed_q8_ready = true;
                    }
                    shared.linear.gemm_gguf(workspace_.chunk_q8, weight, output,
                                            rows, beta);
                } else {
                    shared.linear.gemm(weight, input, output, rows, beta);
                }
            };
            if (const AttentionWeights* attention = State::attention_operator(layer_program)) {
                const AttentionSpec& layout = shape.attention_layout(static_cast<int>(index));
                const size_t q_width = static_cast<size_t>(layout.query_width());
                const size_t kv_width = static_cast<size_t>(layout.key_value_width());
                layer_gemm(attention->q, workspace_.normed.data(), workspace_.qkv.data());
                if (!attention->k.segments.empty()) {
                    layer_gemm(attention->k, workspace_.normed.data(), workspace_.op_output.data());
                    layer_gemm(attention->v, workspace_.normed.data(), workspace_.conv_projected.data());
                }
                rows_for([&](size_t row) {
                    float* q = workspace_.qkv.data() + row * q_width;
                    float* k = workspace_.op_output.data() + row * kv_width;
                    const int position = sessions[row]->session_.position_value;
                    if (layout.positional_encoding == PositionalEncodingKind::None) {
                        const float ratio = shape.numerical_policy.attention_multiplier /
                            (1.0f / std::sqrt(static_cast<float>(layout.head_dim)));
                        for (size_t d = 0; d < q_width; ++d) q[d] *= ratio;
                    } else if (layout.query_key_norm) {
                        cpu_qk_norm_rope(q, attention->q_norm.data(), layout.query_heads,
                                         layout.head_dim, position, static_cast<float>(layout.rope_theta),
                                         shape.numerical_policy.norm_eps,
                                         static_cast<float>(layout.rotary_fraction));
                        if (!attention->k.segments.empty()) {
                            cpu_qk_norm_rope(k, attention->k_norm.data(), layout.key_value_heads,
                                             layout.head_dim, position, static_cast<float>(layout.rope_theta),
                                             shape.numerical_policy.norm_eps,
                                             static_cast<float>(layout.rotary_fraction));
                        }
                    } else {
                        cpu_rope(q, layout.query_heads, layout.head_dim, position,
                                 static_cast<float>(layout.rope_theta),
                                 static_cast<float>(layout.rotary_fraction));
                        if (!attention->k.segments.empty()) {
                            cpu_rope(k, layout.key_value_heads, layout.head_dim, position,
                                     static_cast<float>(layout.rope_theta),
                                     static_cast<float>(layout.rotary_fraction));
                        }
                        const float ratio = shape.numerical_policy.attention_multiplier /
                            (1.0f / std::sqrt(static_cast<float>(layout.head_dim)));
                        for (size_t d = 0; d < q_width; ++d) q[d] *= ratio;
                    }
                });
                for (size_t row = 0; row < rows; ++row) {
                    const int owner = shared.layer_to_kv_owner.at(index);
                    AttentionState& state = sessions[row]->attention_state(static_cast<size_t>(owner));
                    if (!attention->k.segments.empty()) {
                        sessions[row]->store_kv(state, sessions[row]->session_.position_value,
                                                workspace_.op_output.data() + row * kv_width,
                                                workspace_.conv_projected.data() + row * kv_width);
                    }
                    sessions[row]->run_attention(state, layout,
                                                 workspace_.qkv.data() + row * q_width,
                                                 workspace_.op_output.data() + row * q_width,
                                                 sessions[row]->session_.position_value + 1);
                }
                layer_gemm(attention->out, workspace_.op_output.data(), workspace_.hidden.data());
            } else {
                const ConvolutionWeights* convolution = State::convolution_operator(layer_program);
                if (!convolution) throw std::logic_error("packed CPU layer has no operator");
                layer_gemm(convolution->in, workspace_.normed.data(),
                           workspace_.conv_projected.data());
                rows_for([&](size_t row) {
                    ConvolutionState& state = sessions[row]->convolution_state(index);
                    cpu_conv_decode(workspace_.conv_projected.data() + row * 3ULL * hidden,
                                    convolution->weight_tap_major.data(), state.state.data(),
                                    workspace_.op_output.data() + row * hidden, shape.hidden,
                                    shape.conv_cache, sessions[row]->session_.position_value);
                });
                layer_gemm(convolution->out, workspace_.op_output.data(),
                           workspace_.hidden.data());
            }
            if (shape.numerical_policy.residual_multiplier != 1.0f) {
                rows_for([&](size_t row) {
                    float* values = workspace_.hidden.data() + row * hidden;
                    for (size_t d = 0; d < hidden; ++d) values[d] *= shape.numerical_policy.residual_multiplier;
                });
            }
            if (shape.has_split_attention_norms) {
                rmsnorm_rows_inplace(workspace_.hidden.data(), common.post_attention_norm, hidden);
            }
            residual_rows(workspace_.hidden.data(), workspace_.residual.data(), hidden);
            rmsnorm_rows(workspace_.hidden.data(), common.ffn_norm, workspace_.normed.data(), hidden);
            if (const auto* moe = std::get_if<MoeWeights>(&layer_program)) {
                const MoeLayerProgram& semantics = shared.program.layers[index].moe.value();
                const int experts = semantics.router.expert_count;
                const int selected = semantics.router.experts_per_token;
                const int intermediate = semantics.routed.mlp.intermediate_size;
                const size_t routes = rows * static_cast<size_t>(selected);
                std::fill(workspace_.mlp_output.begin(),
                          workspace_.mlp_output.begin() + rows * hidden, 0.0f);
                workspace_.moe_router_logits.resize(rows * static_cast<size_t>(experts));
                workspace_.moe_router_probs.resize(rows * static_cast<size_t>(experts));
                workspace_.moe_router_scored.resize(rows * static_cast<size_t>(experts));
                workspace_.moe_selected.resize(routes);
                workspace_.moe_weights.resize(routes);
                workspace_.moe_route_rows.resize(routes);
                workspace_.moe_route_experts.resize(routes);
                workspace_.moe_route_weights.resize(routes);
                workspace_.moe_group_offsets.assign(static_cast<size_t>(experts) + 1, 0);
                workspace_.moe_group_cursor.resize(static_cast<size_t>(experts));
                workspace_.moe_route_order.resize(routes);
                shared.linear.gemm_raw(moe->router.data(), workspace_.normed.data(),
                                       workspace_.moe_router_logits.data(), rows, experts, shape.hidden);
                for (size_t row = 0; row < rows; ++row) {
                    float* probabilities = workspace_.moe_router_probs.data() +
                        row * static_cast<size_t>(experts);
                    std::pair<float, int>* scored = workspace_.moe_router_scored.data() +
                        row * static_cast<size_t>(experts);
                    const float* logits = workspace_.moe_router_logits.data() +
                        row * static_cast<size_t>(experts);
                    for (int expert = 0; expert < experts; ++expert) {
                        const float probability = 1.0f / (1.0f + std::exp(-logits[expert]));
                        probabilities[expert] = probability;
                        scored[expert] = {probability +
                            (semantics.router.has_expert_bias &&
                             expert < static_cast<int>(moe->router_bias.size())
                                ? moe->router_bias[expert] : 0.0f), expert};
                    }
                    std::partial_sort(scored, scored + selected, scored + experts,
                                      [](const std::pair<float, int>& a,
                                         const std::pair<float, int>& b) {
                                          return a.first == b.first ? a.second < b.second : a.first > b.first;
                                      });
                    float sum = 0.0f;
                    for (int route = 0; route < selected; ++route) {
                        const size_t route_index = row * static_cast<size_t>(selected) + route;
                        const int expert = scored[route].second;
                        workspace_.moe_selected[route_index] = expert;
                        workspace_.moe_weights[route_index] = probabilities[expert];
                        sum += workspace_.moe_weights[route_index];
                    }
                    if (semantics.router.normalization == MoeNormalizationKind::SumSelected) {
                        for (int route = 0; route < selected; ++route) {
                            workspace_.moe_weights[row * static_cast<size_t>(selected) + route] /= sum + 1e-6f;
                        }
                    }
                    for (int route = 0; route < selected; ++route) {
                        const size_t route_index = row * static_cast<size_t>(selected) + route;
                        const int expert = workspace_.moe_selected[route_index];
                        ++workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1];
                        workspace_.moe_route_rows[route_index] = static_cast<int>(row);
                        workspace_.moe_route_experts[route_index] = expert;
                        workspace_.moe_route_weights[route_index] =
                            workspace_.moe_weights[route_index] * semantics.router.routed_scaling;
                    }
                }
                for (int expert = 0; expert < experts; ++expert) {
                    workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1] +=
                        workspace_.moe_group_offsets[static_cast<size_t>(expert)];
                    workspace_.moe_group_cursor[static_cast<size_t>(expert)] =
                        workspace_.moe_group_offsets[static_cast<size_t>(expert)];
                }
                for (size_t route = 0; route < routes; ++route) {
                    const int expert = workspace_.moe_route_experts[route];
                    workspace_.moe_route_order[
                        workspace_.moe_group_cursor[static_cast<size_t>(expert)]++] = route;
                }
                workspace_.moe_gathered_normed.resize(routes * hidden);
                workspace_.moe_gathered_gate_up.resize(routes * 2ULL * intermediate);
                workspace_.moe_gathered_activated.resize(routes * static_cast<size_t>(intermediate));
                workspace_.moe_gathered_output.resize(routes * hidden);
                workspace_.moe_cached_experts.clear();
                workspace_.moe_cached_experts.resize(static_cast<size_t>(experts));
                workspace_.moe_gemm_jobs.clear();
                for (int expert = 0; expert < experts; ++expert) {
                    const size_t begin = workspace_.moe_group_offsets[static_cast<size_t>(expert)];
                    const size_t end = workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1];
                    if (begin == end) continue;
                    const CpuLinearWeight* w13 = nullptr;
                    if (moe->disk_cached) {
                        workspace_.moe_cached_experts[static_cast<size_t>(expert)] =
                            shared.acquire_expert(moe->layer_index, expert);
                        w13 = &workspace_.moe_cached_experts[static_cast<size_t>(expert)]->w13;
                    } else {
                        w13 = &moe->expert_w13[static_cast<size_t>(expert)];
                    }
                    workspace_.moe_gemm_jobs.push_back({w13, begin, end - begin});
                }
                for (size_t packed_route = 0; packed_route < routes; ++packed_route) {
                    const size_t route = workspace_.moe_route_order[packed_route];
                    const size_t row = static_cast<size_t>(workspace_.moe_route_rows[route]);
                    std::copy_n(workspace_.normed.data() + row * hidden, hidden,
                                workspace_.moe_gathered_normed.data() + packed_route * hidden);
                }
                shared.linear.gemm_grouped(workspace_.moe_gemm_jobs,
                                           workspace_.moe_gathered_normed.data(),
                                           workspace_.moe_gathered_gate_up.data());
                parallel_for(routes, [&](size_t route) {
                    cpu_swiglu(workspace_.moe_gathered_gate_up.data() + route * 2ULL * intermediate,
                               workspace_.moe_gathered_activated.data() + route * intermediate,
                               intermediate);
                });
                workspace_.moe_gemm_jobs.clear();
                for (int expert = 0; expert < experts; ++expert) {
                    const size_t begin = workspace_.moe_group_offsets[static_cast<size_t>(expert)];
                    const size_t end = workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1];
                    if (begin == end) continue;
                    const CpuLinearWeight* w2 = moe->disk_cached
                        ? &workspace_.moe_cached_experts[static_cast<size_t>(expert)]->w2
                        : &moe->expert_w2[static_cast<size_t>(expert)];
                    workspace_.moe_gemm_jobs.push_back({w2, begin, end - begin});
                }
                shared.linear.gemm_grouped(workspace_.moe_gemm_jobs,
                                           workspace_.moe_gathered_activated.data(),
                                           workspace_.moe_gathered_output.data());
                for (size_t packed_route = 0; packed_route < routes; ++packed_route) {
                    const size_t route = workspace_.moe_route_order[packed_route];
                    const size_t row = static_cast<size_t>(workspace_.moe_route_rows[route]);
                    float* destination = workspace_.mlp_output.data() + row * hidden;
                    const float* source = workspace_.moe_gathered_output.data() + packed_route * hidden;
                    const float weight = workspace_.moe_route_weights[route];
                    for (size_t d = 0; d < hidden; ++d) destination[d] += weight * source[d];
                }
            } else {
                const int intermediate = shape.feed_forward_intermediates.empty()
                    ? shape.intermediate : shape.feed_forward_intermediates.at(index);
                layer_gemm(common.w13, workspace_.normed.data(), workspace_.gate_up.data());
                rows_for([&](size_t row) {
                    const float* gate_up = workspace_.gate_up.data() + row * 2ULL * intermediate;
                    float* activated = workspace_.activated.data() + row * intermediate;
                    if (!shape.feed_forward_activations.empty() &&
                        shape.feed_forward_activations.at(index) == ActivationKind::GeluTanh) {
                        cpu_gated_gelu_tanh(gate_up, activated, intermediate);
                    } else {
                        cpu_swiglu(gate_up, activated, intermediate);
                    }
                });
                layer_gemm(common.w2, workspace_.activated.data(), workspace_.mlp_output.data());
            }
            if (shape.numerical_policy.residual_multiplier != 1.0f) {
                rows_for([&](size_t row) {
                    float* values = workspace_.mlp_output.data() + row * hidden;
                    for (size_t d = 0; d < hidden; ++d) values[d] *= shape.numerical_policy.residual_multiplier;
                });
            }
            if (shape.has_split_attention_norms) {
                rmsnorm_rows_inplace(workspace_.mlp_output.data(), common.post_feed_forward_norm, hidden);
            }
            residual_rows(workspace_.hidden.data(), workspace_.mlp_output.data(), hidden);
            normed_q8_ready = false;

            if (input_plan.enabled) {
                std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.residual.begin());
                shared.linear.gemm(common.per_layer_input_gate, workspace_.hidden.data(),
                                   workspace_.per_layer_gate.data(), rows);
                const size_t input_size = static_cast<size_t>(input_plan.input_size);
                rows_for([&](size_t row) {
                    float* gate = workspace_.per_layer_gate.data() + row * input_size;
                    cpu_gelu_tanh(gate, input_size);
                    const float* context = workspace_.per_layer_context.data() +
                        row * input_plan.packed_width + index * input_size;
                    for (size_t d = 0; d < input_size; ++d) gate[d] *= context[d];
                });
                shared.linear.gemm(common.per_layer_projection, workspace_.per_layer_gate.data(),
                                   workspace_.hidden.data(), rows);
                rmsnorm_rows_inplace(workspace_.hidden.data(), common.per_layer_input_norm, hidden);
                if (common.layer_scalar != 1.0f) {
                    rows_for([&](size_t row) {
                        float* values = workspace_.hidden.data() + row * hidden;
                        for (size_t d = 0; d < hidden; ++d) values[d] *= common.layer_scalar;
                    });
                }
                residual_rows(workspace_.hidden.data(), workspace_.residual.data(), hidden);
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            State& session = *sessions[row];
            if (compute_logits[row]) {
                cpu_rmsnorm(workspace_.hidden.data() + row * hidden,
                            shared.weight_store.final_norm.data(), workspace_.normed.data() + row * hidden,
                            hidden, shape.numerical_policy.norm_eps);
                shared.linear.gemv(shared.tie_word_embeddings ? shared.weight_store.embedding :
                                    shared.weight_store.lm_head,
                                    workspace_.normed.data() + row * hidden,
                                    session.workspace_.logits.data());
                if (shape.numerical_policy.logits_divisor != 1.0f) {
                    for (float& value : session.workspace_.logits) value /= shape.numerical_policy.logits_divisor;
                }
                if (shared.final_logit_softcap > 0.0f) {
                    for (float& value : session.workspace_.logits) {
                        value = std::tanh(value / shared.final_logit_softcap) * shared.final_logit_softcap;
                    }
                }
            }
            ++session.session_.position_value;
        }
    }

};

void CpuCompiledModel::forward_batch(std::span<CpuCompiledModel* const> sessions,
                                   std::span<const int32_t> tokens,
                                   std::span<const uint8_t> compute_logits) {
    thread_local BatchScratch scratch;
    scratch.forward(sessions, tokens, compute_logits);
}

CpuBatchMetrics CpuModel::prefill_batch(std::span<const CpuPrefillItem> items) {
    if (items.empty()) throw std::invalid_argument("CPU ragged prefill batch is empty");
    std::vector<CpuCompiledModel*> sessions;
    std::vector<int32_t> tokens;
    std::vector<uint8_t> terminal;
    sessions.reserve(items.size());
    tokens.reserve(items.size());
    terminal.reserve(items.size());
    for (const CpuPrefillItem& item : items) {
        if (!item.session || !item.session->state_) {
            throw std::invalid_argument("packed CPU session is null");
        }
        CpuCompiledModel& session = *item.session->state_;
        if (session.session_.phase != SessionPhase::Empty &&
            session.session_.phase != SessionPhase::Prefilling) {
            throw std::runtime_error("CPU session is not eligible for prefill");
        }
        if (item.token < 0 ||
            item.token >= session.shared->shape.vocab_size) {
            throw std::invalid_argument("CPU prefill token out of range");
        }
        session.session_.phase = SessionPhase::Prefilling;
        session.session_.seen[static_cast<size_t>(item.token)] = 1;
        sessions.push_back(item.session->state_.get());
        tokens.push_back(item.token);
        terminal.push_back(item.final_token ? 1 : 0);
    }
    const auto started = std::chrono::steady_clock::now();
    CpuCompiledModel::forward_batch(sessions, tokens, terminal);
    const auto ended = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(ended - started).count();
    for (const CpuPrefillItem& item : items) {
        CpuCompiledModel& session = *item.session->state_;
        ++session.session_.metrics.prefill_tokens;
        session.session_.metrics.last_prefill_ms += elapsed / items.size();
    }
    return {items.size(), elapsed};
}

CpuBatchMetrics CpuModel::prefill_chunk(
    CpuModel& model, std::span<const int32_t> tokens, bool final_chunk) {
    if (tokens.empty()) throw std::invalid_argument("CPU chunked prefill is empty");
    if (!model.state_) throw std::invalid_argument("CPU chunked prefill session is null");
    CpuCompiledModel& session = *model.state_;
    if (session.session_.phase != SessionPhase::Empty &&
        session.session_.phase != SessionPhase::Prefilling) {
        throw std::runtime_error("CPU session is not eligible for chunked prefill");
    }
    for (int32_t token : tokens) {
        if (token < 0 ||
            token >= session.shared->shape.vocab_size) {
            throw std::invalid_argument("CPU chunked prefill token out of range");
        }
        session.session_.seen[static_cast<size_t>(token)] = 1;
    }
    session.session_.phase = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    session.forward_chunk(tokens, final_chunk);
    const auto ended = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(ended - started).count();
    session.session_.metrics.prefill_tokens += tokens.size();
    session.session_.metrics.last_prefill_ms += elapsed;
    session.session_.prefill_profile.total_ms += elapsed;
    if (final_chunk) session.session_.phase = SessionPhase::Ready;
    return {tokens.size(), elapsed};
}

std::pair<std::vector<int32_t>, CpuBatchMetrics>
CpuModel::decode_batch(std::span<CpuModel* const> models) {
    if (models.empty()) throw std::invalid_argument("packed CPU batch is empty");
    std::vector<CpuCompiledModel*> sessions;
    sessions.reserve(models.size());
    for (CpuModel* model : models) {
        if (!model || !model->state_) {
            throw std::invalid_argument("packed CPU session is null");
        }
        sessions.push_back(model->state_.get());
    }
    CpuCompiledModel::BatchScratch::validate_shared(sessions);
    std::vector<int32_t> tokens;
    std::vector<uint8_t> compute_logits(sessions.size(), 1);
    tokens.reserve(sessions.size());
    for (CpuCompiledModel* state : sessions) {
        CpuCompiledModel& session = *state;
        if (session.session_.phase != SessionPhase::Ready) {
            throw std::runtime_error("CPU session is not ready for packed decode");
        }
        const int32_t token = CpuSampler::sample(
            session.workspace_.logits, session.shared->shape,
            session.session_.generation, session.session_.seen,
            session.session_.rng_state);
        session.session_.seen[static_cast<size_t>(token)] = 1;
        tokens.push_back(token);
    }
    const auto started = std::chrono::steady_clock::now();
    CpuCompiledModel::forward_batch(sessions, tokens, compute_logits);
    const auto ended = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(ended - started).count();
    for (CpuCompiledModel* state : sessions) {
        CpuCompiledModel& session = *state;
        session.session_.metrics.cumulative_decode_ms += elapsed / sessions.size();
        ++session.session_.metrics.decoded_tokens;
    }
    return {std::move(tokens), {sessions.size(), elapsed}};
}

} // namespace celeg

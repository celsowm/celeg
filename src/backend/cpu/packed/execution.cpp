#include "../detail/model_internal.hpp"
#include "../operators/attention.hpp"
#include "../operators/moe.hpp"
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
        const ExecutionTopology& shape = shared.shape;
        const size_t hidden = static_cast<size_t>(shared.program.hidden);
        workspace_.ensure(rows, shared.workspace_plan);
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
                            width, shared.program.final_norm.epsilon);
            });
        };
        auto rmsnorm_rows_inplace = [&](float* values, const std::vector<float>& weight,
                                        size_t width) {
            rows_for([&](size_t row) {
                cpu_rmsnorm_inplace(values + row * width, weight.data(), width,
                                    shared.program.final_norm.epsilon);
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
            if (shared.program.embedding_transform.multiplier != 1.0f) {
                float* destination = workspace_.hidden.data() + row * hidden;
                for (size_t d = 0; d < hidden; ++d) {
                    destination[d] *= shared.program.embedding_transform.multiplier;
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
            const CompiledLayerProgram& layer_semantics = shared.program.layers.at(index);
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
            visit_operator_weights(layer_program,
              [&](const GatedDeltaNetWeights* gated_delta) {
                const GatedDeltaNetSpec& spec = gated_delta->spec;
                const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
                    spec.value_heads * spec.value_head_dim;
                const int value_width = spec.value_heads * spec.value_head_dim;
                layer_gemm(gated_delta->qkv, workspace_.normed.data(),
                           workspace_.gated_delta_qkv.data());
                layer_gemm(gated_delta->z, workspace_.normed.data(),
                           workspace_.gated_delta_z.data());
                layer_gemm(gated_delta->b, workspace_.normed.data(),
                           workspace_.gated_delta_b.data());
                layer_gemm(gated_delta->a, workspace_.normed.data(),
                           workspace_.gated_delta_a.data());
                rows_for([&](size_t row) {
                    GatedDeltaNetState& state = sessions[row]->gated_delta_net_state(index);
                    cpu_gated_delta_net_decode(
                        workspace_.gated_delta_qkv.data() + row * qkv_width,
                        workspace_.gated_delta_z.data() + row * value_width,
                        workspace_.gated_delta_b.data() + row * spec.value_heads,
                        workspace_.gated_delta_a.data() + row * spec.value_heads,
                        gated_delta->conv_weight.data(), gated_delta->dt_bias.data(),
                        gated_delta->a_log.data(), gated_delta->norm.data(),
                        state.conv.data(), state.recurrent.data(),
                        workspace_.gated_delta_output.data() + row * value_width,
                        spec.conv_kernel, spec.key_head_dim, spec.value_head_dim,
                        spec.key_heads, spec.value_heads,
                        layer_semantics.operator_norm.epsilon, spec.vector_decay,
                        spec.safe_decay, spec.decay_lower_bound,
                        spec.sigmoid_output_gate);
                });
                layer_gemm(gated_delta->out, workspace_.gated_delta_output.data(),
                           workspace_.hidden.data());
              },
              [&](const State::Mamba2Weights*) {
                throw std::logic_error("packed CPU execution does not implement the Mamba2 mixer");
              },
              [&](const State::MlpOnlyWeights*) {
                throw std::logic_error("packed CPU execution does not implement MLP-only blocks");
              },
              [&](const ConvolutionWeights* convolution) {
                layer_gemm(convolution->in, workspace_.normed.data(),
                           workspace_.conv_projected.data());
                rows_for([&](size_t row) {
                    ConvolutionState& state = sessions[row]->convolution_state(index);
                    cpu_conv_decode(workspace_.conv_projected.data() + row * 3ULL * hidden,
                                    convolution->weight_tap_major.data(), state.state.data(),
                                    workspace_.op_output.data() + row * hidden, shared.program.hidden,
                                    shape.conv_cache, sessions[row]->session_.position_value);
                });
                layer_gemm(convolution->out, workspace_.op_output.data(),
                           workspace_.hidden.data());
              },
              [&](const AttentionWeights* attention) {
                const AttentionSpec& layout = std::get<CompiledAttentionProgram>(
                    layer_semantics.mixer).semantics;
                if (layout.uses_external_memory()) {
                    const size_t q_width = static_cast<size_t>(layout.query_width());
                    const size_t q_projection_width = static_cast<size_t>(attention->q.rows);
                    layer_gemm(attention->q, workspace_.normed.data(), workspace_.qkv.data());
                    rows_for([&](size_t row) {
                        const int position = sessions[row]->session_.position_value;
                        const std::array<int32_t, 3> rope_position = {
                            position, position, position};
                        apply_cpu_attention_qk(
                            layout, *attention,
                            workspace_.qkv.data() + row * q_projection_width,
                            nullptr, position, rope_position);
                    });
                    const auto memory_it = shared.external_attention_memory.find(
                        layout.sources.memory_slot);
                    if (memory_it == shared.external_attention_memory.end()) {
                        throw std::logic_error("external attention memory slot is not bound");
                    }
                    for (size_t row = 0; row < rows; ++row) {
                        sessions[row]->run_external_attention(
                            layout, *memory_it->second,
                            workspace_.qkv.data() + row * q_projection_width,
                            workspace_.op_output.data() + row * q_width,
                            attention->relative_bias);
                        if (layout.output_gate.enabled()) {
                            apply_cpu_attention_output_gate(
                                workspace_.op_output.data() + row * q_width,
                                workspace_.qkv.data() + row * q_projection_width + q_width,
                                q_width);
                        }
                    }
                    layer_gemm(attention->out, workspace_.op_output.data(), workspace_.hidden.data());
                } else if (layout.uses_latent_state()) {
                    const auto& latent = *layout.latent_state();
                    if (layout.output_gate.enabled()) {
                        throw std::invalid_argument("latent attention query gating is not supported");
                    }
                    const size_t content_width = static_cast<size_t>(layout.latent_query_content_width());
                    const size_t rope_width = static_cast<size_t>(layout.latent_query_rope_width());
                    layer_gemm(attention->q, workspace_.normed.data(), workspace_.qkv.data());
                    if (rope_width != 0) {
                        layer_gemm(attention->latent_q_rope, workspace_.normed.data(),
                                   workspace_.latent_rope.data());
                    }
                    layer_gemm(attention->k, workspace_.normed.data(), workspace_.latent_key.data());
                    layer_gemm(attention->v, workspace_.normed.data(), workspace_.latent_value.data());
                    if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                        layer_gemm(attention->latent_k_rope, workspace_.normed.data(),
                                   workspace_.latent_key_rope.data());
                    }
                    rows_for([&](size_t row) {
                        const int position = sessions[row]->session_.position_value;
                        const std::array<int32_t, 3> rope_position = {
                            position, position, position};
                        apply_cpu_latent_attention_positions(
                            layout,
                            rope_width == 0 ? nullptr : workspace_.latent_rope.data() + row * rope_width,
                            latent.decoupled_rope && latent.rope_head_dim != 0
                                ? workspace_.latent_key_rope.data() +
                                    row * static_cast<size_t>(latent.rope_head_dim) : nullptr,
                            position, rope_position);
                    });
                    for (size_t row = 0; row < rows; ++row) {
                        const int owner = shared.layer_to_kv_owner.at(index);
                        AttentionState& state = sessions[row]->attention_state(static_cast<size_t>(owner));
                        sessions[row]->store_latent(
                            state, sessions[row]->session_.position_value,
                            workspace_.latent_key.data() + row * latent.latent_rank,
                            workspace_.latent_value.data() + row * latent.latent_rank,
                            latent.decoupled_rope && latent.rope_head_dim != 0
                                ? workspace_.latent_key_rope.data() +
                                    row * static_cast<size_t>(latent.rope_head_dim) : nullptr);
                        sessions[row]->run_latent_attention(
                            state, layout,
                            workspace_.qkv.data() + row * content_width,
                            rope_width == 0 ? nullptr : workspace_.latent_rope.data() + row * rope_width,
                            workspace_.op_output.data() + row * content_width,
                            sessions[row]->session_.position_value + 1,
                            sessions[row]->session_.position_value,
                            attention->relative_bias);
                    }
                    layer_gemm(attention->out, workspace_.op_output.data(), workspace_.hidden.data());
                } else {
                const size_t q_width = static_cast<size_t>(layout.query_width());
                const size_t q_projection_width = static_cast<size_t>(attention->q.rows);
            const bool attention_output_gate = layout.output_gate.enabled() ||
                    q_projection_width == 2 * q_width;
                const size_t kv_width = static_cast<size_t>(layout.key_value_width());
                layer_gemm(attention->q, workspace_.normed.data(), workspace_.qkv.data());
                if (!attention->k.segments.empty()) {
                    layer_gemm(attention->k, workspace_.normed.data(), workspace_.op_output.data());
                    layer_gemm(attention->v, workspace_.normed.data(), workspace_.conv_projected.data());
                }
                rows_for([&](size_t row) {
                    float* q = workspace_.qkv.data() + row * q_projection_width;
                    float* k = workspace_.op_output.data() + row * kv_width;
                    const int position = sessions[row]->session_.position_value;
                    const std::array<int32_t, 3> rope_position = {
                        position, position, position};
                    apply_cpu_attention_qk(layout, *attention, q, k,
                                           position, rope_position);
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
                                                 workspace_.qkv.data() + row * q_projection_width,
                                                 workspace_.op_output.data() + row * q_width,
                                                 sessions[row]->session_.position_value + 1,
                                                 attention->relative_bias);
                    if (attention_output_gate) {
                        const float* gate = workspace_.qkv.data() + row * q_projection_width + q_width;
                        float* output = workspace_.op_output.data() + row * q_width;
                        apply_cpu_attention_output_gate(output, gate, q_width);
                    }
                }
                layer_gemm(attention->out, workspace_.op_output.data(), workspace_.hidden.data());
                }
              });
            if (layer_semantics.residual.multiplier != 1.0f) {
                rows_for([&](size_t row) {
                    float* values = workspace_.hidden.data() + row * hidden;
                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;
                });
            }
            if (layer_semantics.post_attention_norm.has_value()) {
                rmsnorm_rows_inplace(workspace_.hidden.data(), common.post_attention_norm, hidden);
            }
            residual_rows(workspace_.hidden.data(), workspace_.residual.data(), hidden);
            rmsnorm_rows(workspace_.hidden.data(), common.ffn_norm, workspace_.normed.data(), hidden);
            if (const auto* moe = std::get_if<MoeWeights>(&layer_program)) {
                const MoeLayerProgram& semantics =
                    std::get<MoeLayerProgram>(shared.program.layers[index].feed_forward);
                const int experts = semantics.router.expert_count;
                const int selected = semantics.router.experts_per_token;
                const int intermediate = semantics.routed.mlp.intermediate_size;
                const size_t routes = rows * static_cast<size_t>(selected);
                std::fill(workspace_.mlp_output.begin(),
                          workspace_.mlp_output.begin() + rows * hidden, 0.0f);
                workspace_.moe_router_logits.resize(rows * static_cast<size_t>(experts));
                workspace_.moe_router_probs.resize(rows * static_cast<size_t>(experts));
                workspace_.moe_selected.resize(routes);
                workspace_.moe_weights.resize(routes);
                workspace_.moe_route_rows.resize(routes);
                workspace_.moe_route_experts.resize(routes);
                workspace_.moe_route_weights.resize(routes);
                workspace_.moe_group_offsets.assign(static_cast<size_t>(experts) + 1, 0);
                workspace_.moe_group_cursor.resize(static_cast<size_t>(experts));
                workspace_.moe_route_order.resize(routes);
                shared.linear.gemm_raw(moe->router.data(), workspace_.normed.data(),
                                       workspace_.moe_router_logits.data(), rows, experts,
                                       shared.program.hidden);
                for (size_t row = 0; row < rows; ++row) {
                    const float* logits = workspace_.moe_router_logits.data() +
                        row * static_cast<size_t>(experts);
                    const CpuMoeRoute resolved_route = route_cpu_moe(
                        semantics.router, {logits, static_cast<size_t>(experts)},
                        moe->router_bias);
                    for (int expert = 0; expert < experts; ++expert) {
                        workspace_.moe_router_probs[row * static_cast<size_t>(experts) +
                                                    static_cast<size_t>(expert)] = 0.0f;
                    }
                    for (int route = 0; route < selected; ++route) {
                        const size_t route_index = row * static_cast<size_t>(selected) + route;
                        const int expert = resolved_route.experts[static_cast<size_t>(route)];
                        workspace_.moe_selected[route_index] = expert;
                        workspace_.moe_weights[route_index] =
                            resolved_route.weights[static_cast<size_t>(route)] /
                            semantics.router.routed_scaling;
                        ++workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1];
                        workspace_.moe_route_rows[route_index] = static_cast<int>(row);
                        workspace_.moe_route_experts[route_index] = expert;
                        workspace_.moe_route_weights[route_index] =
                            resolved_route.weights[static_cast<size_t>(route)];
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
                if (semantics.shared) {
                    const int shared_intermediate = semantics.shared->mlp.intermediate_size;
                    layer_gemm(moe->shared_w13, workspace_.normed.data(),
                               workspace_.gate_up.data());
                    rows_for([&](size_t row) {
                        cpu_swiglu(workspace_.gate_up.data() +
                                       row * 2ULL * static_cast<size_t>(shared_intermediate),
                                   workspace_.activated.data() +
                                       row * static_cast<size_t>(shared_intermediate),
                                   shared_intermediate);
                    });
                    layer_gemm(moe->shared_w2, workspace_.activated.data(),
                               workspace_.shared_output.data());
                    shared.linear.gemm(moe->shared_gate, workspace_.normed.data(),
                                       workspace_.shared_gate.data(), rows);
                    rows_for([&](size_t row) {
                        const float gate = 1.0f / (1.0f + std::exp(
                            -workspace_.shared_gate[row]));
                        float* destination = workspace_.mlp_output.data() + row * hidden;
                        const float* source = workspace_.shared_output.data() + row * hidden;
                        for (size_t d = 0; d < hidden; ++d) {
                            destination[d] += gate * source[d];
                        }
                    });
                }
            } else {
                const auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(
                    &layer_semantics.feed_forward);
                if (!dense) throw std::logic_error("packed dense layer has non-dense semantics");
                const int intermediate = dense->intermediate_size;
                layer_gemm(common.w13, workspace_.normed.data(), workspace_.gate_up.data());
                rows_for([&](size_t row) {
                    const float* gate_up = workspace_.gate_up.data() + row * 2ULL * intermediate;
                    float* activated = workspace_.activated.data() + row * intermediate;
                    if (dense->activation == ActivationKind::GeluTanh) {
                        cpu_gated_gelu_tanh(gate_up, activated, intermediate);
                    } else {
                        cpu_swiglu(gate_up, activated, intermediate);
                    }
                });
                layer_gemm(common.w2, workspace_.activated.data(), workspace_.mlp_output.data());
            }
            if (layer_semantics.residual.multiplier != 1.0f) {
                rows_for([&](size_t row) {
                    float* values = workspace_.mlp_output.data() + row * hidden;
                    for (size_t d = 0; d < hidden; ++d) values[d] *= layer_semantics.residual.multiplier;
                });
            }
            if (layer_semantics.post_feed_forward_norm.has_value()) {
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
            if (std::binary_search(shared.program.norm_after_layers.begin(),
                                   shared.program.norm_after_layers.end(),
                                   static_cast<int>(index))) {
                rmsnorm_rows_inplace(workspace_.hidden.data(),
                                     shared.weight_store.final_norm, hidden);
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            State& session = *sessions[row];
            if (compute_logits[row]) {
                cpu_rmsnorm(workspace_.hidden.data() + row * hidden,
                            shared.weight_store.final_norm.data(), workspace_.normed.data() + row * hidden,
                            hidden, shared.program.final_norm.epsilon);
                shared.linear.gemv(shared.tie_word_embeddings ? shared.weight_store.embedding :
                                    shared.weight_store.lm_head,
                                    workspace_.normed.data() + row * hidden,
                                    session.workspace_.logits.data());
                if (shared.program.logits_multiplier != 1.0f) {
                    for (float& value : session.workspace_.logits) value *= shared.program.logits_multiplier;
                }
                if (shared.program.logits_divisor != 1.0f) {
                    for (float& value : session.workspace_.logits) value /= shared.program.logits_divisor;
                }
                if (shared.program.final_logit_softcap > 0.0f) {
                    for (float& value : session.workspace_.logits) {
                        value = std::tanh(value / shared.program.final_logit_softcap) *
                            shared.program.final_logit_softcap;
                    }
                }
            }
            ++session.session_.position_value;
        }
    }

};

void validate_cpu_packed_batch(std::span<CpuCompiledModel* const> sessions) {
    CpuCompiledModel::BatchScratch::validate_shared(sessions);
}

void execute_cpu_packed_batch(std::span<CpuCompiledModel* const> sessions,
                              std::span<const int32_t> tokens,
                              std::span<const uint8_t> compute_logits) {
    thread_local CpuCompiledModel::BatchScratch scratch;
    scratch.forward(sessions, tokens, compute_logits);
}

}

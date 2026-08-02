#include "detail/model_internal.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>

namespace celeg {

namespace {
using Clock = std::chrono::steady_clock;
float moe_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Prefill runs the row-wise elementwise passes (RMSNorm, SwiGLU, workspace_.residual add)
// over a whole chunk. They are independent per row, so they go on the shared
// pool instead of running single-threaded on the calling thread between the
// parallel GEMMs.
template <typename Body>
void parallel_rows(CpuThreadPool& pool, size_t rows, const Body& body) {
    const size_t grain = std::max<size_t>(1, rows / std::max<size_t>(1, pool.size() * 4));
    pool.parallel_for(0, rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) body(row);
    });
}
}

void CpuCompiledModel::forward_token(int32_t token, bool compute_logits) {
    if (session_.position_value >= shared->max_context) {
        throw std::runtime_error("CPU context limit reached");
    }
    shared->linear.embedding(shared->weight_store.embedding, token, workspace_.hidden.data());
    if (shared->shape.embedding_multiplier != 1.0f) {
        for (float& value : workspace_.hidden) value *= shared->shape.embedding_multiplier;
    }
    for (size_t index = 0; index < shared->weight_store.layers.size(); ++index) {
        const WeightLayer& layer_program = shared->weight_store.layers[index];
        const CommonWeights& common = common_weights(index);
        std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.residual.begin());
        cpu_rmsnorm(workspace_.hidden.data(), common.operator_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        if (const auto* attention = attention_operator(layer_program)) {
            shared->linear.gemv(attention->qkv, workspace_.normed.data(), workspace_.qkv.data());
            float* q = workspace_.qkv.data();
            float* k = q + shared->shape.q_width;
            float* v = k + shared->shape.kv_width;
            if (shared->shape.query_key_norm) {
                cpu_qk_norm_rope(q, attention->q_norm.data(), shared->shape.num_attention_heads,
                    shared->shape.head_dim, session_.position_value, shared->shape.rope_theta,
                    shared->shape.norm_eps);
                cpu_qk_norm_rope(k, attention->k_norm.data(), shared->shape.num_key_value_heads,
                    shared->shape.head_dim, session_.position_value, shared->shape.rope_theta,
                    shared->shape.norm_eps);
            } else {
                cpu_rope(q, shared->shape.num_attention_heads, shared->shape.head_dim,
                         session_.position_value, shared->shape.rope_theta);
                cpu_rope(k, shared->shape.num_key_value_heads, shared->shape.head_dim,
                         session_.position_value, shared->shape.rope_theta);
                const float ratio = shared->shape.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(shared->shape.head_dim)));
                for (int i = 0; i < shared->shape.q_width; ++i) q[i] *= ratio;
            }
            AttentionState& state = attention_state(index);
            store_kv(state, session_.position_value, k, v);
            run_attention(state, q, workspace_.op_output.data(), session_.position_value + 1);
            shared->linear.gemv(attention->out, workspace_.op_output.data(), workspace_.hidden.data());
        } else {
        const auto* convolution = convolution_operator(layer_program);
            if (!convolution) throw std::logic_error("CPU layer has no operator");
            ConvolutionState& state = convolution_state(index);
            shared->linear.gemv(convolution->in, workspace_.normed.data(), workspace_.conv_projected.data());
            cpu_conv_decode(workspace_.conv_projected.data(), convolution->weight_tap_major.data(),
                state.state.data(), workspace_.op_output.data(), shared->shape.hidden,
                shared->shape.conv_cache, session_.position_value);
            shared->linear.gemv(convolution->out, workspace_.op_output.data(), workspace_.hidden.data());
        }
        if (shared->shape.residual_multiplier != 1.0f) {
            for (float& value : workspace_.hidden) value *= shared->shape.residual_multiplier;
        }
        cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->shape.hidden);

        cpu_rmsnorm(workspace_.hidden.data(), common.ffn_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);

        if (const auto* moe = std::get_if<MoeWeights>(&layer_program)) {
            // MoE FFN: router -> top-K expert selection -> per-expert FFN GEMVs.
            const int E = moe->num_experts;
            const int K = moe->experts_per_token;
            const int moe_inter = shared->shape.moe_intermediate > 0
                ? shared->shape.moe_intermediate : shared->shape.intermediate;

            const bool profile_moe = session_.phase == SessionPhase::Prefilling;
            auto started = Clock::now();
            workspace_.moe_router_logits.resize(static_cast<size_t>(E));
            workspace_.moe_router_probs.resize(static_cast<size_t>(E));
            workspace_.moe_router_scored.resize(static_cast<size_t>(E));
            workspace_.moe_selected.resize(static_cast<size_t>(K));
            workspace_.moe_weights.resize(static_cast<size_t>(K));

            // Router GEMV: workspace_.normed @ router^T -> workspace_.logits [E]
            shared->linear.gemv_raw(moe->router.data(), workspace_.normed.data(), workspace_.moe_router_logits.data(),
                                    E, shared->shape.hidden);

            // Sigmoid + optional expert bias + top-K selection.
            for (int e = 0; e < E; ++e) {
                workspace_.moe_router_probs[static_cast<size_t>(e)] =
                    moe_sigmoid(workspace_.moe_router_logits[static_cast<size_t>(e)]);
                float score = workspace_.moe_router_probs[static_cast<size_t>(e)];
                if (moe->use_expert_bias && e < static_cast<int>(moe->router_bias.size())) {
                    score += moe->router_bias[static_cast<size_t>(e)];
                }
                workspace_.moe_router_scored[static_cast<size_t>(e)] = {score, e};
            }
            std::partial_sort(workspace_.moe_router_scored.begin(), workspace_.moe_router_scored.begin() + K,
                workspace_.moe_router_scored.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second;
                });

            // Gather selected experts and routing weights.
            float weight_sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                workspace_.moe_selected[static_cast<size_t>(k)] =
                    workspace_.moe_router_scored[static_cast<size_t>(k)].second;
                workspace_.moe_weights[static_cast<size_t>(k)] =
                    workspace_.moe_router_probs[static_cast<size_t>(workspace_.moe_selected[static_cast<size_t>(k)])];
                weight_sum += workspace_.moe_weights[static_cast<size_t>(k)];
            }
            if (moe->normalize_topk) {
                const float inv = 1.0f / (weight_sum + 1e-6f);
                for (int k = 0; k < K; ++k) {
                    workspace_.moe_weights[static_cast<size_t>(k)] *= inv;
                }
            }
            for (int k = 0; k < K; ++k) {
                workspace_.moe_weights[static_cast<size_t>(k)] *= moe->routed_scaling_factor;
            }
            if (profile_moe) session_.prefill_profile.moe_router_ms += milliseconds_since(started);

            // Per-expert FFN GEMVs with routing-weighted accumulation.
            started = Clock::now();
            std::fill(workspace_.mlp_output.begin(), workspace_.mlp_output.end(), 0.0f);
            for (int k = 0; k < K; ++k) {
                const int expert = workspace_.moe_selected[static_cast<size_t>(k)];
                const float rw = workspace_.moe_weights[static_cast<size_t>(k)];
                if (expert < 0 || expert >= E) continue;

                // workspace_.gate_up = expert_w13[expert] @ workspace_.normed  [2 * moe_inter]
                shared->linear.gemv(moe->expert_w13[static_cast<size_t>(expert)],
                                    workspace_.normed.data(), workspace_.gate_up.data());
                // SwiGLU: workspace_.activated[i] = swiglu(workspace_.gate_up[i], workspace_.gate_up[moe_inter + i])
                cpu_swiglu(workspace_.gate_up.data(), workspace_.activated.data(), moe_inter);
                // workspace_.mlp_output += rw * expert_w2[expert] @ workspace_.activated
                shared->linear.gemv(moe->expert_w2[static_cast<size_t>(expert)],
                                    workspace_.activated.data(), workspace_.op_output.data());
                for (int j = 0; j < shared->shape.hidden; ++j) {
                    workspace_.mlp_output[static_cast<size_t>(j)] += rw * workspace_.op_output[static_cast<size_t>(j)];
                }
            }
            if (shared->shape.residual_multiplier != 1.0f) {
                for (float& value : workspace_.mlp_output) value *= shared->shape.residual_multiplier;
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->shape.hidden);
            if (profile_moe) session_.prefill_profile.moe_expert_ms += milliseconds_since(started);
        } else {
            // Dense FFN path.
            shared->linear.gemv(common.w13, workspace_.normed.data(), workspace_.gate_up.data());
            cpu_swiglu(workspace_.gate_up.data(), workspace_.activated.data(), shared->shape.intermediate);
            shared->linear.gemv(common.w2, workspace_.activated.data(), workspace_.mlp_output.data());
            if (shared->shape.residual_multiplier != 1.0f) {
                for (float& value : workspace_.mlp_output) value *= shared->shape.residual_multiplier;
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->shape.hidden);
        }
    }
    if (compute_logits) {
        cpu_rmsnorm(workspace_.hidden.data(), shared->weight_store.final_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->weight_store.embedding :
                            shared->weight_store.lm_head, workspace_.normed.data(), workspace_.logits.data());
        if (shared->shape.logits_divisor != 1.0f) {
            for (float& value : workspace_.logits) value /= shared->shape.logits_divisor;
        }
    }
    ++session_.position_value;
}

void CpuCompiledModel::forward_chunk(std::span<const int32_t> tokens,
                                   bool compute_logits) {
    if (tokens.empty()) return;
    if (session_.position_value < 0 ||
        tokens.size() > static_cast<size_t>(shared->max_context - session_.position_value)) {
        throw std::runtime_error("CPU chunked prefill exceeds context limit");
    }
    // A 32-expert MoE prompt shorter than this produces only a handful of
    // routes per expert. The decode GEMV path is faster for that sparse case;
    // grouped GEMM is reserved for chunks large enough to amortize its setup.
    constexpr size_t kGroupedMoeMinimumRows = 32;
    if (tokens.size() < kGroupedMoeMinimumRows &&
        std::any_of(shared->weight_store.layers.begin(), shared->weight_store.layers.end(),
                    [](const WeightLayer& layer) {
                        return std::holds_alternative<MoeWeights>(layer);
                    })) {
        for (size_t row = 0; row < tokens.size(); ++row) {
            forward_token(tokens[row], compute_logits && row + 1 == tokens.size());
        }
        return;
    }
    const size_t rows = tokens.size();
    const auto chunk_started = Clock::now();
    workspace_.chunk_hidden.resize(rows * shared->shape.hidden);
    workspace_.chunk_residual.resize(rows * shared->shape.hidden);
    workspace_.chunk_normed.resize(rows * shared->shape.hidden);
    workspace_.chunk_op.resize(rows * shared->shape.hidden);
    workspace_.chunk_qkv.resize(rows * shared->shape.qkv_width);
    workspace_.chunk_conv.resize(rows * 3ULL * shared->shape.hidden);
    workspace_.chunk_gate_up.resize(rows * 2ULL * shared->shape.intermediate);
    workspace_.chunk_activated.resize(rows * shared->shape.intermediate);
    workspace_.chunk_mlp.resize(rows * shared->shape.hidden);

    for (size_t row = 0; row < rows; ++row) {
        const int32_t token = tokens[row];
        if (token < 0 || token >= shared->shape.vocab_size) {
            throw std::invalid_argument("CPU chunked prefill token out of range");
        }
    }
    parallel_rows(shared->pool, rows, [&](size_t row) {
        shared->linear.embedding(shared->weight_store.embedding, tokens[row],
            workspace_.chunk_hidden.data() + row * shared->shape.hidden);
        if (shared->shape.embedding_multiplier != 1.0f) {
            float* values = workspace_.chunk_hidden.data() + row * shared->shape.hidden;
            for (int i = 0; i < shared->shape.hidden; ++i) {
                values[i] *= shared->shape.embedding_multiplier;
            }
        }
    });

    const int base_position = session_.position_value;
    for (size_t layer_index = 0; layer_index < shared->weight_store.layers.size(); ++layer_index) {
        const WeightLayer& layer = shared->weight_store.layers[layer_index];
        const CommonWeights& common = common_weights(layer_index);
        std::copy(workspace_.chunk_hidden.begin(), workspace_.chunk_hidden.end(), workspace_.chunk_residual.begin());
        parallel_rows(shared->pool, rows, [&](size_t row) {
            cpu_rmsnorm(workspace_.chunk_hidden.data() + row * shared->shape.hidden,
                        common.operator_norm.data(),
                        workspace_.chunk_normed.data() + row * shared->shape.hidden,
                        shared->shape.hidden, shared->shape.norm_eps);
        });

        if (const auto* attention = attention_operator(layer)) {
            auto started = Clock::now();
            shared->linear.gemm(attention->qkv, workspace_.chunk_normed.data(),
                                workspace_.chunk_qkv.data(), rows);
            session_.prefill_profile.linear_ms += milliseconds_since(started);
            AttentionState& cache = attention_state(layer_index);
            // Normalize/rotate and commit the complete chunk first. Each query
            // still observes only [0, base + row], preserving causality.
            // QK-norm + RoPE is per-row independent; the KV commit that follows
            // appends to the shared page table and stays on this thread.
            parallel_rows(shared->pool, rows, [&](size_t row) {
                float* q = workspace_.chunk_qkv.data() + row * shared->shape.qkv_width;
                float* k = q + shared->shape.q_width;
                const int absolute_position = base_position + static_cast<int>(row);
                if (shared->shape.query_key_norm) {
                    cpu_qk_norm_rope(q, attention->q_norm.data(),
                        shared->shape.num_attention_heads, shared->shape.head_dim,
                        absolute_position, shared->shape.rope_theta,
                        shared->shape.norm_eps);
                    cpu_qk_norm_rope(k, attention->k_norm.data(),
                        shared->shape.num_key_value_heads, shared->shape.head_dim,
                        absolute_position, shared->shape.rope_theta,
                        shared->shape.norm_eps);
                } else {
                    cpu_rope(q, shared->shape.num_attention_heads, shared->shape.head_dim,
                             absolute_position, shared->shape.rope_theta);
                    cpu_rope(k, shared->shape.num_key_value_heads, shared->shape.head_dim,
                             absolute_position, shared->shape.rope_theta);
                    const float ratio = shared->shape.attention_multiplier /
                        (1.0f / std::sqrt(static_cast<float>(shared->shape.head_dim)));
                    for (int i = 0; i < shared->shape.q_width; ++i) q[i] *= ratio;
                }
            });
            for (size_t row = 0; row < rows; ++row) {
                float* q = workspace_.chunk_qkv.data() + row * shared->shape.qkv_width;
                float* k = q + shared->shape.q_width;
                float* v = k + shared->shape.kv_width;
                store_kv(cache, base_position + static_cast<int>(row), k, v);
            }
            const CpuKvPagePool& pool = *shared->kv_pools.at(cache.pool_index);
            started = Clock::now();
            cpu_gqa_prefill_paged(workspace_.chunk_qkv.data(), rows, shared->shape.qkv_width,
                                  pool, cache.pages,
                                  workspace_.chunk_op.data(), base_position,
                                  shared->shape.num_attention_heads,
                                  shared->shape.num_key_value_heads,
                                  shared->shape.head_dim, shared->pool);
            session_.prefill_profile.attention_ms += milliseconds_since(started);
            started = Clock::now();
            shared->linear.gemm(attention->out, workspace_.chunk_op.data(),
                                workspace_.chunk_hidden.data(), rows);
            session_.prefill_profile.linear_ms += milliseconds_since(started);
        } else {
            const auto* convolution = convolution_operator(layer);
            if (!convolution) throw std::logic_error("CPU layer has no operator");
            auto started = Clock::now();
            shared->linear.gemm(convolution->in, workspace_.chunk_normed.data(),
                                workspace_.chunk_conv.data(), rows);
            session_.prefill_profile.linear_ms += milliseconds_since(started);
            ConvolutionState& conv_state = convolution_state(layer_index);
            started = Clock::now();
            cpu_conv_prefill(workspace_.chunk_conv.data(), convolution->weight_tap_major.data(),
                             conv_state.state.data(), workspace_.chunk_op.data(), rows,
                             shared->shape.hidden, shared->shape.conv_cache,
                             base_position, shared->pool);
            session_.prefill_profile.shortconv_ms += milliseconds_since(started);
            started = Clock::now();
            shared->linear.gemm(convolution->out, workspace_.chunk_op.data(),
                                workspace_.chunk_hidden.data(), rows);
            session_.prefill_profile.linear_ms += milliseconds_since(started);
        }

        parallel_rows(shared->pool, rows, [&](size_t row) {
            float* row_hidden = workspace_.chunk_hidden.data() + row * shared->shape.hidden;
            if (shared->shape.residual_multiplier != 1.0f) {
                for (int i = 0; i < shared->shape.hidden; ++i) {
                    row_hidden[i] *= shared->shape.residual_multiplier;
                }
            }
            cpu_residual_add(row_hidden,
                workspace_.chunk_residual.data() + row * shared->shape.hidden,
                shared->shape.hidden);
            cpu_rmsnorm(row_hidden, common.ffn_norm.data(),
                workspace_.chunk_normed.data() + row * shared->shape.hidden,
                shared->shape.hidden, shared->shape.norm_eps);
        });
        if (const auto* moe = std::get_if<MoeWeights>(&layer)) {
            const int E = moe->num_experts;
            const int K = moe->experts_per_token;
            const int moe_inter = shared->shape.moe_intermediate > 0
                ? shared->shape.moe_intermediate : shared->shape.intermediate;
            if (E <= 0 || K <= 0 || K > E) {
                throw std::logic_error("CPU MoE layer has invalid routing dimensions");
            }

            auto started = Clock::now();
            workspace_.moe_router_logits.resize(rows * static_cast<size_t>(E));
            shared->linear.gemm_raw(moe->router.data(), workspace_.chunk_normed.data(),
                                    workspace_.moe_router_logits.data(), rows, E,
                                    shared->shape.hidden);
            workspace_.moe_router_probs.resize(static_cast<size_t>(E));
            workspace_.moe_router_scored.resize(static_cast<size_t>(E));
            workspace_.moe_route_rows.resize(rows * static_cast<size_t>(K));
            workspace_.moe_route_experts.resize(rows * static_cast<size_t>(K));
            workspace_.moe_route_weights.resize(rows * static_cast<size_t>(K));
            for (size_t row = 0; row < rows; ++row) {
                const float* logits = workspace_.moe_router_logits.data() + row * static_cast<size_t>(E);
                for (int expert = 0; expert < E; ++expert) {
                    const float probability = moe_sigmoid(logits[expert]);
                    workspace_.moe_router_probs[static_cast<size_t>(expert)] = probability;
                    const float bias = moe->use_expert_bias &&
                        expert < static_cast<int>(moe->router_bias.size())
                        ? moe->router_bias[static_cast<size_t>(expert)] : 0.0f;
                    workspace_.moe_router_scored[static_cast<size_t>(expert)] = {probability + bias, expert};
                }
                std::partial_sort(workspace_.moe_router_scored.begin(),
                    workspace_.moe_router_scored.begin() + K, workspace_.moe_router_scored.end(),
                    [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                        return a.first != b.first ? a.first > b.first : a.second < b.second;
                    });
                float weight_sum = 0.0f;
                const size_t route_base = row * static_cast<size_t>(K);
                for (int slot = 0; slot < K; ++slot) {
                    const int expert = workspace_.moe_router_scored[static_cast<size_t>(slot)].second;
                    workspace_.moe_route_rows[route_base + static_cast<size_t>(slot)] = static_cast<int>(row);
                    workspace_.moe_route_experts[route_base + static_cast<size_t>(slot)] = expert;
                    workspace_.moe_route_weights[route_base + static_cast<size_t>(slot)] =
                        workspace_.moe_router_probs[static_cast<size_t>(expert)];
                    weight_sum += workspace_.moe_route_weights[route_base + static_cast<size_t>(slot)];
                }
                const float scale = moe->routed_scaling_factor *
                    (moe->normalize_topk ? 1.0f / (weight_sum + 1e-6f) : 1.0f);
                for (int slot = 0; slot < K; ++slot) {
                    workspace_.moe_route_weights[route_base + static_cast<size_t>(slot)] *= scale;
                }
            }
            session_.prefill_profile.moe_router_ms += milliseconds_since(started);

            started = Clock::now();
            const size_t route_count = rows * static_cast<size_t>(K);
            workspace_.moe_group_offsets.assign(static_cast<size_t>(E) + 1, 0);
            for (int expert : workspace_.moe_route_experts) ++workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1];
            for (int expert = 0; expert < E; ++expert) {
                workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1] +=
                    workspace_.moe_group_offsets[static_cast<size_t>(expert)];
            }
            workspace_.moe_group_cursor = workspace_.moe_group_offsets;
            workspace_.moe_route_order.resize(route_count);
            for (size_t route = 0; route < route_count; ++route) {
                const int expert = workspace_.moe_route_experts[route];
                workspace_.moe_route_order[workspace_.moe_group_cursor[static_cast<size_t>(expert)]++] = route;
            }

            // Pack every routed row once in expert-group order. The two FFN
            // stages then share one cross-expert work queue instead of paying
            // a thread-pool barrier for each sparse expert group.
            workspace_.moe_gathered_normed.resize(route_count * static_cast<size_t>(shared->shape.hidden));
            workspace_.moe_gathered_gate_up.resize(route_count * 2ULL * static_cast<size_t>(moe_inter));
            workspace_.moe_gathered_activated.resize(route_count * static_cast<size_t>(moe_inter));
            workspace_.moe_gathered_output.resize(route_count * static_cast<size_t>(shared->shape.hidden));
            workspace_.moe_gemm_jobs.clear();
            workspace_.moe_gemm_jobs.reserve(static_cast<size_t>(E));
            std::fill(workspace_.chunk_mlp.begin(), workspace_.chunk_mlp.end(), 0.0f);
            for (int expert = 0; expert < E; ++expert) {
                const size_t begin = workspace_.moe_group_offsets[static_cast<size_t>(expert)];
                const size_t end = workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1];
                const size_t group_rows = end - begin;
                if (group_rows == 0) continue;
                for (size_t group_row = 0; group_row < group_rows; ++group_row) {
                    const size_t route = workspace_.moe_route_order[begin + group_row];
                    const int row = workspace_.moe_route_rows[route];
                    std::copy_n(workspace_.chunk_normed.data() + static_cast<size_t>(row) * shared->shape.hidden,
                                shared->shape.hidden,
                                workspace_.moe_gathered_normed.data() + (begin + group_row) * shared->shape.hidden);
                }
                workspace_.moe_gemm_jobs.push_back({&moe->expert_w13[static_cast<size_t>(expert)],
                                         begin, group_rows});
            }
            auto linear_started = Clock::now();
            shared->linear.gemm_grouped(workspace_.moe_gemm_jobs, workspace_.moe_gathered_normed.data(),
                                        workspace_.moe_gathered_gate_up.data());
            parallel_rows(shared->pool, route_count, [&](size_t group_row) {
                cpu_swiglu(workspace_.moe_gathered_gate_up.data() + group_row * 2ULL * moe_inter,
                           workspace_.moe_gathered_activated.data() + group_row * moe_inter, moe_inter);
            });
            size_t job_index = 0;
            for (int expert = 0; expert < E; ++expert) {
                const size_t begin = workspace_.moe_group_offsets[static_cast<size_t>(expert)];
                const size_t end = workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1];
                if (begin != end) {
                    workspace_.moe_gemm_jobs[job_index++].weight =
                        &moe->expert_w2[static_cast<size_t>(expert)];
                }
            }
            shared->linear.gemm_grouped(workspace_.moe_gemm_jobs, workspace_.moe_gathered_activated.data(),
                                        workspace_.moe_gathered_output.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            for (int expert = 0; expert < E; ++expert) {
                const size_t begin = workspace_.moe_group_offsets[static_cast<size_t>(expert)];
                const size_t end = workspace_.moe_group_offsets[static_cast<size_t>(expert) + 1];
                const size_t group_rows = end - begin;
                if (group_rows == 0) continue;
                for (size_t group_row = 0; group_row < group_rows; ++group_row) {
                    const size_t route = workspace_.moe_route_order[begin + group_row];
                    float* output = workspace_.chunk_mlp.data() +
                        static_cast<size_t>(workspace_.moe_route_rows[route]) * shared->shape.hidden;
                    const float* expert_output = workspace_.moe_gathered_output.data() +
                        (begin + group_row) * shared->shape.hidden;
                    const float weight = workspace_.moe_route_weights[route];
                    for (int column = 0; column < shared->shape.hidden; ++column) {
                        output[column] += weight * expert_output[column];
                    }
                }
            }
            session_.prefill_profile.moe_expert_ms += milliseconds_since(started);
        } else {
            auto started = Clock::now();
            shared->linear.gemm(common.w13, workspace_.chunk_normed.data(),
                                workspace_.chunk_gate_up.data(), rows);
            session_.prefill_profile.linear_ms += milliseconds_since(started);
            parallel_rows(shared->pool, rows, [&](size_t row) {
                cpu_swiglu(workspace_.chunk_gate_up.data() + row * 2ULL * shared->shape.intermediate,
                           workspace_.chunk_activated.data() + row * shared->shape.intermediate,
                           shared->shape.intermediate);
            });
            started = Clock::now();
            shared->linear.gemm(common.w2, workspace_.chunk_activated.data(),
                                workspace_.chunk_mlp.data(), rows);
            session_.prefill_profile.linear_ms += milliseconds_since(started);
        }
        parallel_rows(shared->pool, rows, [&](size_t row) {
            if (shared->shape.residual_multiplier != 1.0f) {
                float* output = workspace_.chunk_mlp.data() + row * shared->shape.hidden;
                for (int i = 0; i < shared->shape.hidden; ++i) {
                    output[i] *= shared->shape.residual_multiplier;
                }
            }
            cpu_residual_add(workspace_.chunk_hidden.data() + row * shared->shape.hidden,
                             workspace_.chunk_mlp.data() + row * shared->shape.hidden,
                             shared->shape.hidden);
        });
    }

    if (compute_logits) {
        const float* last = workspace_.chunk_hidden.data() + (rows - 1) * shared->shape.hidden;
        cpu_rmsnorm(last, shared->weight_store.final_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->weight_store.embedding :
                            shared->weight_store.lm_head, workspace_.normed.data(), workspace_.logits.data());
        if (shared->shape.logits_divisor != 1.0f) {
            for (float& value : workspace_.logits) value /= shared->shape.logits_divisor;
        }
    }
    session_.position_value += static_cast<int>(rows);
    session_.prefill_profile.total_ms += milliseconds_since(chunk_started);
}

} // namespace celeg

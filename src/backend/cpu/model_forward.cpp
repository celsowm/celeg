#include "detail/model_internal.hpp"

#include "celeg/runtime/moe.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace celeg {

namespace {
using Clock = std::chrono::steady_clock;
double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Prefill runs the row-wise elementwise passes (RMSNorm, SwiGLU, residual add)
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

void CpuModel::Impl::forward_token(int32_t token, bool compute_logits) {
    if (position_value >= shared->max_context) {
        throw std::runtime_error("CPU context limit reached");
    }
    shared->linear.embedding(shared->embedding, token, hidden.data());
    if (shared->shape.embedding_multiplier != 1.0f) {
        for (float& value : hidden) value *= shared->shape.embedding_multiplier;
    }
    for (size_t index = 0; index < shared->layers.size(); ++index) {
        const WeightLayer& layer_variant = shared->layers[index];
        const CommonWeights& common = common_weights(index);
        std::copy(hidden.begin(), hidden.end(), residual.begin());
        cpu_rmsnorm(hidden.data(), common.operator_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        if (const auto* attention = attention_operator(layer_variant)) {
            shared->linear.gemv(attention->qkv, normed.data(), qkv.data());
            float* q = qkv.data();
            float* k = q + shared->shape.q_width;
            float* v = k + shared->shape.kv_width;
            if (shared->shape.query_key_norm) {
                cpu_qk_norm_rope(q, attention->q_norm.data(), shared->shape.num_attention_heads,
                    shared->shape.head_dim, position_value, shared->shape.rope_theta,
                    shared->shape.norm_eps);
                cpu_qk_norm_rope(k, attention->k_norm.data(), shared->shape.num_key_value_heads,
                    shared->shape.head_dim, position_value, shared->shape.rope_theta,
                    shared->shape.norm_eps);
            } else {
                cpu_rope(q, shared->shape.num_attention_heads, shared->shape.head_dim,
                         position_value, shared->shape.rope_theta);
                cpu_rope(k, shared->shape.num_key_value_heads, shared->shape.head_dim,
                         position_value, shared->shape.rope_theta);
                const float ratio = shared->shape.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(shared->shape.head_dim)));
                for (int i = 0; i < shared->shape.q_width; ++i) q[i] *= ratio;
            }
            AttentionState& state = attention_state(index);
            store_kv(state, position_value, k, v);
            run_attention(state, q, op_output.data(), position_value + 1);
            shared->linear.gemv(attention->out, op_output.data(), hidden.data());
        } else {
            const auto* convolution = convolution_operator(layer_variant);
            if (!convolution) throw std::logic_error("CPU layer has no operator");
            ConvolutionState& state = convolution_state(index);
            shared->linear.gemv(convolution->in, normed.data(), conv_projected.data());
            cpu_conv_decode(conv_projected.data(), convolution->weight_tap_major.data(),
                state.state.data(), op_output.data(), shared->shape.hidden,
                shared->shape.conv_cache, position_value);
            shared->linear.gemv(convolution->out, op_output.data(), hidden.data());
        }
        if (shared->shape.residual_multiplier != 1.0f) {
            for (float& value : hidden) value *= shared->shape.residual_multiplier;
        }
        cpu_residual_add(hidden.data(), residual.data(), shared->shape.hidden);

        cpu_rmsnorm(hidden.data(), common.ffn_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);

        if (const auto* moe = std::get_if<MoeWeights>(&layer_variant)) {
            // MoE FFN: router -> top-K expert selection -> per-expert FFN GEMVs.
            const int E = moe->num_experts;
            const int K = moe->experts_per_token;
            const int moe_inter = shared->shape.moe_intermediate > 0
                ? shared->shape.moe_intermediate : shared->shape.intermediate;

            const bool profile_moe = phase == SessionPhase::Prefilling;
            auto started = Clock::now();
            moe_router_logits.resize(static_cast<size_t>(E));
            moe_router_probs.resize(static_cast<size_t>(E));
            moe_router_scored.resize(static_cast<size_t>(E));
            moe_selected.resize(static_cast<size_t>(K));
            moe_weights.resize(static_cast<size_t>(K));

            // Router GEMV: normed @ router^T -> logits [E]
            shared->linear.gemv_raw(moe->router.data(), normed.data(), moe_router_logits.data(),
                                    E, shared->shape.hidden);

            // Sigmoid + optional expert bias + top-K selection.
            for (int e = 0; e < E; ++e) {
                moe_router_probs[static_cast<size_t>(e)] =
                    moe_sigmoid(moe_router_logits[static_cast<size_t>(e)]);
                float score = moe_router_probs[static_cast<size_t>(e)];
                if (moe->use_expert_bias && e < static_cast<int>(moe->router_bias.size())) {
                    score += moe->router_bias[static_cast<size_t>(e)];
                }
                moe_router_scored[static_cast<size_t>(e)] = {score, e};
            }
            std::partial_sort(moe_router_scored.begin(), moe_router_scored.begin() + K,
                moe_router_scored.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second;
                });

            // Gather selected experts and routing weights.
            float weight_sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                moe_selected[static_cast<size_t>(k)] =
                    moe_router_scored[static_cast<size_t>(k)].second;
                moe_weights[static_cast<size_t>(k)] =
                    moe_router_probs[static_cast<size_t>(moe_selected[static_cast<size_t>(k)])];
                weight_sum += moe_weights[static_cast<size_t>(k)];
            }
            if (moe->normalize_topk) {
                const float inv = 1.0f / (weight_sum + 1e-6f);
                for (int k = 0; k < K; ++k) {
                    moe_weights[static_cast<size_t>(k)] *= inv;
                }
            }
            for (int k = 0; k < K; ++k) {
                moe_weights[static_cast<size_t>(k)] *= moe->routed_scaling_factor;
            }
            if (profile_moe) prefill_profile.moe_router_ms += milliseconds_since(started);

            // Per-expert FFN GEMVs with routing-weighted accumulation.
            started = Clock::now();
            std::fill(mlp_output.begin(), mlp_output.end(), 0.0f);
            for (int k = 0; k < K; ++k) {
                const int expert = moe_selected[static_cast<size_t>(k)];
                const float rw = moe_weights[static_cast<size_t>(k)];
                if (expert < 0 || expert >= E) continue;

                // gate_up = expert_w13[expert] @ normed  [2 * moe_inter]
                shared->linear.gemv(moe->expert_w13[static_cast<size_t>(expert)],
                                    normed.data(), gate_up.data());
                // SwiGLU: activated[i] = swiglu(gate_up[i], gate_up[moe_inter + i])
                cpu_swiglu(gate_up.data(), activated.data(), moe_inter);
                // mlp_output += rw * expert_w2[expert] @ activated
                shared->linear.gemv(moe->expert_w2[static_cast<size_t>(expert)],
                                    activated.data(), op_output.data());
                for (int j = 0; j < shared->shape.hidden; ++j) {
                    mlp_output[static_cast<size_t>(j)] += rw * op_output[static_cast<size_t>(j)];
                }
            }
            if (shared->shape.residual_multiplier != 1.0f) {
                for (float& value : mlp_output) value *= shared->shape.residual_multiplier;
            }
            cpu_residual_add(hidden.data(), mlp_output.data(), shared->shape.hidden);
            if (profile_moe) prefill_profile.moe_expert_ms += milliseconds_since(started);
        } else {
            // Dense FFN path.
            shared->linear.gemv(common.w13, normed.data(), gate_up.data());
            cpu_swiglu(gate_up.data(), activated.data(), shared->shape.intermediate);
            shared->linear.gemv(common.w2, activated.data(), mlp_output.data());
            if (shared->shape.residual_multiplier != 1.0f) {
                for (float& value : mlp_output) value *= shared->shape.residual_multiplier;
            }
            cpu_residual_add(hidden.data(), mlp_output.data(), shared->shape.hidden);
        }
    }
    if (compute_logits) {
        cpu_rmsnorm(hidden.data(), shared->final_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->embedding :
                            shared->lm_head, normed.data(), logits.data());
        if (shared->shape.logits_divisor != 1.0f) {
            for (float& value : logits) value /= shared->shape.logits_divisor;
        }
    }
    ++position_value;
}

void CpuModel::Impl::forward_chunk(std::span<const int32_t> tokens,
                                   bool compute_logits) {
    if (tokens.empty()) return;
    if (position_value < 0 ||
        tokens.size() > static_cast<size_t>(shared->max_context - position_value)) {
        throw std::runtime_error("CPU chunked prefill exceeds context limit");
    }
    // A 32-expert MoE prompt shorter than this produces only a handful of
    // routes per expert. The decode GEMV path is faster for that sparse case;
    // grouped GEMM is reserved for chunks large enough to amortize its setup.
    constexpr size_t kGroupedMoeMinimumRows = 32;
    if (tokens.size() < kGroupedMoeMinimumRows &&
        std::any_of(shared->layers.begin(), shared->layers.end(),
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
    chunk_hidden.resize(rows * shared->shape.hidden);
    chunk_residual.resize(rows * shared->shape.hidden);
    chunk_normed.resize(rows * shared->shape.hidden);
    chunk_op.resize(rows * shared->shape.hidden);
    chunk_qkv.resize(rows * shared->shape.qkv_width);
    chunk_conv.resize(rows * 3ULL * shared->shape.hidden);
    chunk_gate_up.resize(rows * 2ULL * shared->shape.intermediate);
    chunk_activated.resize(rows * shared->shape.intermediate);
    chunk_mlp.resize(rows * shared->shape.hidden);

    for (size_t row = 0; row < rows; ++row) {
        const int32_t token = tokens[row];
        if (token < 0 || token >= shared->shape.vocab_size) {
            throw std::invalid_argument("CPU chunked prefill token out of range");
        }
    }
    parallel_rows(shared->pool, rows, [&](size_t row) {
        shared->linear.embedding(shared->embedding, tokens[row],
            chunk_hidden.data() + row * shared->shape.hidden);
        if (shared->shape.embedding_multiplier != 1.0f) {
            float* values = chunk_hidden.data() + row * shared->shape.hidden;
            for (int i = 0; i < shared->shape.hidden; ++i) {
                values[i] *= shared->shape.embedding_multiplier;
            }
        }
    });

    const int base_position = position_value;
    for (size_t layer_index = 0; layer_index < shared->layers.size(); ++layer_index) {
        const WeightLayer& layer = shared->layers[layer_index];
        const CommonWeights& common = common_weights(layer_index);
        std::copy(chunk_hidden.begin(), chunk_hidden.end(), chunk_residual.begin());
        parallel_rows(shared->pool, rows, [&](size_t row) {
            cpu_rmsnorm(chunk_hidden.data() + row * shared->shape.hidden,
                        common.operator_norm.data(),
                        chunk_normed.data() + row * shared->shape.hidden,
                        shared->shape.hidden, shared->shape.norm_eps);
        });

        if (const auto* attention = attention_operator(layer)) {
            auto started = Clock::now();
            shared->linear.gemm(attention->qkv, chunk_normed.data(),
                                chunk_qkv.data(), rows);
            prefill_profile.linear_ms += milliseconds_since(started);
            AttentionState& cache = attention_state(layer_index);
            // Normalize/rotate and commit the complete chunk first. Each query
            // still observes only [0, base + row], preserving causality.
            // QK-norm + RoPE is per-row independent; the KV commit that follows
            // appends to the shared page table and stays on this thread.
            parallel_rows(shared->pool, rows, [&](size_t row) {
                float* q = chunk_qkv.data() + row * shared->shape.qkv_width;
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
                float* q = chunk_qkv.data() + row * shared->shape.qkv_width;
                float* k = q + shared->shape.q_width;
                float* v = k + shared->shape.kv_width;
                store_kv(cache, base_position + static_cast<int>(row), k, v);
            }
            const CpuKvPagePool& pool = *shared->kv_pools.at(cache.pool_index);
            started = Clock::now();
            cpu_gqa_prefill_paged(chunk_qkv.data(), rows, shared->shape.qkv_width,
                                  pool, cache.pages,
                                  chunk_op.data(), base_position,
                                  shared->shape.num_attention_heads,
                                  shared->shape.num_key_value_heads,
                                  shared->shape.head_dim, shared->pool);
            prefill_profile.attention_ms += milliseconds_since(started);
            started = Clock::now();
            shared->linear.gemm(attention->out, chunk_op.data(),
                                chunk_hidden.data(), rows);
            prefill_profile.linear_ms += milliseconds_since(started);
        } else {
            const auto* convolution = convolution_operator(layer);
            if (!convolution) throw std::logic_error("CPU layer has no operator");
            auto started = Clock::now();
            shared->linear.gemm(convolution->in, chunk_normed.data(),
                                chunk_conv.data(), rows);
            prefill_profile.linear_ms += milliseconds_since(started);
            ConvolutionState& conv_state = convolution_state(layer_index);
            started = Clock::now();
            cpu_conv_prefill(chunk_conv.data(), convolution->weight_tap_major.data(),
                             conv_state.state.data(), chunk_op.data(), rows,
                             shared->shape.hidden, shared->shape.conv_cache,
                             base_position, shared->pool);
            prefill_profile.shortconv_ms += milliseconds_since(started);
            started = Clock::now();
            shared->linear.gemm(convolution->out, chunk_op.data(),
                                chunk_hidden.data(), rows);
            prefill_profile.linear_ms += milliseconds_since(started);
        }

        parallel_rows(shared->pool, rows, [&](size_t row) {
            float* row_hidden = chunk_hidden.data() + row * shared->shape.hidden;
            if (shared->shape.residual_multiplier != 1.0f) {
                for (int i = 0; i < shared->shape.hidden; ++i) {
                    row_hidden[i] *= shared->shape.residual_multiplier;
                }
            }
            cpu_residual_add(row_hidden,
                chunk_residual.data() + row * shared->shape.hidden,
                shared->shape.hidden);
            cpu_rmsnorm(row_hidden, common.ffn_norm.data(),
                chunk_normed.data() + row * shared->shape.hidden,
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
            moe_router_logits.resize(rows * static_cast<size_t>(E));
            shared->linear.gemm_raw(moe->router.data(), chunk_normed.data(),
                                    moe_router_logits.data(), rows, E,
                                    shared->shape.hidden);
            moe_router_probs.resize(static_cast<size_t>(E));
            moe_router_scored.resize(static_cast<size_t>(E));
            moe_route_rows.resize(rows * static_cast<size_t>(K));
            moe_route_experts.resize(rows * static_cast<size_t>(K));
            moe_route_weights.resize(rows * static_cast<size_t>(K));
            for (size_t row = 0; row < rows; ++row) {
                const float* logits = moe_router_logits.data() + row * static_cast<size_t>(E);
                for (int expert = 0; expert < E; ++expert) {
                    const float probability = moe_sigmoid(logits[expert]);
                    moe_router_probs[static_cast<size_t>(expert)] = probability;
                    const float bias = moe->use_expert_bias &&
                        expert < static_cast<int>(moe->router_bias.size())
                        ? moe->router_bias[static_cast<size_t>(expert)] : 0.0f;
                    moe_router_scored[static_cast<size_t>(expert)] = {probability + bias, expert};
                }
                std::partial_sort(moe_router_scored.begin(),
                    moe_router_scored.begin() + K, moe_router_scored.end(),
                    [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                        return a.first != b.first ? a.first > b.first : a.second < b.second;
                    });
                float weight_sum = 0.0f;
                const size_t route_base = row * static_cast<size_t>(K);
                for (int slot = 0; slot < K; ++slot) {
                    const int expert = moe_router_scored[static_cast<size_t>(slot)].second;
                    moe_route_rows[route_base + static_cast<size_t>(slot)] = static_cast<int>(row);
                    moe_route_experts[route_base + static_cast<size_t>(slot)] = expert;
                    moe_route_weights[route_base + static_cast<size_t>(slot)] =
                        moe_router_probs[static_cast<size_t>(expert)];
                    weight_sum += moe_route_weights[route_base + static_cast<size_t>(slot)];
                }
                const float scale = moe->routed_scaling_factor *
                    (moe->normalize_topk ? 1.0f / (weight_sum + 1e-6f) : 1.0f);
                for (int slot = 0; slot < K; ++slot) {
                    moe_route_weights[route_base + static_cast<size_t>(slot)] *= scale;
                }
            }
            prefill_profile.moe_router_ms += milliseconds_since(started);

            started = Clock::now();
            const size_t route_count = rows * static_cast<size_t>(K);
            moe_group_offsets.assign(static_cast<size_t>(E) + 1, 0);
            for (int expert : moe_route_experts) ++moe_group_offsets[static_cast<size_t>(expert) + 1];
            for (int expert = 0; expert < E; ++expert) {
                moe_group_offsets[static_cast<size_t>(expert) + 1] +=
                    moe_group_offsets[static_cast<size_t>(expert)];
            }
            moe_group_cursor = moe_group_offsets;
            moe_route_order.resize(route_count);
            for (size_t route = 0; route < route_count; ++route) {
                const int expert = moe_route_experts[route];
                moe_route_order[moe_group_cursor[static_cast<size_t>(expert)]++] = route;
            }

            // Pack every routed row once in expert-group order. The two FFN
            // stages then share one cross-expert work queue instead of paying
            // a thread-pool barrier for each sparse expert group.
            moe_gathered_normed.resize(route_count * static_cast<size_t>(shared->shape.hidden));
            moe_gathered_gate_up.resize(route_count * 2ULL * static_cast<size_t>(moe_inter));
            moe_gathered_activated.resize(route_count * static_cast<size_t>(moe_inter));
            moe_gathered_output.resize(route_count * static_cast<size_t>(shared->shape.hidden));
            moe_gemm_jobs.clear();
            moe_gemm_jobs.reserve(static_cast<size_t>(E));
            std::fill(chunk_mlp.begin(), chunk_mlp.end(), 0.0f);
            for (int expert = 0; expert < E; ++expert) {
                const size_t begin = moe_group_offsets[static_cast<size_t>(expert)];
                const size_t end = moe_group_offsets[static_cast<size_t>(expert) + 1];
                const size_t group_rows = end - begin;
                if (group_rows == 0) continue;
                for (size_t group_row = 0; group_row < group_rows; ++group_row) {
                    const size_t route = moe_route_order[begin + group_row];
                    const int row = moe_route_rows[route];
                    std::copy_n(chunk_normed.data() + static_cast<size_t>(row) * shared->shape.hidden,
                                shared->shape.hidden,
                                moe_gathered_normed.data() + (begin + group_row) * shared->shape.hidden);
                }
                moe_gemm_jobs.push_back({&moe->expert_w13[static_cast<size_t>(expert)],
                                         begin, group_rows});
            }
            auto linear_started = Clock::now();
            shared->linear.gemm_grouped(moe_gemm_jobs, moe_gathered_normed.data(),
                                        moe_gathered_gate_up.data());
            parallel_rows(shared->pool, route_count, [&](size_t group_row) {
                cpu_swiglu(moe_gathered_gate_up.data() + group_row * 2ULL * moe_inter,
                           moe_gathered_activated.data() + group_row * moe_inter, moe_inter);
            });
            size_t job_index = 0;
            for (int expert = 0; expert < E; ++expert) {
                const size_t begin = moe_group_offsets[static_cast<size_t>(expert)];
                const size_t end = moe_group_offsets[static_cast<size_t>(expert) + 1];
                if (begin != end) {
                    moe_gemm_jobs[job_index++].weight =
                        &moe->expert_w2[static_cast<size_t>(expert)];
                }
            }
            shared->linear.gemm_grouped(moe_gemm_jobs, moe_gathered_activated.data(),
                                        moe_gathered_output.data());
            prefill_profile.linear_ms += milliseconds_since(linear_started);
            for (int expert = 0; expert < E; ++expert) {
                const size_t begin = moe_group_offsets[static_cast<size_t>(expert)];
                const size_t end = moe_group_offsets[static_cast<size_t>(expert) + 1];
                const size_t group_rows = end - begin;
                if (group_rows == 0) continue;
                for (size_t group_row = 0; group_row < group_rows; ++group_row) {
                    const size_t route = moe_route_order[begin + group_row];
                    float* output = chunk_mlp.data() +
                        static_cast<size_t>(moe_route_rows[route]) * shared->shape.hidden;
                    const float* expert_output = moe_gathered_output.data() +
                        (begin + group_row) * shared->shape.hidden;
                    const float weight = moe_route_weights[route];
                    for (int column = 0; column < shared->shape.hidden; ++column) {
                        output[column] += weight * expert_output[column];
                    }
                }
            }
            prefill_profile.moe_expert_ms += milliseconds_since(started);
        } else {
            auto started = Clock::now();
            shared->linear.gemm(common.w13, chunk_normed.data(),
                                chunk_gate_up.data(), rows);
            prefill_profile.linear_ms += milliseconds_since(started);
            parallel_rows(shared->pool, rows, [&](size_t row) {
                cpu_swiglu(chunk_gate_up.data() + row * 2ULL * shared->shape.intermediate,
                           chunk_activated.data() + row * shared->shape.intermediate,
                           shared->shape.intermediate);
            });
            started = Clock::now();
            shared->linear.gemm(common.w2, chunk_activated.data(),
                                chunk_mlp.data(), rows);
            prefill_profile.linear_ms += milliseconds_since(started);
        }
        parallel_rows(shared->pool, rows, [&](size_t row) {
            if (shared->shape.residual_multiplier != 1.0f) {
                float* output = chunk_mlp.data() + row * shared->shape.hidden;
                for (int i = 0; i < shared->shape.hidden; ++i) {
                    output[i] *= shared->shape.residual_multiplier;
                }
            }
            cpu_residual_add(chunk_hidden.data() + row * shared->shape.hidden,
                             chunk_mlp.data() + row * shared->shape.hidden,
                             shared->shape.hidden);
        });
    }

    if (compute_logits) {
        const float* last = chunk_hidden.data() + (rows - 1) * shared->shape.hidden;
        cpu_rmsnorm(last, shared->final_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->embedding :
                            shared->lm_head, normed.data(), logits.data());
        if (shared->shape.logits_divisor != 1.0f) {
            for (float& value : logits) value /= shared->shape.logits_divisor;
        }
    }
    position_value += static_cast<int>(rows);
    prefill_profile.total_ms += milliseconds_since(chunk_started);
}

} // namespace celeg

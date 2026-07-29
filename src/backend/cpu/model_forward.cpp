#include "detail/model_internal.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace lfm {

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
    for (size_t index = 0; index < shared->layers.size(); ++index) {
        const WeightLayer& layer_variant = shared->layers[index];
        const CommonWeights& common = common_weights(index);
        std::copy(hidden.begin(), hidden.end(), residual.begin());
        cpu_rmsnorm(hidden.data(), common.operator_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        if (const auto* attention = std::get_if<AttentionWeights>(&layer_variant)) {
            shared->linear.gemv(attention->qkv, normed.data(), qkv.data());
            float* q = qkv.data();
            float* k = q + shared->shape.q_width;
            float* v = k + shared->shape.kv_width;
            cpu_qk_norm_rope(q, attention->q_norm.data(), shared->shape.num_attention_heads,
                shared->shape.head_dim, position_value, shared->shape.rope_theta,
                shared->shape.norm_eps);
            cpu_qk_norm_rope(k, attention->k_norm.data(), shared->shape.num_key_value_heads,
                shared->shape.head_dim, position_value, shared->shape.rope_theta,
                shared->shape.norm_eps);
            AttentionState& state = attention_state(index);
            store_kv(state, position_value, k, v);
            run_attention(state, q, op_output.data(), position_value + 1);
            shared->linear.gemv(attention->out, op_output.data(), hidden.data());
        } else {
            const auto& convolution = std::get<ConvolutionWeights>(layer_variant);
            ConvolutionState& state = convolution_state(index);
            shared->linear.gemv(convolution.in, normed.data(), conv_projected.data());
            cpu_conv_decode(conv_projected.data(), convolution.weight_tap_major.data(),
                state.state.data(), op_output.data(), shared->shape.hidden,
                shared->shape.conv_cache, position_value);
            shared->linear.gemv(convolution.out, op_output.data(), hidden.data());
        }
        cpu_residual_add(hidden.data(), residual.data(), shared->shape.hidden);

        cpu_rmsnorm(hidden.data(), common.ffn_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(common.w13, normed.data(), gate_up.data());
        cpu_swiglu(gate_up.data(), activated.data(), shared->shape.intermediate);
        shared->linear.gemv(common.w2, activated.data(), mlp_output.data());
        cpu_residual_add(hidden.data(), mlp_output.data(), shared->shape.hidden);
    }
    if (compute_logits) {
        cpu_rmsnorm(hidden.data(), shared->final_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(shared->tie_word_embeddings ? shared->embedding :
                            shared->lm_head, normed.data(), logits.data());
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

        if (const auto* attention = std::get_if<AttentionWeights>(&layer)) {
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
                cpu_qk_norm_rope(q, attention->q_norm.data(),
                    shared->shape.num_attention_heads, shared->shape.head_dim,
                    absolute_position, shared->shape.rope_theta,
                    shared->shape.norm_eps);
                cpu_qk_norm_rope(k, attention->k_norm.data(),
                    shared->shape.num_key_value_heads, shared->shape.head_dim,
                    absolute_position, shared->shape.rope_theta,
                    shared->shape.norm_eps);
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
            const auto& convolution = std::get<ConvolutionWeights>(layer);
            auto started = Clock::now();
            shared->linear.gemm(convolution.in, chunk_normed.data(),
                                chunk_conv.data(), rows);
            prefill_profile.linear_ms += milliseconds_since(started);
            ConvolutionState& conv_state = convolution_state(layer_index);
            started = Clock::now();
            cpu_conv_prefill(chunk_conv.data(), convolution.weight_tap_major.data(),
                             conv_state.state.data(), chunk_op.data(), rows,
                             shared->shape.hidden, shared->shape.conv_cache,
                             base_position, shared->pool);
            prefill_profile.shortconv_ms += milliseconds_since(started);
            started = Clock::now();
            shared->linear.gemm(convolution.out, chunk_op.data(),
                                chunk_hidden.data(), rows);
            prefill_profile.linear_ms += milliseconds_since(started);
        }

        parallel_rows(shared->pool, rows, [&](size_t row) {
            float* row_hidden = chunk_hidden.data() + row * shared->shape.hidden;
            cpu_residual_add(row_hidden,
                chunk_residual.data() + row * shared->shape.hidden,
                shared->shape.hidden);
            cpu_rmsnorm(row_hidden, common.ffn_norm.data(),
                chunk_normed.data() + row * shared->shape.hidden,
                shared->shape.hidden, shared->shape.norm_eps);
        });
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
        parallel_rows(shared->pool, rows, [&](size_t row) {
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
    }
    position_value += static_cast<int>(rows);
    prefill_profile.total_ms += milliseconds_since(chunk_started);
}

} // namespace lfm

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
    if (shared->shape.has_per_layer_input) {
        const size_t packed = static_cast<size_t>(shared->shape.num_hidden_layers) *
            static_cast<size_t>(shared->shape.per_layer_input_size);
        workspace_.per_layer_input.resize(packed);
        workspace_.per_layer_context.resize(packed);
        workspace_.per_layer_gate.resize(static_cast<size_t>(shared->shape.per_layer_input_size));
        shared->linear.embedding(shared->weight_store.per_layer_embedding, token,
                                 workspace_.per_layer_input.data());
        const float token_scale = std::sqrt(static_cast<float>(shared->shape.per_layer_input_size));
        for (float& value : workspace_.per_layer_input) value *= token_scale;
        shared->linear.gemv(shared->weight_store.per_layer_context_projection,
                            workspace_.hidden.data(), workspace_.per_layer_context.data());
        const float context_scale = 1.0f / std::sqrt(static_cast<float>(shared->shape.hidden));
        for (float& value : workspace_.per_layer_context) value *= context_scale;
        for (int layer = 0; layer < shared->shape.num_hidden_layers; ++layer) {
            float* context = workspace_.per_layer_context.data() +
                static_cast<size_t>(layer) * shared->shape.per_layer_input_size;
            cpu_rmsnorm_inplace(context,
                shared->weight_store.per_layer_projection_norm.data(),
                shared->shape.per_layer_input_size, shared->shape.norm_eps);
            float* token_values = workspace_.per_layer_input.data() +
                static_cast<size_t>(layer) * shared->shape.per_layer_input_size;
            for (int d = 0; d < shared->shape.per_layer_input_size; ++d) {
                context[d] = (context[d] + token_values[d]) * 0.7071067811865475f;
            }
        }
    }
    for (size_t index = 0; index < shared->weight_store.layers.size(); ++index) {
        const WeightLayer& layer_program = shared->weight_store.layers[index];
        const CommonWeights& common = common_weights(index);
        std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.residual.begin());
        cpu_rmsnorm(workspace_.hidden.data(), common.operator_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        if (const auto* attention = attention_operator(layer_program)) {
            const AttentionSpec& layout = shared->shape.attention_layout(static_cast<int>(index));
            const int q_width = layout.query_width();
            const int kv_width = layout.key_value_width();
            shared->linear.gemv(attention->q, workspace_.normed.data(), workspace_.qkv.data());
            float* q = workspace_.qkv.data();
            float* k = q + q_width;
            float* v = k + kv_width;
            if (!attention->k.segments.empty()) {
                shared->linear.gemv(attention->k, workspace_.normed.data(), k);
                shared->linear.gemv(attention->v, workspace_.normed.data(), v);
            }
            if (layout.query_key_norm) {
                cpu_qk_norm_rope(q, attention->q_norm.data(), layout.query_heads,
                    layout.head_dim, session_.position_value, static_cast<float>(layout.rope_theta),
                    shared->shape.norm_eps, static_cast<float>(layout.rotary_fraction));
                if (!attention->k.segments.empty()) {
                    cpu_qk_norm_rope(k, attention->k_norm.data(), layout.key_value_heads,
                        layout.head_dim, session_.position_value, static_cast<float>(layout.rope_theta),
                        shared->shape.norm_eps, static_cast<float>(layout.rotary_fraction));
                }
            } else {
                cpu_rope(q, layout.query_heads, layout.head_dim,
                         session_.position_value, static_cast<float>(layout.rope_theta),
                         static_cast<float>(layout.rotary_fraction));
                if (!attention->k.segments.empty()) {
                    cpu_rope(k, layout.key_value_heads, layout.head_dim,
                             session_.position_value, static_cast<float>(layout.rope_theta),
                             static_cast<float>(layout.rotary_fraction));
                }
                const float ratio = shared->shape.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(layout.head_dim)));
                for (int i = 0; i < q_width; ++i) q[i] *= ratio;
            }
            const int owner = shared->layer_to_kv_owner.at(index);
            AttentionState& state = attention_state(static_cast<size_t>(owner));
            if (!attention->k.segments.empty()) {
                store_kv(state, session_.position_value, k, v);
            }
            run_attention(state, layout, q, workspace_.op_output.data(),
                          session_.position_value + 1);
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
        if (shared->shape.has_split_attention_norms) {
            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.post_attention_norm.data(),
                                shared->shape.hidden, shared->shape.norm_eps);
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
            if (shared->shape.has_split_attention_norms) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.post_feed_forward_norm.data(),
                                    shared->shape.hidden, shared->shape.norm_eps);
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->shape.hidden);
            if (profile_moe) session_.prefill_profile.moe_expert_ms += milliseconds_since(started);
        } else {
            // Dense FFN path.
            const int intermediate = shared->shape.feed_forward_intermediates.empty()
                ? shared->shape.intermediate
                : shared->shape.feed_forward_intermediates.at(index);
            shared->linear.gemv(common.w13, workspace_.normed.data(), workspace_.gate_up.data());
            if (!shared->shape.feed_forward_activations.empty() &&
                shared->shape.feed_forward_activations.at(index) == ActivationKind::GeluTanh) {
                cpu_gated_gelu_tanh(workspace_.gate_up.data(), workspace_.activated.data(), intermediate);
            } else {
                cpu_swiglu(workspace_.gate_up.data(), workspace_.activated.data(), intermediate);
            }
            shared->linear.gemv(common.w2, workspace_.activated.data(), workspace_.mlp_output.data());
            if (shared->shape.residual_multiplier != 1.0f) {
                for (float& value : workspace_.mlp_output) value *= shared->shape.residual_multiplier;
            }
            if (shared->shape.has_split_attention_norms) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.post_feed_forward_norm.data(),
                                    shared->shape.hidden, shared->shape.norm_eps);
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->shape.hidden);
        }
        if (shared->shape.has_per_layer_input) {
            std::copy(workspace_.hidden.begin(), workspace_.hidden.end(), workspace_.residual.begin());
            shared->linear.gemv(common.per_layer_input_gate, workspace_.hidden.data(),
                                workspace_.per_layer_gate.data());
            cpu_gelu_tanh(workspace_.per_layer_gate.data(),
                          static_cast<size_t>(shared->shape.per_layer_input_size));
            const float* layer_input = workspace_.per_layer_context.data() +
                index * static_cast<size_t>(shared->shape.per_layer_input_size);
            for (int d = 0; d < shared->shape.per_layer_input_size; ++d) {
                workspace_.per_layer_gate[static_cast<size_t>(d)] *= layer_input[d];
            }
            shared->linear.gemv(common.per_layer_projection, workspace_.per_layer_gate.data(),
                                workspace_.hidden.data());
            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.per_layer_input_norm.data(),
                                shared->shape.hidden, shared->shape.norm_eps);
            if (common.layer_scalar != 1.0f) {
                for (float& value : workspace_.hidden) value *= common.layer_scalar;
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->shape.hidden);
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
        if (shared->final_logit_softcap > 0.0f) {
            for (float& value : workspace_.logits) {
                value = std::tanh(value / shared->final_logit_softcap) *
                    shared->final_logit_softcap;
            }
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
    for (size_t row = 0; row < tokens.size(); ++row) {
        forward_token(tokens[row], compute_logits && row + 1 == tokens.size());
    }
    return;
}

} // namespace celeg

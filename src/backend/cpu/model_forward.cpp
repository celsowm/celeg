#include "detail/model_internal.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <memory>
#include <stdexcept>

namespace celeg {

void configure_cpu_expert_backing(CpuCompiledModel::Shared& shared);

namespace {
using Clock = std::chrono::steady_clock;
float moe_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
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
        const bool nemotron = shared->shape.mamba2_layer_count > 0;
        if (nemotron && shared->shape.mixer_kinds[index] == MixerKind::MlpOnly) {
            const auto& mlp = std::get<CpuCompiledModel::MlpOnlyWeights>(layer_program);
            const int intermediate = shared->shape.mlp_only_layouts[index].intermediate_size;
            shared->linear.gemv(mlp.common.mlp_up, workspace_.normed.data(),
                                workspace_.activated.data());
            cpu_relu2(workspace_.activated.data(), workspace_.activated.data(), intermediate);
            shared->linear.gemv(mlp.common.w2, workspace_.activated.data(),
                                workspace_.hidden.data());
            cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->shape.hidden);
            continue;
        }
        if (const auto* gated_delta = gated_delta_net_operator(layer_program)) {
            const GatedDeltaNetSpec& spec = gated_delta->spec;
            shared->linear.gemv(gated_delta->qkv, workspace_.normed.data(),
                                workspace_.gated_delta_qkv.data());
            shared->linear.gemv(gated_delta->z, workspace_.normed.data(),
                                workspace_.gated_delta_z.data());
            shared->linear.gemv(gated_delta->b, workspace_.normed.data(),
                                workspace_.gated_delta_b.data());
            shared->linear.gemv(gated_delta->a, workspace_.normed.data(),
                                workspace_.gated_delta_a.data());
            auto& state = gated_delta_net_state(index);
            cpu_gated_delta_net_decode(workspace_.gated_delta_qkv.data(),
                workspace_.gated_delta_z.data(), workspace_.gated_delta_b.data(),
                workspace_.gated_delta_a.data(), gated_delta->conv_weight.data(),
                gated_delta->dt_bias.data(), gated_delta->a_log.data(),
                gated_delta->norm.data(), state.conv.data(), state.recurrent.data(),
                workspace_.gated_delta_output.data(), spec.conv_kernel,
                spec.key_head_dim, spec.value_head_dim, spec.key_heads,
                spec.value_heads, shared->shape.numerical_policy.norm_eps);
            shared->linear.gemv(gated_delta->out, workspace_.gated_delta_output.data(),
                                workspace_.hidden.data());
        } else
        if (const auto* mamba = mamba2_operator(layer_program)) {
            const auto& spec = shared->shape.mamba2_layouts[index];
            const int inner = spec.intermediate_size;
            const int conv_dim = inner + 2 * spec.group_count * spec.state_size;
            shared->linear.gemv(mamba->in, workspace_.normed.data(),
                                workspace_.mamba_projected.data());
            const float* z = workspace_.mamba_projected.data();
            const float* xbc = z + inner;
            const float* dt_raw = xbc + conv_dim;
            auto& state = mamba2_state(index);
            for (int channel = 0; channel < conv_dim; ++channel) {
                float* history = state.conv.data() + static_cast<size_t>(channel) * spec.conv_kernel;
                for (int tap = 0; tap + 1 < spec.conv_kernel; ++tap) history[tap] = history[tap + 1];
                history[spec.conv_kernel - 1] = xbc[channel];
                float value = mamba->conv_bias[channel];
                for (int tap = 0; tap < spec.conv_kernel; ++tap) {
                    value += history[tap] * mamba->conv_weight[
                        static_cast<size_t>(channel) * spec.conv_kernel + tap];
                }
                workspace_.mamba_bcx[channel] = value / (1.0f + std::exp(-value));
            }
            const int group_size = spec.num_heads / spec.group_count;
            for (int head = 0; head < spec.num_heads; ++head) {
                const float dt = std::log1p(std::exp(dt_raw[head] + mamba->dt_bias[head]));
                const float decay = std::exp(dt * -std::exp(mamba->a_log[head]));
                const int group = head / group_size;
                for (int d = 0; d < spec.head_dim; ++d) {
                    const int channel = head * spec.head_dim + d;
                    const float x = workspace_.mamba_bcx[channel];
                    const float* b = workspace_.mamba_bcx.data() + inner + group * spec.state_size;
                    const float* c = b + spec.group_count * spec.state_size;
                    float* s = state.ssm.data() + static_cast<size_t>(channel) * spec.state_size;
                    float output = 0.0f;
                    for (int n = 0; n < spec.state_size; ++n) {
                        s[n] = decay * s[n] + dt * b[n] * x;
                        output += s[n] * c[n];
                    }
                    workspace_.mamba_inner[channel] = output + mamba->d[head] * x;
                }
            }
            cpu_rmsnorm(workspace_.mamba_inner.data(), mamba->norm.data(),
                        workspace_.op_output.data(), inner,
                        shared->shape.numerical_policy.norm_eps);
            for (int i = 0; i < inner; ++i) {
                const float gate = z[i];
                workspace_.op_output[i] *= gate / (1.0f + std::exp(-gate));
            }
            shared->linear.gemv(mamba->out, workspace_.op_output.data(), workspace_.hidden.data());
            cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->shape.hidden);
            continue;
        }
        else if (const auto* attention = attention_operator(layer_program)) {
            const AttentionSpec& layout = shared->shape.attention_layout(static_cast<int>(index));
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
            if (layout.positional_encoding == PositionalEncodingKind::None) {
                const float ratio = shared->shape.numerical_policy.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(layout.head_dim)));
                for (int i = 0; i < q_width; ++i) q[i] *= ratio;
            } else if (layout.query_key_norm) {
                if (shared->shape.mrope_interleaved) {
                    cpu_qk_norm_rope_mrope(q, attention->q_norm.data(), layout.query_heads,
                        layout.head_dim, rope_position, shared->shape.mrope_section,
                        true, static_cast<float>(layout.rope_theta),
                        shared->shape.numerical_policy.norm_eps,
                        static_cast<float>(layout.rotary_fraction));
                } else {
                    cpu_qk_norm_rope(q, attention->q_norm.data(), layout.query_heads,
                        layout.head_dim, session_.position_value, static_cast<float>(layout.rope_theta),
                        shared->shape.numerical_policy.norm_eps, static_cast<float>(layout.rotary_fraction));
                }
                if (!attention->k.segments.empty()) {
                    if (shared->shape.mrope_interleaved) {
                        cpu_qk_norm_rope_mrope(k, attention->k_norm.data(), layout.key_value_heads,
                            layout.head_dim, rope_position, shared->shape.mrope_section,
                            true, static_cast<float>(layout.rope_theta),
                            shared->shape.numerical_policy.norm_eps,
                            static_cast<float>(layout.rotary_fraction));
                    } else {
                        cpu_qk_norm_rope(k, attention->k_norm.data(), layout.key_value_heads,
                            layout.head_dim, session_.position_value, static_cast<float>(layout.rope_theta),
                            shared->shape.numerical_policy.norm_eps, static_cast<float>(layout.rotary_fraction));
                    }
                }
                for (int i = 0; i < q_width; ++i) q[i] *= layout.query_scale;
            } else {
                if (shared->shape.mrope_interleaved) {
                    cpu_rope_mrope(q, layout.query_heads, layout.head_dim, rope_position,
                                   shared->shape.mrope_section, true,
                                   static_cast<float>(layout.rope_theta),
                                   static_cast<float>(layout.rotary_fraction));
                } else {
                    cpu_rope(q, layout.query_heads, layout.head_dim,
                             session_.position_value, static_cast<float>(layout.rope_theta),
                             static_cast<float>(layout.rotary_fraction));
                }
                if (!attention->k.segments.empty()) {
                    if (shared->shape.mrope_interleaved) {
                        cpu_rope_mrope(k, layout.key_value_heads, layout.head_dim, rope_position,
                                       shared->shape.mrope_section, true,
                                       static_cast<float>(layout.rope_theta),
                                       static_cast<float>(layout.rotary_fraction));
                    } else {
                        cpu_rope(k, layout.key_value_heads, layout.head_dim,
                                 session_.position_value, static_cast<float>(layout.rope_theta),
                                 static_cast<float>(layout.rotary_fraction));
                    }
                }
                const float ratio = shared->shape.numerical_policy.attention_multiplier /
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
            if (layout.query_gate) {
                const float* gate = q + q_width;
                for (int i = 0; i < q_width; ++i) {
                    workspace_.op_output[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
                }
            }
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
        if (shared->shape.numerical_policy.residual_multiplier != 1.0f) {
            for (float& value : workspace_.hidden) value *= shared->shape.numerical_policy.residual_multiplier;
        }
        if (shared->shape.has_split_attention_norms) {
            cpu_rmsnorm_inplace(workspace_.hidden.data(), common.post_attention_norm.data(),
                                shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
        }
        cpu_residual_add(workspace_.hidden.data(), workspace_.residual.data(), shared->shape.hidden);

        // Nemotron-H attention blocks are mixer-only: they do not have the
        // generic post-attention normalization and dense FFN.
        if (nemotron) continue;

        cpu_rmsnorm(workspace_.hidden.data(), common.ffn_norm.data(), workspace_.normed.data(),
                    shared->shape.hidden, shared->shape.numerical_policy.norm_eps);

        if (const auto* moe = std::get_if<MoeWeights>(&layer_program)) {
            const MoeLayerProgram& semantics = shared->program.layers[index].moe.value();
            if (shared->options.expert_backing == CpuExpertBacking::DiskCached &&
                !moe->disk_cached && !shared->native_checkpoint) {
                configure_cpu_expert_backing(*shared);
            }

            const int E = semantics.router.expert_count;
            const int K = semantics.router.experts_per_token;
            const int moe_inter = semantics.routed.mlp.intermediate_size;

            const bool profile_moe = session_.phase == SessionPhase::Prefilling;
            auto started = Clock::now();
            workspace_.moe_router_logits.resize(static_cast<size_t>(E));
            workspace_.moe_router_probs.resize(static_cast<size_t>(E));
            workspace_.moe_router_scored.resize(static_cast<size_t>(E));
            workspace_.moe_selected.resize(static_cast<size_t>(K));
            workspace_.moe_weights.resize(static_cast<size_t>(K));

            shared->linear.gemv_raw(moe->router.data(), workspace_.normed.data(), workspace_.moe_router_logits.data(),
                                    E, shared->shape.hidden);

            float router_max = -std::numeric_limits<float>::infinity();
            if (semantics.router.score == MoeRouterScoreKind::SoftmaxLogits) {
                for (int e = 0; e < E; ++e) {
                    router_max = std::max(router_max,
                        workspace_.moe_router_logits[static_cast<size_t>(e)]);
                }
            }
            float router_sum = 0.0f;
            for (int e = 0; e < E; ++e) {
                const float probability = semantics.router.score == MoeRouterScoreKind::SoftmaxLogits
                    ? std::exp(workspace_.moe_router_logits[static_cast<size_t>(e)] - router_max)
                    : moe_sigmoid(workspace_.moe_router_logits[static_cast<size_t>(e)]);
                workspace_.moe_router_probs[static_cast<size_t>(e)] = probability;
                if (semantics.router.score == MoeRouterScoreKind::SoftmaxLogits) router_sum += probability;
            }
            if (semantics.router.score == MoeRouterScoreKind::SoftmaxLogits) {
                for (float& probability : workspace_.moe_router_probs) probability /= router_sum;
            }
            for (int e = 0; e < E; ++e) {
                float score = workspace_.moe_router_probs[static_cast<size_t>(e)];
                if (semantics.router.has_expert_bias && e < static_cast<int>(moe->router_bias.size())) {
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

            float weight_sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                workspace_.moe_selected[static_cast<size_t>(k)] =
                    workspace_.moe_router_scored[static_cast<size_t>(k)].second;
                workspace_.moe_weights[static_cast<size_t>(k)] =
                    workspace_.moe_router_probs[static_cast<size_t>(workspace_.moe_selected[static_cast<size_t>(k)])];
                weight_sum += workspace_.moe_weights[static_cast<size_t>(k)];
            }
            if (semantics.router.normalization == MoeNormalizationKind::SumSelected) {
                const float inv = 1.0f / (weight_sum + 1e-6f);
                for (int k = 0; k < K; ++k) {
                    workspace_.moe_weights[static_cast<size_t>(k)] *= inv;
                }
            }
            for (int k = 0; k < K; ++k) {
                workspace_.moe_weights[static_cast<size_t>(k)] *= semantics.router.routed_scaling;
            }
            if (profile_moe) session_.prefill_profile.moe_router_ms += milliseconds_since(started);

            started = Clock::now();
            std::fill(workspace_.mlp_output.begin(), workspace_.mlp_output.end(), 0.0f);
            for (int k = 0; k < K; ++k) {
                const int expert = workspace_.moe_selected[static_cast<size_t>(k)];
                const float rw = workspace_.moe_weights[static_cast<size_t>(k)];
                if (expert < 0 || expert >= E) continue;

                std::shared_ptr<const CpuExpertWeights> cached;
                const CpuLinearWeight* w13 = nullptr;
                const CpuLinearWeight* w2 = nullptr;
                if (moe->disk_cached) {
                    cached = shared->acquire_expert(moe->layer_index, expert);
                    w13 = &cached->w13;
                    w2 = &cached->w2;
                } else {
                    w13 = &moe->expert_w13[static_cast<size_t>(expert)];
                    w2 = &moe->expert_w2[static_cast<size_t>(expert)];
                }

                shared->linear.gemv(*w13, workspace_.normed.data(), workspace_.gate_up.data());
                cpu_swiglu(workspace_.gate_up.data(), workspace_.activated.data(), moe_inter);
                shared->linear.gemv(*w2, workspace_.activated.data(), workspace_.op_output.data());
                for (int j = 0; j < shared->shape.hidden; ++j) {
                    workspace_.mlp_output[static_cast<size_t>(j)] += rw * workspace_.op_output[static_cast<size_t>(j)];
                }
            }
            if (semantics.shared) {
                const int shared_intermediate = semantics.shared->mlp.intermediate_size;
                shared->linear.gemv(moe->shared_w13, workspace_.normed.data(),
                                    workspace_.gate_up.data());
                cpu_swiglu(workspace_.gate_up.data(), workspace_.activated.data(),
                           shared_intermediate);
                shared->linear.gemv(moe->shared_w2, workspace_.activated.data(),
                                    workspace_.op_output.data());
                shared->linear.gemv(moe->shared_gate, workspace_.normed.data(),
                                    workspace_.moe_router_probs.data());
                const float gate = moe_sigmoid(workspace_.moe_router_probs.front());
                for (int j = 0; j < shared->shape.hidden; ++j) {
                    workspace_.mlp_output[static_cast<size_t>(j)] += gate *
                        workspace_.op_output[static_cast<size_t>(j)];
                }
            }
            if (shared->shape.numerical_policy.residual_multiplier != 1.0f) {
                for (float& value : workspace_.mlp_output) value *= shared->shape.numerical_policy.residual_multiplier;
            }
            if (shared->shape.has_split_attention_norms) {
                cpu_rmsnorm_inplace(workspace_.mlp_output.data(), common.post_feed_forward_norm.data(),
                                    shared->shape.hidden, shared->shape.numerical_policy.norm_eps);
            }
            cpu_residual_add(workspace_.hidden.data(), workspace_.mlp_output.data(), shared->shape.hidden);
            if (profile_moe) session_.prefill_profile.moe_expert_ms += milliseconds_since(started);
        } else {
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
    if (!embeddings || !embeddings->rope_at_position(static_cast<std::size_t>(session_.position_value - 1))) {
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
    if (shared->shape.mamba2_layer_count > 0) {
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
            const GatedDeltaNetSpec& spec = gated_delta->spec;
            linear_started = Clock::now();
            layer_gemm(gated_delta->qkv, workspace_.chunk_normed.data(),
                       workspace_.chunk_gated_delta_qkv.data());
            layer_gemm(gated_delta->z, workspace_.chunk_normed.data(),
                       workspace_.chunk_gated_delta_z.data());
            layer_gemm(gated_delta->b, workspace_.chunk_normed.data(),
                       workspace_.chunk_gated_delta_b.data());
            layer_gemm(gated_delta->a, workspace_.chunk_normed.data(),
                       workspace_.chunk_gated_delta_a.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            auto gated_delta_started = Clock::now();
            GatedDeltaNetState& state = gated_delta_net_state(index);
            cpu_gated_delta_net_prefill(workspace_.chunk_gated_delta_qkv.data(),
                workspace_.chunk_gated_delta_z.data(), workspace_.chunk_gated_delta_b.data(),
                workspace_.chunk_gated_delta_a.data(), gated_delta->conv_weight.data(),
                gated_delta->dt_bias.data(), gated_delta->a_log.data(),
                gated_delta->norm.data(), state.conv.data(), state.recurrent.data(),
                workspace_.chunk_gated_delta_output.data(), rows, spec.conv_kernel,
                spec.key_head_dim, spec.value_head_dim, spec.key_heads,
                spec.value_heads, shape.numerical_policy.norm_eps);
            session_.prefill_profile.shortconv_ms += milliseconds_since(gated_delta_started);
            linear_started = Clock::now();
            layer_gemm(gated_delta->out, workspace_.chunk_gated_delta_output.data(),
                       workspace_.chunk_hidden.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
        } else if (const auto* attention = attention_operator(layer_program)) {
            const AttentionSpec& layout = shape.attention_layout(static_cast<int>(index));
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
                if (layout.query_key_norm) {
                    for (size_t d = 0; d < q_width; ++d) q[d] *= layout.query_scale;
                }
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
                    for (size_t d = 0; d < q_width; ++d) {
                        output[d] *= 1.0f / (1.0f + std::exp(-gate[d]));
                    }
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
            linear_started = Clock::now();
            layer_gemm(convolution->in, workspace_.chunk_normed.data(),
                       workspace_.chunk_conv.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            auto conv_started = Clock::now();
            ConvolutionState& state = convolution_state(index);
            cpu_conv_prefill(workspace_.chunk_conv.data(), convolution->weight_tap_major.data(),
                             state.state.data(), workspace_.chunk_op.data(), rows, shape.hidden,
                             shape.conv_cache, base_position, shared->pool);
            session_.prefill_profile.shortconv_ms += milliseconds_since(conv_started);
            linear_started = Clock::now();
            layer_gemm(convolution->out, workspace_.chunk_op.data(),
                       workspace_.chunk_hidden.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
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
            const MoeLayerProgram& semantics = shared->program.layers[index].moe.value();
            const int experts = semantics.router.expert_count;
            const int selected = semantics.router.experts_per_token;
            const int intermediate = semantics.routed.mlp.intermediate_size;
            const size_t routes = rows * static_cast<size_t>(selected);
            std::fill(workspace_.chunk_mlp.begin(), workspace_.chunk_mlp.begin() + rows * hidden, 0.0f);
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
            auto router_started = Clock::now();
            shared->linear.gemm_raw(moe->router.data(), workspace_.chunk_normed.data(),
                                    workspace_.moe_router_logits.data(), rows, experts, shape.hidden);
            for (size_t row = 0; row < rows; ++row) {
                float* probabilities = workspace_.moe_router_probs.data() +
                    row * static_cast<size_t>(experts);
                std::pair<float, int>* scored = workspace_.moe_router_scored.data() +
                    row * static_cast<size_t>(experts);
                const float* logits = workspace_.moe_router_logits.data() +
                    row * static_cast<size_t>(experts);
                const float row_max = semantics.router.score == MoeRouterScoreKind::SoftmaxLogits
                    ? *std::max_element(logits, logits + experts) : 0.0f;
                float row_sum = 0.0f;
                for (int expert = 0; expert < experts; ++expert) {
                    const float probability = semantics.router.score == MoeRouterScoreKind::SoftmaxLogits
                        ? std::exp(logits[expert] - row_max)
                        : moe_sigmoid(logits[expert]);
                    probabilities[expert] = probability;
                    row_sum += probability;
                    scored[expert] = {
                        probability + (semantics.router.has_expert_bias &&
                            expert < static_cast<int>(moe->router_bias.size())
                            ? moe->router_bias[expert] : 0.0f), expert};
                }
                if (semantics.router.score == MoeRouterScoreKind::SoftmaxLogits) {
                    for (int expert = 0; expert < experts; ++expert) {
                        probabilities[expert] /= row_sum;
                        scored[expert].first = probabilities[expert] +
                            (semantics.router.has_expert_bias &&
                             expert < static_cast<int>(moe->router_bias.size())
                                ? moe->router_bias[expert] : 0.0f);
                    }
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
                workspace_.moe_route_order[workspace_.moe_group_cursor[static_cast<size_t>(expert)]++] = route;
            }
            session_.prefill_profile.moe_router_ms += milliseconds_since(router_started);

            auto expert_started = Clock::now();
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
                        shared->acquire_expert(moe->layer_index, expert);
                    w13 = &workspace_.moe_cached_experts[static_cast<size_t>(expert)]->w13;
                } else {
                    w13 = &moe->expert_w13[static_cast<size_t>(expert)];
                }
                workspace_.moe_gemm_jobs.push_back({w13, begin, end - begin});
            }
            for (size_t packed_route = 0; packed_route < routes; ++packed_route) {
                const size_t route = workspace_.moe_route_order[packed_route];
                const size_t row = static_cast<size_t>(workspace_.moe_route_rows[route]);
                std::copy_n(workspace_.chunk_normed.data() + row * hidden, hidden,
                            workspace_.moe_gathered_normed.data() + packed_route * hidden);
            }
            shared->linear.gemm_grouped(workspace_.moe_gemm_jobs,
                                        workspace_.moe_gathered_normed.data(),
                                        workspace_.moe_gathered_gate_up.data());
            parallel_rows(shared->pool, routes, [&](size_t route) {
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
            shared->linear.gemm_grouped(workspace_.moe_gemm_jobs,
                                        workspace_.moe_gathered_activated.data(),
                                        workspace_.moe_gathered_output.data());
            for (size_t packed_route = 0; packed_route < routes; ++packed_route) {
                const size_t route = workspace_.moe_route_order[packed_route];
                const size_t row = static_cast<size_t>(workspace_.moe_route_rows[route]);
                float* destination = workspace_.chunk_mlp.data() + row * hidden;
                const float* source = workspace_.moe_gathered_output.data() + packed_route * hidden;
                const float weight = workspace_.moe_route_weights[route];
                for (size_t d = 0; d < hidden; ++d) destination[d] += weight * source[d];
            }
            session_.prefill_profile.moe_expert_ms += milliseconds_since(expert_started);
        } else {
            const int intermediate = shape.feed_forward_intermediates.empty()
                ? shape.intermediate : shape.feed_forward_intermediates.at(index);
            linear_started = Clock::now();
            layer_gemm(common.w13, workspace_.chunk_normed.data(),
                       workspace_.chunk_gate_up.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
            parallel_rows(shared->pool, rows, [&](size_t row) {
                const float* gate_up = workspace_.chunk_gate_up.data() + row * 2ULL * intermediate;
                float* activated = workspace_.chunk_activated.data() + row * intermediate;
                if (!shape.feed_forward_activations.empty() &&
                    shape.feed_forward_activations.at(index) == ActivationKind::GeluTanh) {
                    cpu_gated_gelu_tanh(gate_up, activated, intermediate);
                } else {
                    cpu_swiglu(gate_up, activated, intermediate);
                }
            });
            linear_started = Clock::now();
            layer_gemm(common.w2, workspace_.chunk_activated.data(),
                       workspace_.chunk_mlp.data());
            session_.prefill_profile.linear_ms += milliseconds_since(linear_started);
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
}

} // namespace celeg

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
        if (std::any_of(sessions.front()->shared->weight_store.layers.begin(),
                        sessions.front()->shared->weight_store.layers.end(),
                        [](const LayerWeights& layer) {
                            return std::holds_alternative<MoeWeights>(layer);
                        })) {
            for (size_t row = 0; row < sessions.size(); ++row) {
                sessions[row]->forward_token(tokens[row], compute_logits[row] != 0);
            }
            return;
        }
        const size_t batch = sessions.size();
        SharedWeights& shared = *sessions.front()->shared;
        const RuntimeTopology& shape = shared.shape;
        workspace_.ensure(batch, shape);

        for (size_t row = 0; row < batch; ++row) {
            State& session = *sessions[row];
            if (session.session_.position_value >= shared.max_context) {
                throw std::runtime_error("CPU context limit reached in packed batch");
            }
            if (tokens[row] < 0 || tokens[row] >= shape.vocab_size) {
                throw std::invalid_argument("CPU token out of range in packed batch");
            }
            shared.linear.embedding(shared.weight_store.embedding, tokens[row],
                                    workspace_.hidden.data() + row * shape.hidden);
            if (shape.embedding_multiplier != 1.0f) {
                float* values = workspace_.hidden.data() + row * shape.hidden;
                for (int i = 0; i < shape.hidden; ++i) values[i] *= shape.embedding_multiplier;
            }
        }

        for (size_t layer_index = 0; layer_index < shared.weight_store.layers.size(); ++layer_index) {
            const LayerWeights& layer = shared.weight_store.layers[layer_index];
            const CommonWeights* common_ptr = std::visit(
                [](const auto& value) -> const CommonWeights* {
                    return &value.common;
                }, layer);
            const CommonWeights& common = *common_ptr;

            std::copy(workspace_.hidden.begin(), workspace_.hidden.begin() + static_cast<ptrdiff_t>(
                batch * shape.hidden), workspace_.residual.begin());
            for (size_t row = 0; row < batch; ++row) {
                cpu_rmsnorm(workspace_.hidden.data() + row * shape.hidden,
                            common.operator_norm.data(),
                            workspace_.normed.data() + row * shape.hidden,
                            shape.hidden, shape.norm_eps);
            }

            if (const auto* attention =
                    std::get_if<AttentionWeights>(&layer)) {
                shared.linear.gemm(attention->qkv, workspace_.normed.data(), workspace_.qkv.data(), batch);
                for (size_t row = 0; row < batch; ++row) {
                    State& session = *sessions[row];
                    float* q = workspace_.qkv.data() + row * shape.qkv_width;
                    float* k = q + shape.q_width;
                    float* v = k + shape.kv_width;
                    if (shape.query_key_norm) {
                        cpu_qk_norm_rope(q, attention->q_norm.data(),
                            shape.num_attention_heads, shape.head_dim,
                            session.session_.position_value, shape.rope_theta,
                            shape.norm_eps);
                        cpu_qk_norm_rope(k, attention->k_norm.data(),
                            shape.num_key_value_heads, shape.head_dim,
                            session.session_.position_value, shape.rope_theta,
                            shape.norm_eps);
                    } else {
                        cpu_rope(q, shape.num_attention_heads, shape.head_dim,
                                 session.session_.position_value, shape.rope_theta);
                        cpu_rope(k, shape.num_key_value_heads, shape.head_dim,
                                 session.session_.position_value, shape.rope_theta);
                        const float ratio = shape.attention_multiplier /
                            (1.0f / std::sqrt(static_cast<float>(shape.head_dim)));
                        for (int i = 0; i < shape.q_width; ++i) q[i] *= ratio;
                    }
                    auto& cache = session.attention_state(layer_index);
                    session.store_kv(cache, session.session_.position_value, k, v);
                    session.run_attention(
                        cache, q, workspace_.op_output.data() + row * shape.hidden,
                        session.session_.position_value + 1);
                }
                shared.linear.gemm(attention->out, workspace_.op_output.data(), workspace_.hidden.data(), batch);
            } else {
                const auto& convolution =
                    std::get<ConvolutionWeights>(layer);
                shared.linear.gemm(convolution.in, workspace_.normed.data(),
                                   workspace_.conv_projected.data(), batch);
                for (size_t row = 0; row < batch; ++row) {
                    State& session = *sessions[row];
                    auto& conv_state = session.convolution_state(layer_index);
                    cpu_conv_decode(
                        workspace_.conv_projected.data() + row * 3ULL * shape.hidden,
                        convolution.weight_tap_major.data(), conv_state.state.data(),
                        workspace_.op_output.data() + row * shape.hidden,
                        shape.hidden, shape.conv_cache,
                        session.session_.position_value);
                }
                shared.linear.gemm(convolution.out, workspace_.op_output.data(),
                                   workspace_.hidden.data(), batch);
            }

            for (size_t row = 0; row < batch; ++row) {
                if (shape.residual_multiplier != 1.0f) {
                    float* value = workspace_.hidden.data() + row * shape.hidden;
                    for (int i = 0; i < shape.hidden; ++i) value[i] *= shape.residual_multiplier;
                }
                cpu_residual_add(workspace_.hidden.data() + row * shape.hidden,
                                 workspace_.residual.data() + row * shape.hidden,
                                 shape.hidden);
                cpu_rmsnorm(workspace_.hidden.data() + row * shape.hidden,
                            common.ffn_norm.data(),
                            workspace_.normed.data() + row * shape.hidden,
                            shape.hidden, shape.norm_eps);
            }
            shared.linear.gemm(common.w13, workspace_.normed.data(), workspace_.gate_up.data(), batch);
            for (size_t row = 0; row < batch; ++row) {
                cpu_swiglu(workspace_.gate_up.data() + row * 2ULL * shape.intermediate,
                            workspace_.activated.data() + row * shape.intermediate,
                            shape.intermediate);
            }
            shared.linear.gemm(common.w2, workspace_.activated.data(), workspace_.mlp_output.data(), batch);
            for (size_t row = 0; row < batch; ++row) {
                if (shape.residual_multiplier != 1.0f) {
                    float* value = workspace_.mlp_output.data() + row * shape.hidden;
                    for (int i = 0; i < shape.hidden; ++i) value[i] *= shape.residual_multiplier;
                }
                cpu_residual_add(workspace_.hidden.data() + row * shape.hidden,
                                 workspace_.mlp_output.data() + row * shape.hidden,
                                 shape.hidden);
            }
        }

        workspace_.terminal_rows.clear();
        for (size_t row = 0; row < batch; ++row) {
            if (compute_logits[row]) workspace_.terminal_rows.push_back(row);
        }
        if (!workspace_.terminal_rows.empty()) {
            workspace_.final_normed.resize(workspace_.terminal_rows.size() * shape.hidden);
            workspace_.final_logits.resize(workspace_.terminal_rows.size() * shape.vocab_size);
            for (size_t index = 0; index < workspace_.terminal_rows.size(); ++index) {
                const size_t row = workspace_.terminal_rows[index];
                cpu_rmsnorm(workspace_.hidden.data() + row * shape.hidden,
                            shared.weight_store.final_norm.data(),
                            workspace_.final_normed.data() + index * shape.hidden,
                            shape.hidden, shape.norm_eps);
            }
            shared.linear.gemm(shared.weight_store.embedding, workspace_.final_normed.data(),
                               workspace_.final_logits.data(), workspace_.terminal_rows.size());
            if (shape.logits_divisor != 1.0f) {
                for (float& value : workspace_.final_logits) value /= shape.logits_divisor;
            }
            for (size_t index = 0; index < workspace_.terminal_rows.size(); ++index) {
                State& session = *sessions[workspace_.terminal_rows[index]];
                std::copy(
                    workspace_.final_logits.begin() + static_cast<ptrdiff_t>(
                        index * shape.vocab_size),
                    workspace_.final_logits.begin() + static_cast<ptrdiff_t>(
                        (index + 1) * shape.vocab_size),
                    session.workspace_.logits.begin());
            }
        }

        for (size_t row = 0; row < batch; ++row) {
            State& session = *sessions[row];
            ++session.session_.position_value;
            if (compute_logits[row]) session.session_.phase = SessionPhase::Ready;
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

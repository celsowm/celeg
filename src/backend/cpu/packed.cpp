#include "lfm/backend/cpu/packed.hpp"

#include "detail/model_internal.hpp"
#include "lfm/backend/cpu/kernels.hpp"
#include "lfm/backend/cpu/model.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lfm {

struct CpuPackedExecutor::Impl {
    void ensure(size_t batch, const ModelShape& shape) {
        const size_t hidden_count = batch * shape.hidden;
        const size_t qkv_count = batch * shape.qkv_width;
        const size_t conv_count = batch * 3ULL * shape.hidden;
        const size_t gate_count = batch * 2ULL * shape.intermediate;
        const size_t intermediate_count = batch * shape.intermediate;
        hidden.resize(hidden_count);
        residual.resize(hidden_count);
        normed.resize(hidden_count);
        op_output.resize(hidden_count);
        qkv.resize(qkv_count);
        conv_projected.resize(conv_count);
        gate_up.resize(gate_count);
        activated.resize(intermediate_count);
        mlp_output.resize(hidden_count);
    }

    static CpuModel::Impl& state(CpuModel* model) {
        if (!model || !model->impl_) {
            throw std::invalid_argument("packed CPU session is null");
        }
        return *model->impl_;
    }

    static void validate_shared(std::span<CpuModel* const> sessions) {
        if (sessions.empty()) throw std::invalid_argument("packed CPU batch is empty");
        const auto shared = state(sessions.front()).shared;
        for (CpuModel* session : sessions) {
            CpuModel::Impl& impl = state(session);
            if (impl.shared.get() != shared.get()) {
                throw std::invalid_argument(
                    "packed CPU sessions must share the same model weights");
            }
        }
    }

    void forward(std::span<CpuModel* const> sessions,
                 std::span<const int32_t> tokens,
                 std::span<const uint8_t> compute_logits) {
        if (sessions.size() != tokens.size() ||
            sessions.size() != compute_logits.size()) {
            throw std::invalid_argument("packed CPU metadata size mismatch");
        }
        validate_shared(sessions);
        const size_t batch = sessions.size();
        CpuModel::Impl::Shared& shared = *state(sessions.front()).shared;
        const ModelShape& shape = shared.shape;
        ensure(batch, shape);

        for (size_t row = 0; row < batch; ++row) {
            CpuModel::Impl& session = state(sessions[row]);
            if (session.position_value >= shared.max_context) {
                throw std::runtime_error("CPU context limit reached in packed batch");
            }
            if (tokens[row] < 0 || tokens[row] >= shape.vocab_size) {
                throw std::invalid_argument("CPU token out of range in packed batch");
            }
            shared.linear.embedding(shared.embedding, tokens[row],
                                    hidden.data() + row * shape.hidden);
        }

        for (size_t layer_index = 0; layer_index < shared.layers.size(); ++layer_index) {
            const CpuModel::Impl::WeightLayer& layer = shared.layers[layer_index];
            const CpuModel::Impl::CommonWeights* common_ptr = std::visit(
                [](const auto& value) -> const CpuModel::Impl::CommonWeights* {
                    return &value.common;
                }, layer);
            const CpuModel::Impl::CommonWeights& common = *common_ptr;

            std::copy(hidden.begin(), hidden.begin() + static_cast<ptrdiff_t>(
                batch * shape.hidden), residual.begin());
            for (size_t row = 0; row < batch; ++row) {
                cpu_rmsnorm(hidden.data() + row * shape.hidden,
                            common.operator_norm.data(),
                            normed.data() + row * shape.hidden,
                            shape.hidden, shape.norm_eps);
            }

            if (const auto* attention =
                    std::get_if<CpuModel::Impl::AttentionWeights>(&layer)) {
                shared.linear.gemm(attention->qkv, normed.data(), qkv.data(), batch);
                for (size_t row = 0; row < batch; ++row) {
                    CpuModel::Impl& session = state(sessions[row]);
                    float* q = qkv.data() + row * shape.qkv_width;
                    float* k = q + shape.q_width;
                    float* v = k + shape.kv_width;
                    cpu_qk_norm_rope(q, attention->q_norm.data(),
                        shape.num_attention_heads, shape.head_dim,
                        session.position_value, shape.rope_theta,
                        shape.norm_eps);
                    cpu_qk_norm_rope(k, attention->k_norm.data(),
                        shape.num_key_value_heads, shape.head_dim,
                        session.position_value, shape.rope_theta,
                        shape.norm_eps);
                    auto& cache = session.attention_state(layer_index);
                    session.store_kv(cache, session.position_value, k, v);
                    session.run_attention(
                        cache, q, op_output.data() + row * shape.hidden,
                        session.position_value + 1);
                }
                shared.linear.gemm(attention->out, op_output.data(), hidden.data(), batch);
            } else {
                const auto& convolution =
                    std::get<CpuModel::Impl::ConvolutionWeights>(layer);
                shared.linear.gemm(convolution.in, normed.data(),
                                   conv_projected.data(), batch);
                for (size_t row = 0; row < batch; ++row) {
                    CpuModel::Impl& session = state(sessions[row]);
                    auto& conv_state = session.convolution_state(layer_index);
                    cpu_conv_decode(
                        conv_projected.data() + row * 3ULL * shape.hidden,
                        convolution.weight.data(), conv_state.state.data(),
                        op_output.data() + row * shape.hidden,
                        shape.hidden, shape.conv_cache,
                        session.position_value);
                }
                shared.linear.gemm(convolution.out, op_output.data(),
                                   hidden.data(), batch);
            }

            for (size_t row = 0; row < batch; ++row) {
                cpu_residual_add(hidden.data() + row * shape.hidden,
                                 residual.data() + row * shape.hidden,
                                 shape.hidden);
                cpu_rmsnorm(hidden.data() + row * shape.hidden,
                            common.ffn_norm.data(),
                            normed.data() + row * shape.hidden,
                            shape.hidden, shape.norm_eps);
            }
            shared.linear.gemm(common.w13, normed.data(), gate_up.data(), batch);
            for (size_t row = 0; row < batch; ++row) {
                cpu_swiglu(gate_up.data() + row * 2ULL * shape.intermediate,
                            activated.data() + row * shape.intermediate,
                            shape.intermediate);
            }
            shared.linear.gemm(common.w2, activated.data(), mlp_output.data(), batch);
            for (size_t row = 0; row < batch; ++row) {
                cpu_residual_add(hidden.data() + row * shape.hidden,
                                 mlp_output.data() + row * shape.hidden,
                                 shape.hidden);
            }
        }

        terminal_rows.clear();
        for (size_t row = 0; row < batch; ++row) {
            if (compute_logits[row]) terminal_rows.push_back(row);
        }
        if (!terminal_rows.empty()) {
            final_normed.resize(terminal_rows.size() * shape.hidden);
            final_logits.resize(terminal_rows.size() * shape.vocab_size);
            for (size_t index = 0; index < terminal_rows.size(); ++index) {
                const size_t row = terminal_rows[index];
                cpu_rmsnorm(hidden.data() + row * shape.hidden,
                            shared.final_norm.data(),
                            final_normed.data() + index * shape.hidden,
                            shape.hidden, shape.norm_eps);
            }
            shared.linear.gemm(shared.embedding, final_normed.data(),
                               final_logits.data(), terminal_rows.size());
            for (size_t index = 0; index < terminal_rows.size(); ++index) {
                CpuModel::Impl& session = state(sessions[terminal_rows[index]]);
                std::copy(
                    final_logits.begin() + static_cast<ptrdiff_t>(
                        index * shape.vocab_size),
                    final_logits.begin() + static_cast<ptrdiff_t>(
                        (index + 1) * shape.vocab_size),
                    session.logits.begin());
            }
        }

        for (size_t row = 0; row < batch; ++row) {
            CpuModel::Impl& session = state(sessions[row]);
            ++session.position_value;
            if (compute_logits[row]) session.phase = SessionPhase::Ready;
        }
    }

    std::vector<float> hidden;
    std::vector<float> residual;
    std::vector<float> normed;
    std::vector<float> op_output;
    std::vector<float> qkv;
    std::vector<float> conv_projected;
    std::vector<float> gate_up;
    std::vector<float> activated;
    std::vector<float> mlp_output;
    std::vector<size_t> terminal_rows;
    std::vector<float> final_normed;
    std::vector<float> final_logits;
};

CpuPackedExecutor::CpuPackedExecutor() : impl_(std::make_unique<Impl>()) {}
CpuPackedExecutor::~CpuPackedExecutor() = default;
CpuPackedExecutor::CpuPackedExecutor(CpuPackedExecutor&&) noexcept = default;
CpuPackedExecutor& CpuPackedExecutor::operator=(CpuPackedExecutor&&) noexcept = default;

CpuPackedStepMetrics CpuPackedExecutor::prefill_step(
    std::span<const CpuPrefillItem> items) {
    if (items.empty()) throw std::invalid_argument("CPU ragged prefill batch is empty");
    std::vector<CpuModel*> sessions;
    std::vector<int32_t> tokens;
    std::vector<uint8_t> terminal;
    sessions.reserve(items.size());
    tokens.reserve(items.size());
    terminal.reserve(items.size());
    for (const CpuPrefillItem& item : items) {
        CpuModel::Impl& session = Impl::state(item.session);
        if (session.phase != SessionPhase::Empty &&
            session.phase != SessionPhase::Prefilling) {
            throw std::runtime_error("CPU session is not eligible for prefill");
        }
        if (item.token < 0 ||
            item.token >= session.shared->shape.vocab_size) {
            throw std::invalid_argument("CPU prefill token out of range");
        }
        session.phase = SessionPhase::Prefilling;
        session.seen[static_cast<size_t>(item.token)] = 1;
        sessions.push_back(item.session);
        tokens.push_back(item.token);
        terminal.push_back(item.final_token ? 1 : 0);
    }
    const auto started = std::chrono::steady_clock::now();
    impl_->forward(sessions, tokens, terminal);
    const auto ended = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(ended - started).count();
    for (const CpuPrefillItem& item : items) {
        CpuModel::Impl& session = Impl::state(item.session);
        ++session.metrics.prefill_tokens;
        session.metrics.last_prefill_ms += elapsed / items.size();
    }
    return {items.size(), elapsed};
}

CpuPackedStepMetrics CpuPackedExecutor::prefill_chunk(
    CpuModel* model, std::span<const int32_t> tokens, bool final_chunk) {
    if (!model) throw std::invalid_argument("CPU chunked prefill session is null");
    if (tokens.empty()) throw std::invalid_argument("CPU chunked prefill is empty");
    CpuModel::Impl& session = Impl::state(model);
    if (session.phase != SessionPhase::Empty &&
        session.phase != SessionPhase::Prefilling) {
        throw std::runtime_error("CPU session is not eligible for chunked prefill");
    }
    for (int32_t token : tokens) {
        if (token < 0 ||
            token >= session.shared->shape.vocab_size) {
            throw std::invalid_argument("CPU chunked prefill token out of range");
        }
        session.seen[static_cast<size_t>(token)] = 1;
    }
    session.phase = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    session.forward_chunk(tokens, final_chunk);
    const auto ended = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(ended - started).count();
    session.metrics.prefill_tokens += tokens.size();
    session.metrics.last_prefill_ms += elapsed;
    if (final_chunk) session.phase = SessionPhase::Ready;
    return {tokens.size(), elapsed};
}

std::pair<std::vector<int32_t>, CpuPackedStepMetrics>
CpuPackedExecutor::decode_step(std::span<CpuModel* const> sessions) {
    Impl::validate_shared(sessions);
    std::vector<int32_t> tokens;
    std::vector<uint8_t> compute_logits(sessions.size(), 1);
    tokens.reserve(sessions.size());
    for (CpuModel* model : sessions) {
        CpuModel::Impl& session = Impl::state(model);
        if (session.phase != SessionPhase::Ready) {
            throw std::runtime_error("CPU session is not ready for packed decode");
        }
        const int32_t token = session.sample();
        session.seen[static_cast<size_t>(token)] = 1;
        tokens.push_back(token);
    }
    const auto started = std::chrono::steady_clock::now();
    impl_->forward(sessions, tokens, compute_logits);
    const auto ended = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(ended - started).count();
    for (CpuModel* model : sessions) {
        CpuModel::Impl& session = Impl::state(model);
        session.metrics.cumulative_decode_ms += elapsed / sessions.size();
        ++session.metrics.decoded_tokens;
    }
    return {std::move(tokens), {sessions.size(), elapsed}};
}

} // namespace lfm

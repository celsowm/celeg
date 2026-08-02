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
        for (size_t row = 0; row < sessions.size(); ++row) {
            sessions[row]->forward_token(tokens[row], compute_logits[row] != 0);
        }
        return;
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

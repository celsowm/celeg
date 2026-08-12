#include "detail/model_internal.hpp"

#include "celeg/backend/cpu/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace celeg {

namespace {

bool requires_sequential_execution(const CompiledModelProgram& program) {
    return std::any_of(program.layers.begin(), program.layers.end(),
        [](const CompiledLayerProgram& layer) {
            return layer.chunk_capability != CompiledChunkCapability::Native ||
                (layer.attention.has_value() &&
                 (!std::holds_alternative<NoAttentionOutputTransformSpec>(
                      layer.attention->output_transform) ||
                  (layer.attention->latent_state() &&
                   layer.attention->latent_state()->factorized)));
        });
}

} // namespace

void CpuCompiledModel::forward_batch(std::span<CpuCompiledModel* const> sessions,
                                     std::span<const int32_t> tokens,
                                     std::span<const uint8_t> compute_logits) {
    const bool has_sequential_only_layer = !sessions.empty() &&
        requires_sequential_execution(sessions.front()->shared->program);
    if (has_sequential_only_layer) {
        if (tokens.size() != sessions.size()) {
            throw std::invalid_argument(
                "sequential-only model batch requires one token per session");
        }
        for (size_t row = 0; row < sessions.size(); ++row) {
            sessions[row]->forward_token(tokens[row],
                                         compute_logits.empty() ? false : compute_logits[row]);
        }
        return;
    }
    execute_cpu_packed_batch(sessions, tokens, compute_logits);
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
        if (item.token < 0 || item.token >= session.shared->shape.checkpoint.vocab_size) {
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
        if (item.final_token) session.session_.phase = SessionPhase::Ready;
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
    if (token < 0 || token >= session.shared->shape.checkpoint.vocab_size) {
            throw std::invalid_argument("CPU chunked prefill token out of range");
        }
        session.session_.seen[static_cast<size_t>(token)] = 1;
    }
    session.session_.phase = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    if (requires_sequential_execution(session.shared->program)) {
        for (size_t index = 0; index < tokens.size(); ++index) {
            session.forward_token(tokens[index], final_chunk && index + 1 == tokens.size());
        }
    } else {
        session.forward_chunk(tokens, final_chunk);
    }
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
    validate_cpu_packed_batch(sessions);
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

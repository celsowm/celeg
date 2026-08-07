#include "celeg/backend/cpu/model.hpp"

#include "detail/model_internal.hpp"
#include "celeg/backend/cpu/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace celeg {

const char* cpu_kv_cache_mode_name(CpuKvCacheMode mode) {
    switch (mode) {
        case CpuKvCacheMode::Fp32: return "fp32";
        case CpuKvCacheMode::Bf16: return "bf16";
    }
    return "unknown";
}

CpuKvCacheMode parse_cpu_kv_cache_mode(const std::string& text) {
    if (text == "fp32") return CpuKvCacheMode::Fp32;
    if (text == "bf16") return CpuKvCacheMode::Bf16;
    throw std::invalid_argument("CPU KV cache mode must be fp32 or bf16");
}

CpuModel::CpuModel(const std::string& path, int context,
                   CpuModelOptions options, GenerationConfig generation,
                   std::shared_ptr<const RuntimeContext> runtime)
    : state_(std::make_unique<CpuCompiledModel>(
          std::make_shared<CpuCompiledModel::Shared>(
              path, context, std::move(options),
              runtime ? std::move(runtime) : create_builtin_runtime_context()),
          generation, -1)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

CpuModel::CpuModel(std::unique_ptr<CpuCompiledModel> impl)
    : state_(std::move(impl)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {
    if (!state_) throw std::invalid_argument("CPU model implementation is required");
}

CpuModel::~CpuModel() = default;

CpuModel::CpuModel(CpuModel&& other) noexcept
    : state_(std::move(other.state_)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

CpuModel& CpuModel::operator=(CpuModel&& other) noexcept {
    if (this != &other) {
        state_ = std::move(other.state_);
    }
    return *this;
}

std::unique_ptr<CpuModel> CpuModel::clone_session() const {
    return clone_session_on_node(-1);
}

std::unique_ptr<CpuModel> CpuModel::clone_session_on_node(int numa_node) const {
    return std::unique_ptr<CpuModel>(new CpuModel(
        std::make_unique<CpuCompiledModel>(state_->shared, state_->session_.generation, numa_node)));
}

std::vector<std::shared_ptr<CpuKvPagePool>> CpuModel::shared_kv_pools() const {
    return state_->shared->kv_pools;
}

void CpuModel::reset_session() { state_->reset(); }

void CpuModel::prefill_session(const std::vector<int32_t>& tokens) {
    prefill_session(tokens, {});
}

void CpuModel::prefill_session(const std::vector<int32_t>& tokens,
                               const PromptEmbedding& embeddings) {
    if (tokens.empty()) throw std::invalid_argument("CPU prefill needs at least one token");
    if (tokens.size() > static_cast<size_t>(state_->shared->max_context)) {
        throw std::invalid_argument("CPU prefill exceeds context");
    }
    for (int32_t token : tokens) {
        if (token < 0 || token >= state_->shared->shape.vocab_size) {
            throw std::invalid_argument("CPU token out of range");
        }
    }
    if (!embeddings.empty() &&
        (embeddings.width <= 0 ||
         embeddings.values.size() != embeddings.positions.size() *
             static_cast<size_t>(embeddings.width))) {
        throw std::invalid_argument("invalid CPU prompt embedding layout");
    }
    if (embeddings.has_rope_positions && embeddings.rope_positions.size() != tokens.size()) {
        throw std::invalid_argument("CPU prompt M-RoPE positions must cover every token");
    }
    state_->reset();
    state_->session_.phase = SessionPhase::Prefilling;
    state_->session_.prefill_profile = {};
    const auto started = std::chrono::steady_clock::now();
    const bool has_per_token_rope = embeddings.has_rope_positions;
    if (has_per_token_rope || tokens.size() < state_->shared->options.prefill_chunk_threshold) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            state_->session_.seen[static_cast<size_t>(tokens[i])] = 1;
            state_->forward_token(tokens[i], i + 1 == tokens.size(),
                                  embeddings.empty() ? nullptr : &embeddings);
        }
    } else {
        const size_t chunk = state_->shared->options.prefill_chunk_tokens;
        for (size_t begin = 0; begin < tokens.size(); begin += chunk) {
            const size_t end = std::min(tokens.size(), begin + chunk);
            for (size_t i = begin; i < end; ++i) {
                state_->session_.seen[static_cast<size_t>(tokens[i])] = 1;
            }
            state_->forward_chunk(
                std::span<const int32_t>(tokens.data() + begin, end - begin),
                end == tokens.size(), embeddings.empty() ? nullptr : &embeddings);
        }
    }
    if (embeddings.has_rope_positions) {
        state_->session_.next_rope_position = embeddings.next_rope_position;
    }
    const auto ended = std::chrono::steady_clock::now();
    state_->session_.metrics.last_prefill_ms =
        std::chrono::duration<double, std::milli>(ended - started).count();
    state_->session_.prefill_profile.total_ms =
        state_->session_.metrics.last_prefill_ms;
    state_->session_.metrics.prefill_tokens = tokens.size();
    state_->session_.phase = SessionPhase::Ready;
}

int32_t CpuModel::decode_session() {
    if (state_->session_.phase != SessionPhase::Ready) {
        throw std::runtime_error("CPU model is not ready for decode");
    }
    const auto started = std::chrono::steady_clock::now();
    const int32_t token = CpuSampler::sample(
        state_->workspace_.logits, state_->shared->shape,
        state_->session_.generation, state_->session_.seen,
        state_->session_.rng_state);
    state_->session_.seen[static_cast<size_t>(token)] = 1;
    state_->forward_token(token, true);
    const auto ended = std::chrono::steady_clock::now();
    state_->session_.metrics.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    ++state_->session_.metrics.decoded_tokens;
    return token;
}

void CpuModel::eval_token_session(int32_t token) {
    if (state_->session_.phase != SessionPhase::Ready) {
        throw std::runtime_error("CPU model is not ready for token evaluation");
    }
    if (token < 0 || token >= state_->shared->shape.vocab_size) {
        throw std::invalid_argument("CPU token out of range");
    }
    const auto started = std::chrono::steady_clock::now();
    state_->session_.seen[static_cast<size_t>(token)] = 1;
    state_->forward_token(token, true);
    const auto ended = std::chrono::steady_clock::now();
    state_->session_.metrics.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    ++state_->session_.metrics.decoded_tokens;
}

void CpuModel::set_session_generation(GenerationConfig generation) {
    state_->set_generation(generation);
}

int CpuModel::session_position() const { return state_->session_.position_value; }

bool CpuModel::session_ready_for_decode() const {
    return state_->session_.phase == SessionPhase::Ready;
}

std::vector<float> CpuModel::session_logits() const { return state_->workspace_.logits; }
RuntimeMetrics CpuModel::session_metrics() const { return state_->session_.metrics; }
CpuPrefillProfile CpuModel::session_prefill_profile() const { return state_->session_.prefill_profile; }
void CpuModel::clear_session_metrics() { state_->session_.metrics = {}; }
CpuModelMemoryStats CpuModel::session_memory_stats() const { return state_->memory_stats(); }
int CpuModel::session_vocab_size() const { return state_->shared->shape.vocab_size; }
CpuIsa CpuModel::session_isa() const { return state_->shared->options.isa; }
CpuKvCacheMode CpuModel::session_kv_cache_mode() const {
    return state_->shared->options.kv_cache_mode;
}

std::string CpuModel::session_backend_description() const {
    std::ostringstream out;
    out << "cpu-native isa=" << cpu_isa_name(state_->shared->options.isa)
        << " weights="
        << (state_->shared->native_checkpoint
                ? "gguf-native(q4_k,q6_k)"
                : "q4-group")
        << (state_->shared->native_checkpoint
                ? "" : ("-" + std::to_string(state_->shared->group_size)))
        << " kv=" << cpu_kv_cache_mode_name(state_->shared->options.kv_cache_mode)
        << " kv-page=" << state_->shared->options.kv_page_tokens
        << " prefill-chunk=" << state_->shared->options.prefill_chunk_tokens
        << " threads=" << (state_->shared->pool.size() + 1)
        << " affinity=" << cpu_affinity_name(state_->shared->options.affinity)
        << " numa=" << cpu_numa_mode_name(state_->shared->options.numa_mode)
        << " attention-threshold=" << state_->shared->options.attention_parallel_threshold
        << " attention-page-tile=" << state_->shared->options.attention_page_tile
        << " pinned-workers=" << state_->shared->pool.pinned_workers()
        << " pack="
        << (state_->shared->native_checkpoint
                ? "none"
                : (state_->shared->loaded_pack ? "hit" : "built"))
        << " hardware-best="
        << cpu_isa_name(state_->shared->capabilities.best_isa());
    return out.str();
}

const std::filesystem::path& CpuModel::session_pack_path() const {
    return state_->shared->pack_file;
}

bool CpuModel::session_loaded_from_pack() const { return state_->shared->loaded_pack; }

uint64_t CpuModel::session_attention_parallel_calls() const {
    return state_->session_.attention_parallel_calls;
}

CpuPrefixSnapshot CpuModel::export_session_prefix() const {
    return state_->export_prefix_snapshot();
}

void CpuModel::restore_session_prefix(CpuPrefixSnapshot snapshot,
                                      bool ready_for_decode) {
    state_->restore_prefix_snapshot(std::move(snapshot), ready_for_decode);
}

void CpuInferenceSession::reset() { owner_->reset_session(); }
void CpuInferenceSession::prefill(const std::vector<int32_t>& tokens) {
    owner_->prefill_session(tokens);
}
void CpuInferenceSession::prefill(const std::vector<int32_t>& tokens,
                                  const PromptEmbedding& embeddings) {
    owner_->prefill_session(tokens, embeddings);
}
int32_t CpuInferenceSession::decode() { return owner_->decode_session(); }
void CpuInferenceSession::eval_token(int32_t token) {
    owner_->eval_token_session(token);
}
void CpuInferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->set_session_generation(generation);
}
int CpuInferenceSession::position() const { return owner_->session_position(); }
bool CpuInferenceSession::ready_for_decode() const {
    return owner_->session_ready_for_decode();
}

std::vector<float> CpuDiagnostics::copy_logits() const { return owner_->session_logits(); }
RuntimeMetrics CpuDiagnostics::runtime_metrics() const { return owner_->session_metrics(); }
CpuPrefillProfile CpuDiagnostics::prefill_profile() const { return owner_->session_prefill_profile(); }
void CpuDiagnostics::clear_runtime_metrics() { owner_->clear_session_metrics(); }
CpuModelMemoryStats CpuDiagnostics::memory_stats() const { return owner_->session_memory_stats(); }
int CpuDiagnostics::vocab_size() const { return owner_->session_vocab_size(); }
CpuIsa CpuDiagnostics::isa() const { return owner_->session_isa(); }
CpuKvCacheMode CpuDiagnostics::kv_cache_mode() const { return owner_->session_kv_cache_mode(); }
std::string CpuDiagnostics::backend_description() const {
    return owner_->session_backend_description();
}
const std::filesystem::path& CpuDiagnostics::pack_path() const {
    return owner_->session_pack_path();
}
bool CpuDiagnostics::loaded_from_pack() const { return owner_->session_loaded_from_pack(); }
uint64_t CpuDiagnostics::attention_parallel_calls() const {
    return owner_->session_attention_parallel_calls();
}

CpuPrefixSnapshot CpuPersistence::export_prefix_snapshot() const {
    return owner_->export_session_prefix();
}
void CpuPersistence::restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                             bool ready_for_decode) {
    owner_->restore_session_prefix(std::move(snapshot), ready_for_decode);
}

} // namespace celeg

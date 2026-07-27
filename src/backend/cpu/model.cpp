#include "lfm/backend/cpu/model.hpp"

#include "detail/model_internal.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace lfm {

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
                   CpuModelOptions options, GenerationConfig generation)
    : impl_(std::make_unique<Impl>(
          std::make_shared<Impl::Shared>(path, context, std::move(options)),
          generation, -1)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

CpuModel::CpuModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {
    if (!impl_) throw std::invalid_argument("CPU model implementation is required");
}

CpuModel::~CpuModel() = default;

CpuModel::CpuModel(CpuModel&& other) noexcept
    : impl_(std::move(other.impl_)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

CpuModel& CpuModel::operator=(CpuModel&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

std::unique_ptr<CpuModel> CpuModel::clone_session() const {
    return clone_session_on_node(-1);
}

std::unique_ptr<CpuModel> CpuModel::clone_session_on_node(int numa_node) const {
    return std::unique_ptr<CpuModel>(new CpuModel(
        std::make_unique<Impl>(impl_->shared, impl_->generation, numa_node)));
}

std::vector<std::shared_ptr<CpuKvPagePool>> CpuModel::shared_kv_pools() const {
    return impl_->shared->kv_pools;
}

void CpuModel::reset_session() { impl_->reset(); }

void CpuModel::prefill_session(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) throw std::invalid_argument("CPU prefill needs at least one token");
    if (tokens.size() > static_cast<size_t>(impl_->shared->max_context)) {
        throw std::invalid_argument("CPU prefill exceeds context");
    }
    for (int32_t token : tokens) {
        if (token < 0 || token >= impl_->shared->shape.vocab_size) {
            throw std::invalid_argument("CPU token out of range");
        }
    }
    impl_->reset();
    impl_->phase = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    if (tokens.size() < impl_->shared->options.prefill_chunk_threshold) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            impl_->seen[static_cast<size_t>(tokens[i])] = 1;
            impl_->forward_token(tokens[i], i + 1 == tokens.size());
        }
    } else {
        const size_t chunk = impl_->shared->options.prefill_chunk_tokens;
        for (size_t begin = 0; begin < tokens.size(); begin += chunk) {
            const size_t end = std::min(tokens.size(), begin + chunk);
            for (size_t i = begin; i < end; ++i) {
                impl_->seen[static_cast<size_t>(tokens[i])] = 1;
            }
            impl_->forward_chunk(
                std::span<const int32_t>(tokens.data() + begin, end - begin),
                end == tokens.size());
        }
    }
    const auto ended = std::chrono::steady_clock::now();
    impl_->metrics.last_prefill_ms =
        std::chrono::duration<double, std::milli>(ended - started).count();
    impl_->metrics.prefill_tokens = tokens.size();
    impl_->phase = SessionPhase::Ready;
}

int32_t CpuModel::decode_session() {
    if (impl_->phase != SessionPhase::Ready) {
        throw std::runtime_error("CPU model is not ready for decode");
    }
    const auto started = std::chrono::steady_clock::now();
    const int32_t token = impl_->sample();
    impl_->seen[static_cast<size_t>(token)] = 1;
    impl_->forward_token(token, true);
    const auto ended = std::chrono::steady_clock::now();
    impl_->metrics.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    ++impl_->metrics.decoded_tokens;
    return token;
}

void CpuModel::eval_token_session(int32_t token) {
    if (impl_->phase != SessionPhase::Ready) {
        throw std::runtime_error("CPU model is not ready for token evaluation");
    }
    if (token < 0 || token >= impl_->shared->shape.vocab_size) {
        throw std::invalid_argument("CPU token out of range");
    }
    const auto started = std::chrono::steady_clock::now();
    impl_->seen[static_cast<size_t>(token)] = 1;
    impl_->forward_token(token, true);
    const auto ended = std::chrono::steady_clock::now();
    impl_->metrics.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    ++impl_->metrics.decoded_tokens;
}

void CpuModel::set_session_generation(GenerationConfig generation) {
    impl_->set_generation(generation);
}

int CpuModel::session_position() const { return impl_->position_value; }

bool CpuModel::session_ready_for_decode() const {
    return impl_->phase == SessionPhase::Ready;
}

std::vector<float> CpuModel::session_logits() const { return impl_->logits; }
RuntimeMetrics CpuModel::session_metrics() const { return impl_->metrics; }
void CpuModel::clear_session_metrics() { impl_->metrics = {}; }
CpuModelMemoryStats CpuModel::session_memory_stats() const { return impl_->memory_stats(); }
int CpuModel::session_vocab_size() const { return impl_->shared->shape.vocab_size; }
CpuIsa CpuModel::session_isa() const { return impl_->shared->options.isa; }
CpuKvCacheMode CpuModel::session_kv_cache_mode() const {
    return impl_->shared->options.kv_cache_mode;
}

std::string CpuModel::session_backend_description() const {
    std::ostringstream out;
    out << "cpu-native isa=" << cpu_isa_name(impl_->shared->options.isa)
        << " q4-group=" << impl_->shared->group_size
        << " kv=" << cpu_kv_cache_mode_name(impl_->shared->options.kv_cache_mode)
        << " kv-page=" << impl_->shared->options.kv_page_tokens
        << " prefill-chunk=" << impl_->shared->options.prefill_chunk_tokens
        << " threads=" << (impl_->shared->pool.size() + 1)
        << " affinity=" << cpu_affinity_name(impl_->shared->options.affinity)
        << " numa=" << cpu_numa_mode_name(impl_->shared->options.numa_mode)
        << " attention-threshold=" << impl_->shared->options.attention_parallel_threshold
        << " attention-page-tile=" << impl_->shared->options.attention_page_tile
        << " pinned-workers=" << impl_->shared->pool.pinned_workers()
        << " pack=" << (impl_->shared->loaded_pack ? "hit" : "built")
        << " hardware-best="
        << cpu_isa_name(impl_->shared->capabilities.best_isa());
    return out.str();
}

const std::filesystem::path& CpuModel::session_pack_path() const {
    return impl_->shared->pack_file;
}

bool CpuModel::session_loaded_from_pack() const { return impl_->shared->loaded_pack; }

uint64_t CpuModel::session_attention_parallel_calls() const {
    return impl_->attention_parallel_calls;
}

CpuPrefixSnapshot CpuModel::export_session_prefix() const {
    return impl_->export_prefix_snapshot();
}

void CpuModel::restore_session_prefix(CpuPrefixSnapshot snapshot,
                                      bool ready_for_decode) {
    impl_->restore_prefix_snapshot(std::move(snapshot), ready_for_decode);
}

void CpuInferenceSession::reset() { owner_->reset_session(); }
void CpuInferenceSession::prefill(const std::vector<int32_t>& tokens) {
    owner_->prefill_session(tokens);
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

} // namespace lfm

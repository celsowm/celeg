#include "lfm/detail/model/impl.hpp"

namespace lfm {

LfmModel::LfmModel(const std::string& model_path,
                   int max_context,
                   ModelOptions options,
                   GenerationConfig generation)
    : impl_(std::make_unique<Impl>(model_path, max_context,
                                   std::move(options), std::move(generation))),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

LfmModel::~LfmModel() = default;

IPackedSession& LfmModel::packed_session() { return *impl_; }
const IPackedSession& LfmModel::packed_session() const { return *impl_; }

void LfmInferenceSession::reset(bool allocate_local_kv) { owner_->impl_->reset(allocate_local_kv); }
void LfmInferenceSession::prefill(const std::vector<int32_t>& tokens) { owner_->impl_->prefill(tokens); }
void LfmInferenceSession::prefill_chunk(const std::vector<int32_t>& tokens, bool begin, bool finalize) {
    owner_->impl_->prefill_chunk(tokens, begin, finalize);
}
void LfmInferenceSession::prefill_chunk_paged(const std::vector<int32_t>& tokens, bool begin,
                                              bool finalize, PhysicalPagedKvCache& paged_kv,
                                              const std::vector<uint32_t>& page_table) {
    owner_->impl_->prefill_chunk_paged(tokens, begin, finalize, paged_kv, page_table);
}
int32_t LfmInferenceSession::decode() { return owner_->impl_->decode(); }
void LfmInferenceSession::decode_async_begin() { owner_->impl_->decode_async_begin(); }
int32_t LfmInferenceSession::decode_async_finish() { return owner_->impl_->decode_async_finish(); }
void LfmInferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->impl_->set_generation_config(std::move(generation));
}
void LfmInferenceSession::release_local_kv_cache() { owner_->impl_->release_local_kv_cache(); }
bool LfmInferenceSession::local_kv_cache_available() const { return owner_->impl_->local_kv_cache_available(); }
SessionPhase LfmInferenceSession::phase() const { return owner_->impl_->phase(); }
bool LfmInferenceSession::ready_for_decode() const { return owner_->impl_->ready_for_decode(); }
bool LfmInferenceSession::decode_pending() const { return owner_->impl_->decode_pending(); }
int LfmInferenceSession::position() const { return owner_->impl_->position(); }

std::vector<float> LfmDiagnostics::copy_logits() const { return owner_->impl_->copy_logits(); }
int LfmDiagnostics::vocab_size() const { return owner_->impl_->shape_.vocab_size; }
DecodeBenchmark LfmDiagnostics::benchmark_decode(int warmup_steps, int measured_steps) {
    return owner_->impl_->benchmark_decode(warmup_steps, measured_steps);
}
ModelMemoryStats LfmDiagnostics::memory_stats() const { return owner_->impl_->memory_stats(); }
LfmDiagnostics::ExpertOffloadStats LfmDiagnostics::expert_offload_stats() const {
    return owner_->impl_->expert_offload_stats();
}
RuntimeMetrics LfmDiagnostics::runtime_metrics() const { return owner_->impl_->runtime_metrics(); }
void LfmDiagnostics::clear_runtime_metrics() { owner_->impl_->clear_runtime_metrics(); }
bool LfmDiagnostics::cuda_graph_ready() const { return owner_->impl_->cuda_graph_ready(); }

} // namespace lfm



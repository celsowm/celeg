#include "celeg/detail/model/compiled_model.hpp"

namespace celeg {

CudaModel::CudaModel(const std::string& model_path,
                   int max_context,
                   CudaModelOptions options,
                   GenerationConfig generation)
    : state_(std::make_unique<CudaCompiledModel>(model_path, max_context,
                                   std::move(options), std::move(generation))),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

CudaModel::~CudaModel() = default;

bool CudaCompiledModel::packed_segmented_attention_callback(
    const void* owner, int host_position) {
    return static_cast<const CudaCompiledModel*>(owner)->use_segmented_attention(host_position);
}

void CudaCompiledModel::packed_expert_residency_callback(
    void* owner, int layer, const int* sel_dev, int rows,
    cudaStream_t stream, const float* route_scores_dev) {
    static_cast<CudaCompiledModel*>(owner)->ensure_moe_experts_resident_packed(
        layer, sel_dev, rows, stream, route_scores_dev);
}

PackedSessionContext CudaCompiledModel::packed_session_context() {
    PackedSessionContext context;
    context.owner = this;
    context.phase_state = &session_.phase_;
    context.position_state = &session_.position_;
    context.max_context_value = max_context_;
    context.local_kv_cache_available_state = &local_kv_cache_available_;
    context.active_segmented_attention_state = &session_.active_segmented_attention_;
    context.options_state = &resources_.options_;
    context.generation_state = &session_.generation_;
    context.shape_state = &resources_.shape_;
    context.weights_state = resources_.weights_;
    context.logits_state = &workspace_.logits_;
    context.seen_tokens_state = &sampling_.seen_tokens;
    context.rng_state_buffer = &sampling_.rng_state;
    context.sampled_device_state = &sampling_.sampled_device;
    context.position_device_state = &position_device_;
    context.sampled_host_state = &sampling_.sampled_host;
    context.layers_state = &resources_.layers_;
    context.metrics_state = &session_.metrics_;
    context.weight_layout_state = resources_.weight_layout_.get();
    context.embedding_weight = resources_.embedding_;
    context.logits_weight_value = resources_.lm_head_ ? resources_.lm_head_ : resources_.embedding_;
    context.final_norm_value = resources_.final_norm_;
    context.rope_cos_value = workspace_.rope_cos_.data();
    context.rope_sin_value = workspace_.rope_sin_.data();
    context.segmented_attention = &CudaCompiledModel::packed_segmented_attention_callback;
    context.ensure_expert_residency = &CudaCompiledModel::packed_expert_residency_callback;
    return context;
}

PackedSessionContext packed_session_context(CudaModel& model) {
    return model.state_->packed_session_context();
}

void CudaInferenceSession::reset(bool allocate_local_kv) { owner_->state_->reset(allocate_local_kv); }
void CudaInferenceSession::prefill(const std::vector<int32_t>& tokens) { owner_->state_->prefill(tokens); }
void CudaInferenceSession::prefill_chunk(const std::vector<int32_t>& tokens, bool begin, bool finalize) {
    owner_->state_->prefill_chunk(tokens, begin, finalize);
}
void CudaInferenceSession::prefill_chunk_paged(const std::vector<int32_t>& tokens, bool begin,
                                              bool finalize, PhysicalPagedKvCache& paged_kv,
                                              const std::vector<uint32_t>& page_table) {
    owner_->state_->prefill_chunk_paged(tokens, begin, finalize, paged_kv, page_table);
}
int32_t CudaInferenceSession::decode() { return owner_->state_->decode(); }
void CudaInferenceSession::decode_async_begin() { owner_->state_->decode_async_begin(); }
int32_t CudaInferenceSession::decode_async_finish() { return owner_->state_->decode_async_finish(); }
void CudaInferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->state_->set_generation_config(std::move(generation));
}
void CudaInferenceSession::release_local_kv_cache() { owner_->state_->release_local_kv_cache(); }
bool CudaInferenceSession::local_kv_cache_available() const { return owner_->state_->local_kv_cache_available(); }
SessionPhase CudaInferenceSession::phase() const { return owner_->state_->phase(); }
bool CudaInferenceSession::ready_for_decode() const { return owner_->state_->ready_for_decode(); }
bool CudaInferenceSession::decode_pending() const { return owner_->state_->decode_pending(); }
int CudaInferenceSession::position() const { return owner_->state_->position(); }

std::vector<float> CudaModelDiagnostics::copy_logits() const { return owner_->state_->copy_logits(); }
int CudaModelDiagnostics::vocab_size() const { return owner_->state_->resources_.shape_.vocab_size; }
DecodeBenchmark CudaModelDiagnostics::benchmark_decode(int warmup_steps, int measured_steps) {
    return owner_->state_->benchmark_decode(warmup_steps, measured_steps);
}
ModelMemoryStats CudaModelDiagnostics::memory_stats() const { return owner_->state_->memory_stats(); }
CudaModelDiagnostics::ExpertOffloadStats CudaModelDiagnostics::expert_offload_stats() const {
    return owner_->state_->expert_offload_stats();
}
RuntimeMetrics CudaModelDiagnostics::runtime_metrics() const { return owner_->state_->runtime_metrics(); }
void CudaModelDiagnostics::clear_runtime_metrics() { owner_->state_->clear_runtime_metrics(); }
bool CudaModelDiagnostics::cuda_graph_ready() const { return owner_->state_->cuda_graph_ready(); }

} // namespace celeg

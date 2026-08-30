#include "detail/compiled_model.hpp"

#include <functional>

namespace celeg {

CudaModel::CudaModel(const std::string& model_path,
                   int max_context,
                   CudaModelOptions options,
                   GenerationConfig generation,
                   std::shared_ptr<const RuntimeContext> runtime,
                   int tokenizer_vocab_size)
    : state_(std::make_unique<CudaCompiledModel>(
          model_path, max_context, cuda_model_options_from_environment(std::move(options)),
          std::move(generation), std::move(runtime), tokenizer_vocab_size)) {}

CudaModel::~CudaModel() = default;

PackedSessionContext CudaCompiledModel::packed_session_context() {
    PackedSessionContext context;
    context.owner = PackedExecutionServices{
        &session_,
        &resources_.plan_,
        &resources_.program_,
        resources_.weights_.get(),
        &workspace_.residency_workspace_};
    context.storage_generation_value = storage_generation_;
    const uint64_t execution_plan_fingerprint = resources_.plan_.fingerprint();
    const uint64_t compiled_program_id = static_cast<uint64_t>(
        std::hash<std::string>{}(resources_.program_.semantic_fingerprint));
    const int device_ordinal = resources_.plan_.device().device_ordinal;
    context.compatibility_key = PackedCompatibilityKey{
        resources_.weights_.get(),
        execution_plan_fingerprint,
        compiled_program_id,
        device_ordinal};

    context.session.phase_state = &session_.phase_;
    context.session.position_state = &session_.position_;
    context.session.local_kv_cache_available_state = &local_kv_cache_available_;
    context.session.active_segmented_attention_state =
        &session_.active_segmented_attention_;
    context.session.generation_state = &session_.generation_;
    context.session.logits_state = &workspace_.logits_;
    context.session.seen_tokens_state = &sampling_.seen_tokens;
    context.session.rng_state_buffer = &sampling_.rng_state;
    context.session.sampled_device_state = &sampling_.sampled_device;
    context.session.position_device_state = &position_device_;
    context.session.sampled_host_state = &sampling_.sampled_host;
    context.session.layers_state = &resources_.layers_;
    context.session.metrics_state = &session_.metrics_;

    context.immutable.max_context_value = max_context_;
    context.immutable.options_state = &resources_.options();
    context.immutable.shape_state = &resources_.shape_;
    context.immutable.program_state = &resources_.program_;
    context.immutable.weights_state = resources_.weights_;
    context.immutable.weight_layout_state = resources_.weight_layout_.get();
    context.immutable.embedding_weight = resources_.embedding_;
    context.immutable.logits_weight_value =
        resources_.lm_head_ ? resources_.lm_head_ : resources_.embedding_;
    context.immutable.final_norm_value = resources_.final_norm_;
    return context;
}

PackedSessionContext packed_session_context(CudaModel& model) {
    return model.state_->packed_session_context();
}

void CudaInferenceSession::reset(bool allocate_local_kv) { owner_->state_->reset(allocate_local_kv); }
void CudaInferenceSession::prefill(const std::vector<int32_t>& tokens) { owner_->state_->prefill(tokens); }
void CudaInferenceSession::prefill(const std::vector<int32_t>& tokens,
                                   const PromptEmbedding& embeddings) {
    owner_->state_->prefill(tokens, embeddings);
}
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
int CudaModelDiagnostics::vocab_size() const { return owner_->state_->resources_.dims_.vocab_size; }
DecodeBenchmark CudaModelDiagnostics::benchmark_decode(int warmup_steps, int measured_steps) {
    return owner_->state_->benchmark_decode(warmup_steps, measured_steps);
}
ModelMemoryStats CudaModelDiagnostics::memory_stats() const { return owner_->state_->memory_stats(); }
CudaModelDiagnostics::ExpertOffloadStats CudaModelDiagnostics::expert_offload_stats() const {
    return owner_->state_->expert_offload_stats();
}
RuntimeMetrics CudaModelDiagnostics::runtime_metrics() const { return owner_->state_->runtime_metrics(); }
void CudaModelDiagnostics::clear_runtime_metrics() { owner_->state_->clear_runtime_metrics(); }
std::string CudaModelDiagnostics::execution_plan_description() const {
    return owner_->state_->execution_plan_description();
}
bool CudaModelDiagnostics::cuda_graph_ready() const { return owner_->state_->cuda_graph_ready(); }

}

#include "celeg/detail/model/impl.hpp"

namespace celeg {

Model::Model(const std::string& model_path,
                   int max_context,
                   ModelOptions options,
                   GenerationConfig generation)
    : impl_(std::make_unique<Impl>(model_path, max_context,
                                   std::move(options), std::move(generation))),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

Model::~Model() = default;

bool Model::Impl::packed_segmented_attention_callback(
    const void* owner, int host_position) {
    return static_cast<const Impl*>(owner)->use_segmented_attention(host_position);
}

void Model::Impl::packed_expert_residency_callback(
    void* owner, int layer, const int* sel_dev, int rows,
    cudaStream_t stream, const float* route_scores_dev) {
    static_cast<Impl*>(owner)->ensure_moe_experts_resident_packed(
        layer, sel_dev, rows, stream, route_scores_dev);
}

PackedSessionContext Model::Impl::packed_session_context() {
    PackedSessionContext context;
    context.owner = this;
    context.phase_state = &phase_;
    context.position_state = &position_;
    context.max_context_value = max_context_;
    context.local_kv_cache_available_state = &local_kv_cache_available_;
    context.active_segmented_attention_state = &active_segmented_attention_;
    context.options_state = &options_;
    context.generation_state = &generation_;
    context.shape_state = &shape_;
    context.weights_state = weights_;
    context.logits_state = &logits_;
    context.seen_tokens_state = &seen_tokens_;
    context.rng_state_buffer = &rng_state_;
    context.sampled_device_state = &sampled_device_;
    context.position_device_state = &position_device_;
    context.sampled_host_state = &sampled_host_;
    context.layers_state = &layers_;
    context.metrics_state = &metrics_;
    context.weight_layout_state = weight_layout_.get();
    context.embedding_weight = embedding_;
    context.logits_weight_value = lm_head_ ? lm_head_ : embedding_;
    context.final_norm_value = final_norm_;
    context.rope_cos_value = rope_cos_.data();
    context.rope_sin_value = rope_sin_.data();
    context.segmented_attention = &Impl::packed_segmented_attention_callback;
    context.ensure_expert_residency = &Impl::packed_expert_residency_callback;
    return context;
}

PackedSessionContext packed_session_context(Model& model) {
    return model.impl_->packed_session_context();
}

void InferenceSession::reset(bool allocate_local_kv) { owner_->impl_->reset(allocate_local_kv); }
void InferenceSession::prefill(const std::vector<int32_t>& tokens) { owner_->impl_->prefill(tokens); }
void InferenceSession::prefill_chunk(const std::vector<int32_t>& tokens, bool begin, bool finalize) {
    owner_->impl_->prefill_chunk(tokens, begin, finalize);
}
void InferenceSession::prefill_chunk_paged(const std::vector<int32_t>& tokens, bool begin,
                                              bool finalize, PhysicalPagedKvCache& paged_kv,
                                              const std::vector<uint32_t>& page_table) {
    owner_->impl_->prefill_chunk_paged(tokens, begin, finalize, paged_kv, page_table);
}
int32_t InferenceSession::decode() { return owner_->impl_->decode(); }
void InferenceSession::decode_async_begin() { owner_->impl_->decode_async_begin(); }
int32_t InferenceSession::decode_async_finish() { return owner_->impl_->decode_async_finish(); }
void InferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->impl_->set_generation_config(std::move(generation));
}
void InferenceSession::release_local_kv_cache() { owner_->impl_->release_local_kv_cache(); }
bool InferenceSession::local_kv_cache_available() const { return owner_->impl_->local_kv_cache_available(); }
SessionPhase InferenceSession::phase() const { return owner_->impl_->phase(); }
bool InferenceSession::ready_for_decode() const { return owner_->impl_->ready_for_decode(); }
bool InferenceSession::decode_pending() const { return owner_->impl_->decode_pending(); }
int InferenceSession::position() const { return owner_->impl_->position(); }

std::vector<float> ModelDiagnostics::copy_logits() const { return owner_->impl_->copy_logits(); }
int ModelDiagnostics::vocab_size() const { return owner_->impl_->shape_.vocab_size; }
DecodeBenchmark ModelDiagnostics::benchmark_decode(int warmup_steps, int measured_steps) {
    return owner_->impl_->benchmark_decode(warmup_steps, measured_steps);
}
ModelMemoryStats ModelDiagnostics::memory_stats() const { return owner_->impl_->memory_stats(); }
ModelDiagnostics::ExpertOffloadStats ModelDiagnostics::expert_offload_stats() const {
    return owner_->impl_->expert_offload_stats();
}
RuntimeMetrics ModelDiagnostics::runtime_metrics() const { return owner_->impl_->runtime_metrics(); }
void ModelDiagnostics::clear_runtime_metrics() { owner_->impl_->clear_runtime_metrics(); }
bool ModelDiagnostics::cuda_graph_ready() const { return owner_->impl_->cuda_graph_ready(); }

} // namespace celeg

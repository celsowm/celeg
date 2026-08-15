#include "celeg/serve/cpu_inference_service.hpp"
#include "celeg/serve/visual_request.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace celeg::serve {

CpuInferenceService::CpuInferenceService(const std::string& model_path,
                                         int max_context,
                                         CpuModelOptions model_options,
                                         CpuConcurrentEngineOptions engine_options,
                                         VisualEmbeddingProvider visual_embeddings,
                                         std::shared_ptr<const RuntimeContext> runtime)
    : engine_(model_path, max_context, std::move(model_options),
              engine_options, std::move(runtime)),
      visual_embeddings_(std::move(visual_embeddings)) {
    model_info_.name = std::filesystem::path(model_path).stem().string();
    model_info_.backend = "cpu";
    model_info_.max_context = max_context;
    model_info_.limits.max_active_requests = engine_options.max_active_requests;
    model_info_.limits.max_batched_tokens = engine_options.max_batched_tokens;
    model_info_.limits.prefill_chunk_tokens = engine_options.long_prefill_chunk_tokens;
    model_info_.limits.supports_multimodal =
        static_cast<bool>(visual_embeddings_);
}

RequestId CpuInferenceService::submit(GenerateRequest request) {
    materialize_visual_prompt(request, visual_embeddings_);
    ConcurrentRequestOptions options;
    options.max_new_tokens = request.max_output_tokens;
    options.eos_tokens = request.eos_token_ids;
    options.priority = request.priority;
    options.generation = request.generation;
    options.prompt_embedding = std::move(request.prompt_embedding);

    const RequestId id =
        engine_.submit(std::move(request.prompt_tokens), options);

    lifecycle_.submitted(id, request.eos_token_ids);
    return id;
}

GenerateEvent CpuInferenceService::poll(RequestId id, std::size_t max_tokens) {
    GenerateEvent event;
    event.request_id = id;
    const PollResult result = engine_.poll(id, max_tokens);
    event.tokens = result.tokens;
    event.finished = result.finished;
    event.status = result.status;
    event.error = result.error;

    if (event.finished) {
        event.finish_reason = lifecycle_.finish_reason(id, event.status, event.tokens);
    } else {
        lifecycle_.finish_reason(id, event.status, event.tokens);
    }
    return event;
}

RequestStatus CpuInferenceService::status(RequestId id) const {
    return engine_.status(id);
}

bool CpuInferenceService::cancel(RequestId id) {
    return engine_.cancel(id);
}

bool CpuInferenceService::release(RequestId id) {
    const bool released = engine_.release(id);
    if (released) {
        lifecycle_.released(id);
    }
    return released;
}

ModelInfo CpuInferenceService::model_info() const { return model_info_; }

ServingMetrics CpuInferenceService::metrics() const {
    const ConcurrentMetrics snapshot = engine_.metrics();
    const CpuConcurrentMetricsExtras extras = engine_.metrics_extras();
    ServingMetrics result;
    result.submitted_requests = snapshot.submitted;
    result.completed_requests = snapshot.completed;
    result.cancelled_requests = snapshot.cancelled;
    result.failed_requests = snapshot.failed;
    result.active_requests = snapshot.active_requests;
    result.queued_requests = snapshot.queued_requests;
    const double cumulative_prefill_ms =
        snapshot.cumulative_ragged_prefill_ms + extras.cumulative_direct_prefill_ms;
    if (cumulative_prefill_ms > 0.0) {
        result.prefill_tokens_per_second =
            static_cast<double>(snapshot.prefill_tokens) * 1000.0 / cumulative_prefill_ms;
    }
    if (snapshot.cumulative_packed_decode_ms > 0.0) {
        result.decode_tokens_per_second =
            static_cast<double>(snapshot.decoded_tokens) * 1000.0 /
            snapshot.cumulative_packed_decode_ms;
    }
    if (snapshot.ttft_samples != 0) {
        result.average_ttft_ms = snapshot.average_ttft_ms();
    }
    if (snapshot.itl_samples != 0) {
        result.average_itl_ms = snapshot.average_itl_ms();
    }
    return result;
}

bool CpuInferenceService::step() { return engine_.step(); }
void CpuInferenceService::start() { engine_.start(); }
void CpuInferenceService::stop() { engine_.stop(); }

}

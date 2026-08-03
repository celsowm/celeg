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
                                         VisualEmbeddingProvider visual_embeddings)
    : engine_(model_path, max_context, std::move(model_options),
              std::move(engine_options)),
      visual_embeddings_(std::move(visual_embeddings)) {
    model_info_.name = std::filesystem::path(model_path).stem().string();
    model_info_.backend = "cpu";
    model_info_.max_context = max_context;
}

RequestId CpuInferenceService::submit(GenerateRequest request) {
    materialize_visual_prompt(request, visual_embeddings_);
    ConcurrentRequestOptions options;
    options.max_new_tokens = request.max_output_tokens;
    options.eos_token = request.eos_token_id;
    options.priority = request.priority;
    options.generation = request.generation;
    options.prompt_embedding = std::move(request.prompt_embedding);

    const RequestId id =
        engine_.submit(std::move(request.prompt_tokens), options);

    lifecycle_.submitted(id, request.eos_token_id);
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
    const CpuConcurrentMetrics snapshot = engine_.metrics();
    ServingMetrics result;
    result.submitted_requests = snapshot.submitted_requests;
    result.completed_requests = snapshot.completed_requests;
    result.cancelled_requests = snapshot.cancelled_requests;
    result.failed_requests = snapshot.failed_requests;
    result.active_requests = snapshot.active_requests;
    result.queued_requests = snapshot.queued_requests;
    if (snapshot.cumulative_prefill_ms > 0.0) {
        result.prefill_tokens_per_second = snapshot.prefill_tokens_per_second();
    }
    if (snapshot.cumulative_decode_ms > 0.0) {
        result.decode_tokens_per_second = snapshot.decode_tokens_per_second();
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

} // namespace celeg::serve

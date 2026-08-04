#include "celeg/backend/cuda/cuda_inference_service.hpp"
#include "celeg/serve/visual_request.hpp"

#include <filesystem>
#include <limits>
#include <utility>

namespace celeg::serve {

namespace {

RequestStatus map_status(celeg::RequestStatus status) {
    switch (status) {
        case celeg::RequestStatus::Queued: return RequestStatus::Queued;
        case celeg::RequestStatus::Prefill: return RequestStatus::Prefill;
        case celeg::RequestStatus::Decoding: return RequestStatus::Decoding;
        case celeg::RequestStatus::Finished: return RequestStatus::Finished;
        case celeg::RequestStatus::Cancelled: return RequestStatus::Cancelled;
        case celeg::RequestStatus::Failed: return RequestStatus::Failed;
    }
    return RequestStatus::Failed;
}

} // namespace

CudaInferenceService::CudaInferenceService(std::string model_path,
                                           int max_context,
                                           CudaModelOptions model_options,
                                           ConcurrentEngineOptions engine_options,
                                           VisualEmbeddingProvider visual_embeddings,
                                           std::shared_ptr<const RuntimeContext> runtime)
    : engine_(model_path, max_context, std::move(model_options),
              engine_options, std::move(runtime)),
      visual_embeddings_(std::move(visual_embeddings)) {
    model_info_.name = std::filesystem::path(model_path).stem().string();
    model_info_.backend = "cuda";
    model_info_.max_context = max_context;
    model_info_.limits.max_active_requests = engine_options.max_active_requests;
    model_info_.limits.max_batched_tokens = engine_options.max_batched_tokens;
    model_info_.limits.prefill_chunk_tokens = engine_options.prefill_chunk_tokens;
    model_info_.limits.supports_paged_kv = true;
    model_info_.limits.supports_packed_decode = engine_options.packed_decode;
    model_info_.limits.supports_multimodal =
        static_cast<bool>(visual_embeddings_);
}

RequestId CudaInferenceService::submit(GenerateRequest request) {
    materialize_visual_prompt(request, visual_embeddings_);
    if (request.max_output_tokens > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("max_output_tokens exceeds CUDA request limit");
    }
    ConcurrentRequestOptions options;
    options.max_new_tokens = static_cast<int>(request.max_output_tokens);
    options.eos_tokens = request.eos_token_ids;
    options.priority = request.priority;
    options.generation = request.generation;
    options.prompt_embedding = std::move(request.prompt_embedding);

    const ConcurrentEngine::RequestId id =
        engine_.submit(std::move(request.prompt_tokens), options);

    lifecycle_.submitted(id, request.eos_token_ids);
    return id;
}

GenerateEvent CudaInferenceService::poll(RequestId id, std::size_t max_tokens) {
    const PollResult result = engine_.poll(id, max_tokens);

    GenerateEvent event;
    event.request_id = id;
    event.tokens = result.tokens;
    event.finished = result.finished;
    event.status = map_status(result.status);
    event.error = result.error;

    if (event.finished) {
        event.finish_reason = lifecycle_.finish_reason(id, event.status, event.tokens);
    } else {
        lifecycle_.finish_reason(id, event.status, event.tokens);
    }
    return event;
}

RequestStatus CudaInferenceService::status(RequestId id) const {
    return map_status(engine_.status(id));
}

bool CudaInferenceService::cancel(RequestId id) { return engine_.cancel(id); }

bool CudaInferenceService::release(RequestId id) {
    const bool released = engine_.release(id);
    if (released) {
        lifecycle_.released(id);
    }
    return released;
}

ModelInfo CudaInferenceService::model_info() const { return model_info_; }

ServingMetrics CudaInferenceService::metrics() const {
    const ConcurrentMetrics snapshot = engine_.metrics();
    ServingMetrics result;
    result.submitted_requests = snapshot.submitted;
    result.completed_requests = snapshot.completed;
    result.cancelled_requests = snapshot.cancelled;
    result.failed_requests = snapshot.failed;
    result.active_requests = snapshot.active_requests;
    result.queued_requests = snapshot.queued_requests;
    if (snapshot.ttft_samples != 0) {
        result.average_ttft_ms = snapshot.average_ttft_ms();
    }
    if (snapshot.itl_samples != 0) {
        result.average_itl_ms = snapshot.average_itl_ms();
    }
    return result;
}

bool CudaInferenceService::step() { return engine_.step(); }
void CudaInferenceService::start() { engine_.start(); }
void CudaInferenceService::stop() { engine_.stop(); }

} // namespace celeg::serve

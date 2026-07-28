#include "lfm/serve/cuda_inference_service.hpp"

#include <filesystem>
#include <limits>
#include <utility>

namespace lfm::serve {

namespace {

RequestStatus map_status(lfm::RequestStatus status) {
    switch (status) {
        case lfm::RequestStatus::Queued: return RequestStatus::Queued;
        case lfm::RequestStatus::Prefill: return RequestStatus::Prefill;
        case lfm::RequestStatus::Decoding: return RequestStatus::Decoding;
        case lfm::RequestStatus::Finished: return RequestStatus::Finished;
        case lfm::RequestStatus::Cancelled: return RequestStatus::Cancelled;
        case lfm::RequestStatus::Failed: return RequestStatus::Failed;
    }
    return RequestStatus::Failed;
}

FinishReason finish_reason_for(RequestStatus status, bool saw_eos) {
    switch (status) {
        case RequestStatus::Cancelled: return FinishReason::Cancelled;
        case RequestStatus::Failed: return FinishReason::Error;
        case RequestStatus::Finished:
            return saw_eos ? FinishReason::Stop : FinishReason::Length;
        default: return FinishReason::None;
    }
}

} // namespace

CudaInferenceService::CudaInferenceService(std::string model_path,
                                           int max_context,
                                           ModelOptions model_options,
                                           ConcurrentEngineOptions engine_options)
    : engine_(model_path, max_context, std::move(model_options),
              std::move(engine_options)) {
    model_info_.name = std::filesystem::path(model_path).stem().string();
    model_info_.backend = "cuda";
    model_info_.max_context = max_context;
}

RequestId CudaInferenceService::submit(GenerateRequest request) {
    ConcurrentRequestOptions options;
    options.max_new_tokens = static_cast<int>(request.max_output_tokens);
    options.eos_token = request.eos_token_id;
    options.priority = request.priority;
    options.generation = request.generation;

    const ConcurrentEngine::RequestId id =
        engine_.submit(std::move(request.prompt_tokens), options);

    std::lock_guard<std::mutex> lock(meta_mutex_);
    meta_.emplace(id, RequestMeta{request.eos_token_id, false});
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

    std::lock_guard<std::mutex> lock(meta_mutex_);
    auto it = meta_.find(id);
    const std::int32_t eos_token_id = it != meta_.end() ? it->second.eos_token_id : 7;
    if (it != meta_.end() && !event.tokens.empty() &&
        event.tokens.back() == eos_token_id) {
        it->second.saw_eos = true;
    }
    const bool saw_eos = it != meta_.end() && it->second.saw_eos;
    if (event.finished) event.finish_reason = finish_reason_for(event.status, saw_eos);
    return event;
}

RequestStatus CudaInferenceService::status(RequestId id) const {
    return map_status(engine_.status(id));
}

bool CudaInferenceService::cancel(RequestId id) { return engine_.cancel(id); }

bool CudaInferenceService::release(RequestId id) {
    const bool released = engine_.release(id);
    if (released) {
        std::lock_guard<std::mutex> lock(meta_mutex_);
        meta_.erase(id);
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
    result.average_ttft_ms = snapshot.average_ttft_ms();
    result.average_itl_ms = snapshot.average_itl_ms();
    return result;
}

bool CudaInferenceService::step() { return engine_.step(); }
void CudaInferenceService::start() { engine_.start(); }
void CudaInferenceService::stop() { engine_.stop(); }

} // namespace lfm::serve

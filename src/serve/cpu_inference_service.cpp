#include "lfm/serve/cpu_inference_service.hpp"

#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lfm::serve {

namespace {

RequestStatus map_status(CpuRequestStatus status) {
    switch (status) {
        case CpuRequestStatus::Queued: return RequestStatus::Queued;
        case CpuRequestStatus::Prefilling: return RequestStatus::Prefill;
        case CpuRequestStatus::Decoding: return RequestStatus::Decoding;
        case CpuRequestStatus::Completed: return RequestStatus::Finished;
        case CpuRequestStatus::Cancelled: return RequestStatus::Cancelled;
        case CpuRequestStatus::Failed: return RequestStatus::Failed;
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

CpuInferenceService::CpuInferenceService(const std::string& safetensors_path,
                                         int max_context,
                                         CpuModelOptions model_options,
                                         CpuConcurrentEngineOptions engine_options)
    : engine_(safetensors_path, max_context, std::move(model_options),
              std::move(engine_options)) {
    model_info_.name = std::filesystem::path(safetensors_path).stem().string();
    model_info_.backend = "cpu";
    model_info_.max_context = max_context;
}

RequestId CpuInferenceService::submit(GenerateRequest request) {
    CpuRequestOptions options;
    options.max_new_tokens = request.max_output_tokens;
    options.eos_token_id = request.eos_token_id;
    options.priority = request.priority;
    options.generation = request.generation;

    const CpuRequestId id =
        engine_.submit(std::move(request.prompt_tokens), options);

    std::lock_guard<std::mutex> lock(meta_mutex_);
    meta_.emplace(id, RequestMeta{request.eos_token_id, false});
    return id;
}

GenerateEvent CpuInferenceService::poll(RequestId id, std::size_t max_tokens) {
    // The CPU engine treats max_tokens == 0 as an invalid argument, whereas
    // this interface treats it as "drain everything currently buffered".
    const std::size_t query =
        max_tokens == 0 ? std::numeric_limits<std::size_t>::max() : max_tokens;

    GenerateEvent event;
    event.request_id = id;
    bool finished = false;
    event.tokens = engine_.poll(id, query, &finished);
    event.finished = finished;
    event.status = map_status(engine_.status(id));

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

RequestStatus CpuInferenceService::status(RequestId id) const {
    return map_status(engine_.status(id));
}

bool CpuInferenceService::cancel(RequestId id) {
    try {
        engine_.cancel(id);
    } catch (const std::out_of_range&) {
        return false;
    }
    return true;
}

bool CpuInferenceService::release(RequestId id) {
    const bool released = engine_.release(id);
    if (released) {
        std::lock_guard<std::mutex> lock(meta_mutex_);
        meta_.erase(id);
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
    result.prefill_tokens_per_second = snapshot.prefill_tokens_per_second();
    result.decode_tokens_per_second = snapshot.decode_tokens_per_second();
    result.average_ttft_ms = snapshot.average_ttft_ms();
    result.average_itl_ms = snapshot.average_itl_ms();
    return result;
}

bool CpuInferenceService::step() { return engine_.step(); }
void CpuInferenceService::start() { engine_.start(); }
void CpuInferenceService::stop() { engine_.stop(); }

} // namespace lfm::serve

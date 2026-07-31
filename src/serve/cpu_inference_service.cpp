#include "celeg/serve/cpu_inference_service.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace celeg::serve {

namespace {

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

CpuInferenceService::CpuInferenceService(const std::string& model_path,
                                         int max_context,
                                         CpuModelOptions model_options,
                                         CpuConcurrentEngineOptions engine_options)
    : engine_(model_path, max_context, std::move(model_options),
              std::move(engine_options)) {
    model_info_.name = std::filesystem::path(model_path).stem().string();
    model_info_.backend = "cpu";
    model_info_.max_context = max_context;
}

RequestId CpuInferenceService::submit(GenerateRequest request) {
    ConcurrentRequestOptions options;
    options.max_new_tokens = request.max_output_tokens;
    options.eos_token = request.eos_token_id;
    options.priority = request.priority;
    options.generation = request.generation;

    const RequestId id =
        engine_.submit(std::move(request.prompt_tokens), options);

    std::lock_guard<std::mutex> lock(meta_mutex_);
    meta_.emplace(id, RequestMeta{request.eos_token_id, false});
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
    return engine_.status(id);
}

bool CpuInferenceService::cancel(RequestId id) {
    return engine_.cancel(id);
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

} // namespace celeg::serve

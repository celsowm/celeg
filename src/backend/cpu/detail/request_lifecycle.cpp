#include "request_lifecycle.hpp"

#include "celeg/backend/cpu/model.hpp"

#include <algorithm>

namespace celeg {

void CpuRequestLifecycle::cancel(const std::shared_ptr<Request>& request,
                                  ConcurrentMetrics& metrics) {
    if (!request || is_terminal(request->status)) return;
    request->status = RequestStatus::Cancelled;
    request->session.reset();
    ++metrics.cancelled;
}

void CpuRequestLifecycle::fail(const std::shared_ptr<Request>& request,
                               const std::string& error,
                               ConcurrentMetrics& metrics) {
    if (!request || is_terminal(request->status)) return;
    if (request->cancel_requested) {
        cancel(request, metrics);
        return;
    }
    request->status = RequestStatus::Failed;
    request->error = error;
    request->session.reset();
    ++metrics.failed;
}

void CpuRequestLifecycle::observe_attention(
    const std::shared_ptr<Request>& request,
    CpuConcurrentMetricsExtras& extras) {
    if (!request || !request->session) return;
    const uint64_t current = request->session->diagnostics().attention_parallel_calls();
    if (current >= request->attention_parallel_observed) {
        extras.attention_parallel_calls +=
            current - request->attention_parallel_observed;
    }
    request->attention_parallel_observed = current;
}

void CpuRequestLifecycle::apply_decode_token(
    const std::shared_ptr<Request>& request,
    int32_t token,
    std::chrono::steady_clock::time_point now,
    ConcurrentMetrics& metrics,
    CpuConcurrentMetricsExtras& extras) {
    if (!request || request->status == RequestStatus::Cancelled) return;
    if (request->cancel_requested) {
        cancel(request, metrics);
        return;
    }
    request->generated.push_back(token);
    if (!request->first_token_recorded) {
        request->first_token_recorded = true;
        metrics.cumulative_ttft_ms +=
            std::chrono::duration<double, std::milli>(now - request->submitted_at).count();
        ++metrics.ttft_samples;
    } else {
        metrics.cumulative_itl_ms +=
            std::chrono::duration<double, std::milli>(now - request->last_token_at).count();
        ++metrics.itl_samples;
    }
    request->last_token_at = now;
    observe_attention(request, extras);
    if (is_stop_token(request->options.eos_tokens, token) ||
        request->generated.size() >=
            static_cast<size_t>(request->options.max_new_tokens)) {
        request->status = RequestStatus::Finished;
        request->session.reset();
        ++metrics.completed;
    }
}

void CpuRequestLifecycle::apply_prefill(
    const std::shared_ptr<Request>& request,
    size_t consumed_tokens,
    ConcurrentMetrics& metrics) {
    if (!request || is_terminal(request->status)) return;
    if (request->cancel_requested) {
        cancel(request, metrics);
        return;
    }
    request->prompt_offset += consumed_tokens;
    if (request->prompt_offset >= request->prompt.size()) {
        request->prompt_offset = request->prompt.size();
        request->status = RequestStatus::Decoding;
    }
}

} // namespace celeg

#include "engine_internal.hpp"

namespace celeg {
ConcurrentEngine::ConcurrentEngine(std::string model_path,
                                   int max_context,
                                   CudaModelOptions model_options,
                                   ConcurrentEngineOptions engine_options)
    : state_(std::make_unique<CudaSchedulerDriver>(std::move(model_path), max_context,
                                   model_options, engine_options)) {}

ConcurrentEngine::~ConcurrentEngine() = default;

ConcurrentEngine::RequestId ConcurrentEngine::submit(
    std::vector<int32_t> prompt, ConcurrentRequestOptions options) {
    return state_->submit(std::move(prompt), std::move(options));
}

bool ConcurrentEngine::cancel(RequestId id) { return state_->cancel(id); }

bool ConcurrentEngine::release(RequestId id) { return state_->release(id); }

PollResult ConcurrentEngine::poll(RequestId id, size_t max_tokens) {
    return state_->poll(id, max_tokens);
}

RequestStatus ConcurrentEngine::status(RequestId id) const {
    return state_->status(id);
}

bool ConcurrentEngine::step() { return state_->step(); }
void ConcurrentEngine::start() { state_->start(); }
void ConcurrentEngine::stop() { state_->stop(); }
bool ConcurrentEngine::running() const { return state_->running(); }
ConcurrentMetrics ConcurrentEngine::metrics() const { return state_->metrics(); }
GroupedConcurrentMetrics ConcurrentEngine::grouped_metrics() const {
    return state_->grouped_metrics();
}



} // namespace celeg

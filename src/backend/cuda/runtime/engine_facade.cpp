#include "engine_internal.hpp"

namespace lfm {
ConcurrentEngine::ConcurrentEngine(std::string model_path,
                                   int max_context,
                                   ModelOptions model_options,
                                   ConcurrentEngineOptions engine_options)
    : impl_(std::make_unique<Impl>(std::move(model_path), max_context,
                                   model_options, engine_options)) {}

ConcurrentEngine::~ConcurrentEngine() = default;

ConcurrentEngine::RequestId ConcurrentEngine::submit(
    std::vector<int32_t> prompt, ConcurrentRequestOptions options) {
    return impl_->submit(std::move(prompt), std::move(options));
}

bool ConcurrentEngine::cancel(RequestId id) { return impl_->cancel(id); }

bool ConcurrentEngine::release(RequestId id) { return impl_->release(id); }

PollResult ConcurrentEngine::poll(RequestId id, size_t max_tokens) {
    return impl_->poll(id, max_tokens);
}

RequestStatus ConcurrentEngine::status(RequestId id) const {
    return impl_->status(id);
}

bool ConcurrentEngine::step() { return impl_->step(); }
void ConcurrentEngine::start() { impl_->start(); }
void ConcurrentEngine::stop() { impl_->stop(); }
bool ConcurrentEngine::running() const { return impl_->running(); }
ConcurrentMetrics ConcurrentEngine::metrics() const { return impl_->metrics(); }
GroupedConcurrentMetrics ConcurrentEngine::grouped_metrics() const {
    return impl_->grouped_metrics();
}



} // namespace lfm

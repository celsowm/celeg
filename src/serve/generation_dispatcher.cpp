#include "lfm/serve/generation_dispatcher.hpp"

#include <utility>
#include <vector>

namespace lfm::serve {

GenerationDispatcher::GenerationDispatcher(IInferenceService& service,
                                           std::chrono::microseconds idle_interval)
    : service_(service), idle_interval_(idle_interval) {}

GenerationDispatcher::~GenerationDispatcher() { stop(); }

void GenerationDispatcher::start() {
    if (running_.exchange(true)) return;
    service_.start();
    thread_ = std::thread(&GenerationDispatcher::run, this);
}

void GenerationDispatcher::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    service_.stop();
}

void GenerationDispatcher::watch(RequestId id, EventCallback callback) {
    std::lock_guard<std::mutex> lock(watchers_mutex_);
    watchers_[id] = std::move(callback);
}

void GenerationDispatcher::unwatch(RequestId id) {
    std::lock_guard<std::mutex> lock(watchers_mutex_);
    watchers_.erase(id);
}

void GenerationDispatcher::run() {
    while (running_.load(std::memory_order_relaxed)) {
        dispatch_once();
    }
}

void GenerationDispatcher::dispatch_once() {
    const bool progressed = service_.step();

    std::vector<std::pair<RequestId, EventCallback>> active;
    {
        std::lock_guard<std::mutex> lock(watchers_mutex_);
        active.reserve(watchers_.size());
        for (const auto& entry : watchers_) active.emplace_back(entry.first, entry.second);
    }

    bool delivered = false;
    for (auto& entry : active) {
        const RequestId id = entry.first;
        GenerateEvent event = service_.poll(id, 0);
        if (event.tokens.empty() && !event.finished) continue;
        delivered = true;
        entry.second(event);
        if (event.finished) {
            std::lock_guard<std::mutex> lock(watchers_mutex_);
            watchers_.erase(id);
            service_.release(id);
        }
    }

    if (!progressed && !delivered) {
        std::this_thread::sleep_for(idle_interval_);
    }
}

} // namespace lfm::serve

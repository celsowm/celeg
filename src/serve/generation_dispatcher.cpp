#include "celeg/serve/generation_dispatcher.hpp"

#include <utility>
#include <vector>

namespace celeg::serve {

GenerationDispatcher::GenerationDispatcher(IRequestService& requests,
                                           ISchedulerDriver& scheduler,
                                           std::chrono::microseconds idle_interval)
    : requests_(requests), scheduler_(scheduler), idle_interval_(idle_interval) {}

GenerationDispatcher::~GenerationDispatcher() { stop(); }

void GenerationDispatcher::start() {
    if (running_.exchange(true)) return;
    scheduler_.start();
    thread_ = std::thread(&GenerationDispatcher::run, this);
}

void GenerationDispatcher::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    scheduler_.stop();
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
    const bool progressed = scheduler_.step();

    std::vector<RequestId> ids;
    {
        std::lock_guard<std::mutex> lock(watchers_mutex_);
        ids.reserve(watchers_.size());
        for (const auto& entry : watchers_) ids.push_back(entry.first);
    }

    bool delivered = false;
    for (RequestId id : ids) {
        GenerateEvent event = requests_.poll(id, 0);
        if (event.tokens.empty() && !event.finished) continue;
        delivered = true;

        // Hold watchers_mutex_ across the callback invocation itself (not just
        // the map lookup): this is the only thread that ever calls a watcher,
        // so unwatch() blocking on this same lock is what gives it its
        // guarantee -- once unwatch() returns, no in-flight call for that id
        // can still be running, and the erased entry can never be found again.
        std::lock_guard<std::mutex> lock(watchers_mutex_);
        auto it = watchers_.find(id);
        if (it == watchers_.end()) continue; // unwatched between snapshot and now
        it->second(event);
        if (event.finished) {
            watchers_.erase(it);
            requests_.release(id);
        }
    }

    if (!progressed && !delivered) {
        std::this_thread::sleep_for(idle_interval_);
    }
}

} // namespace celeg::serve

#include "celeg/serve/generation_dispatcher.hpp"

#include <utility>
#include <vector>
#include <iostream>

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

void GenerationDispatcher::cancel(RequestId id) {
    std::lock_guard<std::mutex> lock(watchers_mutex_);

    watchers_.erase(id);
    if (requests_.cancel(id)) {
        watchers_[id] = [](const GenerateEvent&) {};
    } else {
        requests_.release(id);
    }
}

void GenerationDispatcher::run() {
    while (running_.load(std::memory_order_relaxed)) {
        try {
            dispatch_once();
        } catch (const std::exception& error) {
            std::cerr << "generation dispatcher: " << error.what() << std::endl;
            running_.store(false, std::memory_order_relaxed);
        } catch (...) {
            std::cerr << "generation dispatcher: unknown exception" << std::endl;
            running_.store(false, std::memory_order_relaxed);
        }
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

        std::lock_guard<std::mutex> lock(watchers_mutex_);
        auto it = watchers_.find(id);
        if (it == watchers_.end()) continue;
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

}

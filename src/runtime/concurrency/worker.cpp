#include "lfm/detail/runtime/concurrency/worker.hpp"

#include <chrono>
#include <stdexcept>

namespace lfm::detail {

EngineWorker::~EngineWorker() { stop(); }

void EngineWorker::start(std::function<bool()> step,
                         int idle_sleep_microseconds) {
    if (!step) throw std::invalid_argument("worker step callback is empty");
    if (idle_sleep_microseconds < 0) {
        throw std::invalid_argument("idle sleep must not be negative");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_.joinable()) return;
    step_ = std::move(step);
    idle_sleep_microseconds_ = idle_sleep_microseconds;
    stopping_ = false;
    work_pending_ = true;
    thread_ = std::thread([this] { loop(); });
}

void EngineWorker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable()) return;
        stopping_ = true;
        cv_.notify_all();
    }
    thread_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    step_ = {};
}

void EngineWorker::notify() {
    std::lock_guard<std::mutex> lock(mutex_);
    work_pending_ = true;
    cv_.notify_all();
}

bool EngineWorker::running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return thread_.joinable() && !stopping_;
}

void EngineWorker::loop() {
    while (true) {
        std::function<bool()> step;
        int sleep_us = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stopping_) break;
            step = step_;
            sleep_us = idle_sleep_microseconds_;
        }
        const bool worked = step ? step() : false;
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopping_) break;
        if (!worked) {
            work_pending_ = false;
            cv_.wait_for(lock, std::chrono::microseconds(sleep_us),
                         [this] { return stopping_ || work_pending_; });
        }
    }
}

} // namespace lfm::detail

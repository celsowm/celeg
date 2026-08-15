#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace celeg::detail {

class EngineWorker {
public:
    EngineWorker() = default;
    ~EngineWorker();

    EngineWorker(const EngineWorker&) = delete;
    EngineWorker& operator=(const EngineWorker&) = delete;

    void start(std::function<bool()> step, int idle_sleep_microseconds);
    void stop();
    void notify();
    bool running() const;

private:
    void loop();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    std::function<bool()> step_;
    int idle_sleep_microseconds_ = 100;
    bool stopping_ = false;
    bool work_pending_ = false;
};

}

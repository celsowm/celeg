#pragma once

#include "celeg/serve/inference_service.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace celeg::serve {

class GenerationDispatcher {
public:
    using EventCallback = std::function<void(const GenerateEvent&)>;

    GenerationDispatcher(IRequestService& requests,
                          ISchedulerDriver& scheduler,
                                  std::chrono::microseconds idle_interval =
                                      std::chrono::milliseconds(5));
    ~GenerationDispatcher();

    GenerationDispatcher(const GenerationDispatcher&) = delete;
    GenerationDispatcher& operator=(const GenerationDispatcher&) = delete;

    void start();
    void stop();

    void watch(RequestId id, EventCallback callback);

    void unwatch(RequestId id);

    void cancel(RequestId id);

private:
    void run();
    void dispatch_once();

    IRequestService& requests_;
    ISchedulerDriver& scheduler_;
    std::chrono::microseconds idle_interval_;

    std::thread thread_;
    std::atomic<bool> running_{false};

    std::mutex watchers_mutex_;
    std::unordered_map<RequestId, EventCallback> watchers_;
};

}

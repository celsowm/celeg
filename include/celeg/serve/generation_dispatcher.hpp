#pragma once

#include "celeg/serve/inference_service.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace celeg::serve {

// Single background thread that drives IInferenceService::step() and fans
// out poll() results to watchers. Intended to be the only caller of step()
// and poll() for a given service; HTTP handlers register a callback via
// watch() and receive GenerateEvents until the request finishes, at which
// point the dispatcher releases the request automatically.
class GenerationDispatcher {
public:
    using EventCallback = std::function<void(const GenerateEvent&)>;

    explicit GenerationDispatcher(IInferenceService& service,
                                  std::chrono::microseconds idle_interval =
                                      std::chrono::milliseconds(5));
    ~GenerationDispatcher();

    GenerationDispatcher(const GenerationDispatcher&) = delete;
    GenerationDispatcher& operator=(const GenerationDispatcher&) = delete;

    void start();
    void stop();

    // Registers callback to receive GenerateEvents for id until it finishes
    // (the final, finished event is delivered before the request is
    // released). Replaces any existing watch for the same id.
    void watch(RequestId id, EventCallback callback);

    // Drops the watch without releasing the request.
    void unwatch(RequestId id);

private:
    void run();
    void dispatch_once();

    IInferenceService& service_;
    std::chrono::microseconds idle_interval_;

    std::thread thread_;
    std::atomic<bool> running_{false};

    std::mutex watchers_mutex_;
    std::unordered_map<RequestId, EventCallback> watchers_;
};

} // namespace celeg::serve

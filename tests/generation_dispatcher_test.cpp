#include "lfm/serve/generation_dispatcher.hpp"
#include "support/assertions.hpp"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using lfm::serve::FinishReason;
using lfm::serve::GenerateEvent;
using lfm::serve::GenerateRequest;
using lfm::serve::GenerationDispatcher;
using lfm::serve::IInferenceService;
using lfm::serve::ModelInfo;
using lfm::serve::RequestId;
using lfm::serve::RequestStatus;
using lfm::serve::ServingMetrics;

// Minimal in-memory stand-in for CpuInferenceService/CudaInferenceService:
// each step() emits one token per active request until max_output_tokens is
// reached, letting the dispatcher be exercised without a real model.
class FakeInferenceService final : public IInferenceService {
public:
    RequestId submit(GenerateRequest request) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const RequestId id = next_id_++;
        requests_[id] = State{request.max_output_tokens, {}, 0, false};
        return id;
    }

    GenerateEvent poll(RequestId id, std::size_t) override {
        std::lock_guard<std::mutex> lock(mutex_);
        GenerateEvent event;
        event.request_id = id;
        auto it = requests_.find(id);
        if (it == requests_.end()) {
            event.status = RequestStatus::Failed;
            return event;
        }
        event.tokens = std::move(it->second.buffered);
        it->second.buffered.clear();
        event.finished = it->second.finished;
        event.status = it->second.finished ? RequestStatus::Finished : RequestStatus::Decoding;
        if (event.finished) event.finish_reason = FinishReason::Stop;
        return event;
    }

    RequestStatus status(RequestId id) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = requests_.find(id);
        if (it == requests_.end()) return RequestStatus::Failed;
        return it->second.finished ? RequestStatus::Finished : RequestStatus::Decoding;
    }

    bool cancel(RequestId id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = requests_.find(id);
        if (it == requests_.end()) return false;
        it->second.finished = true;
        return true;
    }

    bool release(RequestId id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_.erase(id) > 0;
    }

    ModelInfo model_info() const override { return {}; }
    ServingMetrics metrics() const override { return {}; }

    bool step() override {
        std::lock_guard<std::mutex> lock(mutex_);
        bool progressed = false;
        for (auto& entry : requests_) {
            State& state = entry.second;
            if (state.finished || state.remaining == 0) continue;
            state.buffered.push_back(static_cast<std::int32_t>(state.emitted));
            state.emitted += 1;
            state.remaining -= 1;
            if (state.remaining == 0) state.finished = true;
            progressed = true;
        }
        return progressed;
    }

    void start() override { started_ = true; }
    void stop() override { started_ = false; }

    bool started() const { return started_; }

private:
    struct State {
        std::size_t remaining = 0;
        std::vector<std::int32_t> buffered;
        std::size_t emitted = 0;
        bool finished = false;
    };

    mutable std::mutex mutex_;
    std::unordered_map<RequestId, State> requests_;
    RequestId next_id_ = 1;
    bool started_ = false;
};

} // namespace

int main() {
    FakeInferenceService service;
    GenerationDispatcher dispatcher(service, std::chrono::milliseconds(1));

    dispatcher.start();
    LFM_TEST_CHECK(service.started());

    // A request the dispatcher should drive to completion and auto-release.
    GenerateRequest request;
    request.max_output_tokens = 5;
    const RequestId id = service.submit(request);

    std::mutex result_mutex;
    std::condition_variable result_cv;
    std::vector<std::int32_t> received;
    bool finished = false;

    dispatcher.watch(id, [&](const GenerateEvent& event) {
        std::lock_guard<std::mutex> lock(result_mutex);
        received.insert(received.end(), event.tokens.begin(), event.tokens.end());
        if (event.finished) {
            finished = true;
            result_cv.notify_all();
        }
    });

    {
        std::unique_lock<std::mutex> lock(result_mutex);
        const bool completed = result_cv.wait_for(
            lock, std::chrono::seconds(5), [&] { return finished; });
        LFM_TEST_CHECK(completed);
        LFM_TEST_CHECK(received.size() == 5);
        for (std::int32_t i = 0; i < 5; ++i) LFM_TEST_CHECK(received[i] == i);
    }

    // The callback is notified before the dispatcher performs its mandated
    // post-callback release, so wait for that final step rather than assuming
    // that notification and release are simultaneous.
    const auto release_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (service.status(id) != RequestStatus::Failed &&
           std::chrono::steady_clock::now() < release_deadline) {
        std::this_thread::yield();
    }

    // The dispatcher must have released the finished request on our behalf.
    LFM_TEST_CHECK(service.status(id) == RequestStatus::Failed);

    // unwatch() must stop delivery without releasing the request.
    GenerateRequest long_request;
    long_request.max_output_tokens = std::numeric_limits<std::size_t>::max();
    const RequestId long_id = service.submit(long_request);
    int callback_count = 0;
    dispatcher.watch(long_id, [&](const GenerateEvent&) { ++callback_count; });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    dispatcher.unwatch(long_id);
    const int count_at_unwatch = callback_count;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    LFM_TEST_CHECK(callback_count == count_at_unwatch);
    LFM_TEST_CHECK(service.status(long_id) == RequestStatus::Decoding);
    service.cancel(long_id);
    service.release(long_id);

    dispatcher.stop();
    LFM_TEST_CHECK(!service.started());

    std::cout << "generation_dispatcher_test: ok\n";
    return 0;
}

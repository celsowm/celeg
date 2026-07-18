#include "lfm/detail/engine_worker.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

int main() {
    lfm::detail::EngineWorker worker;
    std::atomic<int> calls{0};
    worker.start([&] {
        ++calls;
        return false;
    }, 1000);
    worker.notify();

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (calls.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    assert(calls.load() > 0);
    assert(worker.running());
    worker.stop();
    assert(!worker.running());
    std::cout << "engine_worker_test: ok\n";
}

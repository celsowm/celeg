#include "celeg/detail/runtime/concurrency/worker.hpp"
#include "support/assertions.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

int main() {
    celeg::detail::EngineWorker worker;
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
    CELEG_TEST_CHECK(calls.load() > 0);
    CELEG_TEST_CHECK(worker.running());
    worker.stop();
    CELEG_TEST_CHECK(!worker.running());
    std::cout << "engine_worker_test: ok\n";
}

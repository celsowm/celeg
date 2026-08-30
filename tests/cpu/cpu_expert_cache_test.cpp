#include "detail/expert_cache.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

celeg::CpuExpertWeights make_weights(std::size_t bytes) {
    celeg::Q4GroupMatrix w13;
    w13.rows = 1;
    w13.cols = static_cast<std::uint32_t>(bytes);
    w13.values.resize(bytes / 2);
    w13.scales_bf16.resize(bytes / 4);

    celeg::Q4GroupMatrix w2;
    w2.rows = 1;
    w2.cols = static_cast<std::uint32_t>(bytes);
    w2.values.resize(bytes / 2);
    w2.scales_bf16.resize(bytes / 4);

    celeg::CpuExpertWeights result;
    result.w13.rows = 1;
    result.w13.cols = static_cast<std::uint32_t>(bytes);
    result.w13.segments.emplace_back(std::move(w13));
    result.w2.rows = 1;
    result.w2.cols = static_cast<std::uint32_t>(bytes);
    result.w2.segments.emplace_back(std::move(w2));
    return result;
}

void frequency_protects_hot_expert() {
    const std::size_t expert_bytes = make_weights(64).memory_bytes();
    celeg::CpuExpertCache cache(expert_bytes * 2);
    std::atomic<int> loads{0};
    auto load = [&]() {
        ++loads;
        return make_weights(64);
    };

    cache.acquire(0, 1, load);
    cache.acquire(0, 1, load);
    check(cache.protected_resident(0, 1), "hot expert was not protected");

    cache.acquire(0, 2, load);
    cache.acquire(0, 3, load);

    check(cache.resident(0, 1), "protected expert was evicted");
    check(!cache.resident(0, 2), "cold probation expert survived eviction");
    check(cache.resident(0, 3), "new expert was not admitted");
    check(loads == 3, "unexpected expert loader count");
}

void concurrent_misses_coalesce() {
    celeg::CpuExpertCache cache(1U << 20);
    std::atomic<int> loads{0};
    auto load = [&]() {
        ++loads;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        return make_weights(64);
    };

    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<const celeg::CpuExpertWeights>> leases(4);
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&, i]() {
            leases[static_cast<std::size_t>(i)] = cache.acquire(2, 7, load);
        });
    }
    for (std::thread& thread : threads) thread.join();

    check(loads == 1, "concurrent misses performed duplicate loads");
    check(cache.misses() == 1, "true miss count is incorrect");
    check(cache.coalesced_waits() == 3, "coalesced waiter count is incorrect");
    for (const auto& lease : leases) {
        check(static_cast<bool>(lease), "waiter received an empty expert lease");
    }
}

}

int main() {
    try {
        frequency_protects_hot_expert();
        concurrent_misses_coalesce();
        std::cout << "cpu_expert_cache_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpu_expert_cache_test: " << error.what() << '\n';
        return 1;
    }
}

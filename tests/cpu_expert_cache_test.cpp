#include "detail/expert_cache.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {

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
    assert(cache.protected_resident(0, 1));

    cache.acquire(0, 2, load);
    cache.acquire(0, 3, load);

    assert(cache.resident(0, 1));
    assert(!cache.resident(0, 2));
    assert(cache.resident(0, 3));
    assert(loads == 3);
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

    assert(loads == 1);
    assert(cache.misses() == 1);
    assert(cache.coalesced_waits() == 3);
    for (const auto& lease : leases) assert(lease);
}

} // namespace

int main() {
    frequency_protects_hot_expert();
    concurrent_misses_coalesce();
    std::cout << "cpu_expert_cache_test: ok\n";
    return 0;
}

#include "lfm/pinned_expert_cache.hpp"

#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>

using namespace lfm;

void test_basic_hits_misses() {
    // 3 slots (each 10 bytes)
    PinnedExpertCache cache(30, 10, 5, 5);

    int load_count = 0;
    auto loader = [&](std::span<std::byte> gu, std::span<std::byte> dn) {
        load_count++;
        assert(gu.size() == 5);
        assert(dn.size() == 5);
        std::memset(gu.data(), 1, 5);
        std::memset(dn.data(), 2, 5);
    };

    // Miss 1
    ExpertHostLease lease1 = cache.acquire(0, 1, loader);
    assert(lease1.valid());
    assert(load_count == 1);
    assert(cache.misses() == 1);
    assert(cache.hits() == 0);

    // Hit 1
    ExpertHostLease lease2 = cache.acquire(0, 1, loader);
    assert(lease2.valid());
    assert(load_count == 1); // no new load
    assert(cache.misses() == 1);
    assert(cache.hits() == 1);

    // Miss 2
    ExpertHostLease lease3 = cache.acquire(0, 2, loader);
    assert(lease3.valid());
    assert(load_count == 2);
    assert(cache.misses() == 2);
    assert(cache.hits() == 1);
}

void test_eviction_and_lease_protection() {
    // 2 slots
    PinnedExpertCache cache(20, 10, 5, 5);

    auto loader = [](std::span<std::byte>, std::span<std::byte>) {};

    ExpertHostLease lease1 = cache.acquire(0, 1, loader);
    ExpertHostLease lease2 = cache.acquire(0, 2, loader);

    // Both slots leased, cache full. Acquire a 3rd expert should throw as no slot is evictable (ref_count > 0).
    bool threw = false;
    try {
        cache.acquire(0, 3, loader);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    // Release lease1
    lease1.release();

    // Now slot 1 can be evicted
    ExpertHostLease lease3 = cache.acquire(0, 3, loader);
    assert(lease3.valid());
    assert(cache.evictions() == 1);
}

void test_coalescing_and_concurrency() {
    PinnedExpertCache cache(20, 10, 5, 5);
    std::atomic<int> load_starts{0};
    std::atomic<int> load_completes{0};

    auto slow_loader = [&](std::span<std::byte>, std::span<std::byte>) {
        load_starts++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        load_completes++;
    };

    std::vector<std::thread> threads;
    std::vector<ExpertHostLease> leases(4);
    std::atomic<int> errors{0};

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&cache, &slow_loader, &leases, i, &errors]() {
            try {
                leases[i] = cache.acquire(0, 1, slow_loader);
            } catch (...) {
                errors++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    assert(errors == 0);
    assert(load_starts == 1);
    assert(load_completes == 1);
    assert(cache.misses() == 4); // 1 true miss, 3 coalesced misses

    for (int i = 0; i < 4; ++i) {
        assert(leases[i].valid());
    }
}

void test_loader_failure() {
    PinnedExpertCache cache(10, 10, 5, 5);
    auto failing_loader = [](std::span<std::byte>, std::span<std::byte>) {
        throw std::runtime_error("Disk read failed");
    };

    bool threw = false;
    try {
        cache.acquire(0, 1, failing_loader);
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()) == "Disk read failed") {
            threw = true;
        }
    }
    assert(threw);

    // Slot should be clean and reusable
    int load_count = 0;
    auto loader = [&](std::span<std::byte>, std::span<std::byte>) {
        load_count++;
    };
    ExpertHostLease lease = cache.acquire(0, 1, loader);
    assert(lease.valid());
    assert(load_count == 1);
}

int main() {
    test_basic_hits_misses();
    test_eviction_and_lease_protection();
    test_coalescing_and_concurrency();
    test_loader_failure();
    std::cout << "pinned_expert_cache_test: ok\n";
    return 0;
}

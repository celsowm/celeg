#include "lfm/runtime/cache/pinned_expert_cache.hpp"

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

void test_failure_propagation_to_waiters() {
    PinnedExpertCache cache(10, 10, 5, 5);
    std::atomic<int> waiter_exceptions{0};

    auto failing_loader = [](std::span<std::byte>, std::span<std::byte>) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        throw std::runtime_error("Loader crash");
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&cache, &failing_loader, &waiter_exceptions]() {
            try {
                cache.acquire(0, 1, failing_loader);
            } catch (const std::runtime_error& e) {
                if (std::string(e.what()) == "Loader crash") {
                    waiter_exceptions++;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All 4 threads (1 loader, 3 waiters) should catch the exception!
    assert(waiter_exceptions == 4);
}

void test_slot_protection_and_generation() {
    // 1 slot capacity
    PinnedExpertCache cache(10, 10, 5, 5);

    std::atomic<bool> loader_running{true};
    std::atomic<bool> waiter_running{false};

    auto slow_loader = [&](std::span<std::byte>, std::span<std::byte>) {
        while (loader_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    };

    // Thread 1: starts a slow load
    std::thread t1([&]() {
        auto lease = cache.acquire(0, 1, slow_loader);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Thread 2: waiter on the same expert (0, 1)
    std::thread t2([&]() {
        waiter_running = true;
        auto lease = cache.acquire(0, 1, [](std::span<std::byte>, std::span<std::byte>){});
        assert(lease.valid());
        assert(lease.expert() == 1);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(waiter_running);

    // Thread 3: tries to evict the slot for expert (0, 2)
    // Since waiter has reserved ref_count, this eviction must throw because no slot is free or evictable!
    bool threw = false;
    try {
        cache.acquire(0, 2, [](std::span<std::byte>, std::span<std::byte>){});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw); // Eviction is safely blocked!

    // Finish loader
    loader_running = false;
    t1.join();
    t2.join();
}

int main() {
    test_basic_hits_misses();
    test_eviction_and_lease_protection();
    test_coalescing_and_concurrency();
    test_loader_failure();
    test_failure_propagation_to_waiters();
    test_slot_protection_and_generation();
    std::cout << "pinned_expert_cache_test: ok\n";
    return 0;
}

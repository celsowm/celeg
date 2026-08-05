#include "celeg/runtime/cache/pinned_expert_cache.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>

using celeg::ExpertHostLease;
using celeg::HostExpertCache;

void test_basic_hits_misses() {
    // 3 slots (each 10 bytes)
    HostExpertCache cache(30, 10);

    int load_count = 0;
    auto loader = [&](std::span<std::byte> payload) {
        load_count++;
        CELEG_TEST_CHECK(payload.size() == 10);
        std::memset(payload.data(), 1, payload.size());
    };

    // Miss 1
    ExpertHostLease lease1 = cache.acquire(0, 1, loader);
    CELEG_TEST_CHECK(lease1.valid());
    CELEG_TEST_CHECK(load_count == 1);
    CELEG_TEST_CHECK(cache.misses() == 1);
    CELEG_TEST_CHECK(cache.hits() == 0);

    // Hit 1
    ExpertHostLease lease2 = cache.acquire(0, 1, loader);
    CELEG_TEST_CHECK(lease2.valid());
    CELEG_TEST_CHECK(load_count == 1); // no new load
    CELEG_TEST_CHECK(cache.misses() == 1);
    CELEG_TEST_CHECK(cache.hits() == 1);

    // Miss 2
    ExpertHostLease lease3 = cache.acquire(0, 2, loader);
    CELEG_TEST_CHECK(lease3.valid());
    CELEG_TEST_CHECK(load_count == 2);
    CELEG_TEST_CHECK(cache.misses() == 2);
    CELEG_TEST_CHECK(cache.hits() == 1);
}

void test_eviction_and_lease_protection() {
    // 2 slots
    HostExpertCache cache(20, 10);

    auto loader = [](std::span<std::byte>) {};

    ExpertHostLease lease1 = cache.acquire(0, 1, loader);
    ExpertHostLease lease2 = cache.acquire(0, 2, loader);

    // Both slots leased, cache full. Acquire a 3rd expert should throw as no slot is evictable (ref_count > 0).
    bool threw = false;
    try {
        cache.acquire(0, 3, loader);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CELEG_TEST_CHECK(threw);

    // Release lease1
    lease1.release();

    // Now slot 1 can be evicted
    ExpertHostLease lease3 = cache.acquire(0, 3, loader);
    CELEG_TEST_CHECK(lease3.valid());
    CELEG_TEST_CHECK(cache.evictions() == 1);
}

void test_frequency_aware_eviction() {
    // Two slots: expert 1 becomes hot, expert 2 is only touched once. Even
    // though expert 2 is more recent, admitting expert 3 must evict expert 2.
    HostExpertCache cache(20, 10);
    int load_count = 0;
    auto loader = [&](std::span<std::byte>) {
        load_count++;
    };

    for (int i = 0; i < 6; ++i) {
        auto lease = cache.acquire(0, 1, loader);
    }
    {
        auto lease = cache.acquire(0, 2, loader);
    }
    {
        auto lease = cache.acquire(0, 3, loader);
    }

    const int after_three_experts = load_count;
    {
        auto lease = cache.acquire(0, 1, loader);
    }
    CELEG_TEST_CHECK(load_count == after_three_experts); // hot expert survived

    {
        auto lease = cache.acquire(0, 2, loader);
    }
    CELEG_TEST_CHECK(load_count == after_three_experts + 1); // cold expert was evicted
}

void test_coalescing_and_concurrency() {
    HostExpertCache cache(20, 10);
    std::atomic<int> load_starts{0};
    std::atomic<int> load_completes{0};

    auto slow_loader = [&](std::span<std::byte>) {
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
            } catch (const std::exception& error) {
                std::cerr << "worker error: " << error.what() << '\n';
                errors++;
            } catch (...) {
                std::cerr << "worker error: unknown exception\n";
                errors++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    CELEG_TEST_CHECK(errors == 0);
    CELEG_TEST_CHECK(load_starts == 1);
    CELEG_TEST_CHECK(load_completes == 1);
    CELEG_TEST_CHECK(cache.misses() == 1);
    CELEG_TEST_CHECK(cache.coalesced_waits() == 3);

    for (int i = 0; i < 4; ++i) {
        CELEG_TEST_CHECK(leases[i].valid());
    }
}

void test_loader_failure() {
    HostExpertCache cache(10, 10);
    auto failing_loader = [](std::span<std::byte>) {
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
    CELEG_TEST_CHECK(threw);

    // Slot should be clean and reusable
    int load_count = 0;
    auto loader = [&](std::span<std::byte>) {
        load_count++;
    };
    ExpertHostLease lease = cache.acquire(0, 1, loader);
    CELEG_TEST_CHECK(lease.valid());
    CELEG_TEST_CHECK(load_count == 1);
}

void test_failure_propagation_to_waiters() {
    HostExpertCache cache(10, 10);
    std::atomic<int> waiter_exceptions{0};

    auto failing_loader = [](std::span<std::byte>) {
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
    CELEG_TEST_CHECK(waiter_exceptions == 4);
}

void test_slot_protection_and_generation() {
    // 1 slot capacity
    HostExpertCache cache(10, 10);

    std::atomic<bool> loader_started{false};
    std::atomic<bool> loader_running{true};
    std::atomic<bool> waiter_running{false};
    std::atomic<int> waiter_errors{0};

    auto slow_loader = [&](std::span<std::byte>) {
        loader_started.store(true, std::memory_order_release);
        while (loader_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    };

    // Thread 1: starts a slow load
    std::thread t1([&]() {
        try {
            auto lease = cache.acquire(0, 1, slow_loader);
        } catch (const std::exception& error) {
            std::cerr << "slot loader error: " << error.what() << '\n';
            waiter_errors.fetch_add(1);
        } catch (...) {
            std::cerr << "slot loader error: unknown exception\n";
            waiter_errors.fetch_add(1);
        }
    });

    while (!loader_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Thread 2: waiter on the same expert (0, 1)
    std::thread t2([&]() {
        try {
            waiter_running = true;
            auto lease = cache.acquire(0, 1, [](std::span<std::byte>){});
            CELEG_TEST_CHECK(lease.valid());
            CELEG_TEST_CHECK(lease.expert() == 1);
        } catch (const std::exception& error) {
            std::cerr << "slot waiter error: " << error.what() << '\n';
            waiter_errors.fetch_add(1);
        } catch (...) {
            std::cerr << "slot waiter error: unknown exception\n";
            waiter_errors.fetch_add(1);
        }
    });

    while (!waiter_running.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    CELEG_TEST_CHECK(waiter_running);

    // Thread 3: tries to evict the slot for expert (0, 2)
    // Since waiter has reserved ref_count, this eviction must throw because no slot is free or evictable!
    bool threw = false;
    try {
        cache.acquire(0, 2, [](std::span<std::byte>){});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CELEG_TEST_CHECK(threw); // Eviction is safely blocked!

    // Finish loader
    loader_running = false;
    t1.join();
    t2.join();
    CELEG_TEST_CHECK(waiter_errors.load() == 0);
}

int main(int argc, char** argv) {
    try {
        const std::string selected = argc > 1 ? argv[1] : "all";
        if (selected == "basic" || selected == "all") test_basic_hits_misses();
        if (selected == "eviction" || selected == "all") test_eviction_and_lease_protection();
        if (selected == "frequency" || selected == "all") test_frequency_aware_eviction();
        if (selected == "coalescing" || selected == "all") test_coalescing_and_concurrency();
        if (selected == "failure" || selected == "all") test_loader_failure();
        if (selected == "propagation" || selected == "all") test_failure_propagation_to_waiters();
        if (selected == "generation" || selected == "all") test_slot_protection_and_generation();
        std::cout << "pinned_expert_cache_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pinned_expert_cache_test: " << error.what() << '\n';
        return 1;
    }
}

#include "lfm/cpu_thread_pool.hpp"
#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    lfm::CpuThreadPool pool(4, lfm::CpuAffinityPolicy::Compact);
    std::vector<int> values(10000, 0);
    pool.parallel_for(0, values.size(), 37, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) values[i] = static_cast<int>(i + 1);
    });
    for (size_t i = 0; i < values.size(); ++i) assert(values[i] == static_cast<int>(i + 1));
    std::atomic<size_t> sum{0};
    pool.parallel_for(0, 1000, 11, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) sum.fetch_add(i, std::memory_order_relaxed);
    });
    assert(sum.load() == 999 * 1000 / 2);
    std::cout << "cpu_thread_pool_test: pinned=" << pool.pinned_workers() << "\n";
}

#pragma once

#include "lfm/backend/cpu/topology.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace lfm {

class CpuThreadPool {
public:
    explicit CpuThreadPool(size_t threads = 0,
                           CpuAffinityPolicy affinity = CpuAffinityPolicy::None);
    ~CpuThreadPool();

    CpuThreadPool(const CpuThreadPool&) = delete;
    CpuThreadPool& operator=(const CpuThreadPool&) = delete;

    size_t size() const { return workers_.size(); }
    CpuAffinityPolicy affinity() const { return affinity_; }
    size_t pinned_workers() const { return pinned_workers_.load(); }

    void parallel_for(size_t begin, size_t end, size_t grain,
                      const std::function<void(size_t, size_t)>& task);

private:
    void worker_loop(size_t worker_index);

    std::vector<std::thread> workers_;
    CpuAffinityPolicy affinity_ = CpuAffinityPolicy::None;
    std::vector<int> worker_cpu_ids_;
    std::atomic<size_t> pinned_workers_{0};
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    bool stopping_ = false;
    uint64_t generation_ = 0;
    size_t begin_ = 0;
    size_t end_ = 0;
    size_t grain_ = 1;
    std::function<void(size_t, size_t)> task_;
    std::atomic<size_t> next_{0};
    size_t pending_workers_ = 0;
};

} // namespace lfm

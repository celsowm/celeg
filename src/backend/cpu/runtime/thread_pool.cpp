#include "lfm/backend/cpu/thread_pool.hpp"

#include <algorithm>
#include <stdexcept>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace lfm {

CpuThreadPool::CpuThreadPool(size_t threads, CpuAffinityPolicy affinity)
    : affinity_(affinity) {
    if (threads == 0) {
        threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    }
    // The calling thread also participates, so keep one logical core for it.
    const size_t worker_count = threads > 1 ? threads - 1 : 0;
    worker_cpu_ids_ = detect_cpu_topology().worker_cpu_order(affinity_, worker_count);
    workers_.reserve(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

CpuThreadPool::~CpuThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        ++generation_;
    }
    work_cv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void CpuThreadPool::parallel_for(
    size_t begin, size_t end, size_t grain,
    const std::function<void(size_t, size_t)>& task) {
    if (!task) throw std::invalid_argument("parallel_for task is empty");
    if (end <= begin) return;
    grain = std::max<size_t>(1, grain);
    if (workers_.empty() || end - begin <= grain) {
        task(begin, end);
        return;
    }

    uint64_t target_generation = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        begin_ = begin;
        end_ = end;
        grain_ = grain;
        task_ = task;
        next_.store(begin, std::memory_order_relaxed);
        pending_workers_ = workers_.size();
        target_generation = ++generation_;
    }
    (void)target_generation;
    work_cv_.notify_all();

    while (true) {
        const size_t chunk_begin = next_.fetch_add(grain, std::memory_order_relaxed);
        if (chunk_begin >= end) break;
        task(chunk_begin, std::min(end, chunk_begin + grain));
    }

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return pending_workers_ == 0; });
    task_ = {};
}

void CpuThreadPool::worker_loop(size_t worker_index) {
#if defined(__linux__)
    if (worker_index < worker_cpu_ids_.size()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(worker_cpu_ids_[worker_index], &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0) {
            pinned_workers_.fetch_add(1, std::memory_order_relaxed);
        }
    }
#else
    (void)worker_index;
#endif
    uint64_t observed_generation = 0;
    for (;;) {
        std::function<void(size_t, size_t)> task;
        size_t end = 0;
        size_t grain = 1;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [&] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) return;
            observed_generation = generation_;
            task = task_;
            end = end_;
            grain = grain_;
        }

        while (true) {
            const size_t chunk_begin =
                next_.fetch_add(grain, std::memory_order_relaxed);
            if (chunk_begin >= end) break;
            task(chunk_begin, std::min(end, chunk_begin + grain));
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_workers_ == 0) {
                std::terminate();
            }
            --pending_workers_;
            if (pending_workers_ == 0) done_cv_.notify_one();
        }
    }
}

} // namespace lfm

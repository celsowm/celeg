#include "engine_internal.hpp"

namespace celeg {
bool CudaSchedulerDriver::step() {
    std::lock_guard<std::mutex> step_lock(step_mutex_);
    const auto started = std::chrono::steady_clock::now();
    bool did_work = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        did_work |= admit_requests_locked();
    }
    did_work |= run_prefill_work();
    did_work |= run_decode_work();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        did_work |= admit_requests_locked();
        ++metrics_.scheduler_steps;
        const auto ended = std::chrono::steady_clock::now();
        metrics_.cumulative_step_ms +=
            std::chrono::duration<double, std::milli>(ended - started).count();
    }
    return did_work;
}

void CudaSchedulerDriver::start() {
    if (!engine_options_.worker_thread) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
    }
    worker_.start([this] { return step(); },
                  engine_options_.idle_sleep_microseconds);
}

void CudaSchedulerDriver::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    worker_.stop();
}

bool CudaSchedulerDriver::running() const {
    return worker_.running();
}

ConcurrentMetrics CudaSchedulerDriver::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ConcurrentMetrics snapshot = metrics_;
    if (prefix_cache_) {
        const PrefixCacheMetrics& cache = prefix_cache_->metrics();
        snapshot.prefix_cache_hits = cache.hits;
        snapshot.prefix_cache_misses = cache.misses;
        snapshot.prefix_cache_inserts = cache.inserts;
        snapshot.prefix_cache_evictions = cache.evictions;
        snapshot.prefix_cache_partial_hits = cache.partial_hits;
        snapshot.prefix_reused_tokens = cache.reused_tokens;
        snapshot.prefix_cow_pages = cache.cow_pages;
        snapshot.prefix_radix_lookups = cache.radix_lookups;
        snapshot.prefix_radix_nodes = prefix_cache_->radix_nodes();
        snapshot.prefix_cow_bytes_copied = cache.cow_bytes_copied;
        snapshot.prefix_cow_bytes_saved = cache.cow_bytes_saved;
    }
    snapshot.logical_pages_used = paged_kv_ ? paged_kv_->used_pages() : 0;
    return snapshot;
}

GroupedConcurrentMetrics CudaSchedulerDriver::grouped_metrics() const {
    return group_concurrent_metrics(metrics());
}




} // namespace celeg

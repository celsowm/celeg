bool ConcurrentEngine::Impl::step() {
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

void ConcurrentEngine::Impl::start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
    }
    worker_.start([this] { return step(); },
                  engine_options_.idle_sleep_microseconds);
}

void ConcurrentEngine::Impl::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    worker_.stop();
}

bool ConcurrentEngine::Impl::running() const {
    return worker_.running();
}

ConcurrentMetrics ConcurrentEngine::Impl::metrics() const {
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

GroupedConcurrentMetrics ConcurrentEngine::Impl::grouped_metrics() const {
    return group_concurrent_metrics(metrics());
}


ConcurrentEngine::ConcurrentEngine(std::string safetensors_path,
                                   int max_context,
                                   ModelOptions model_options,
                                   ConcurrentEngineOptions engine_options)
    : impl_(std::make_unique<Impl>(std::move(safetensors_path), max_context,
                                   model_options, engine_options)) {}

ConcurrentEngine::~ConcurrentEngine() = default;

ConcurrentEngine::RequestId ConcurrentEngine::submit(
    std::vector<int32_t> prompt, ConcurrentRequestOptions options) {
    return impl_->submit(std::move(prompt), std::move(options));
}

bool ConcurrentEngine::cancel(RequestId id) { return impl_->cancel(id); }

PollResult ConcurrentEngine::poll(RequestId id, size_t max_tokens) {
    return impl_->poll(id, max_tokens);
}

RequestStatus ConcurrentEngine::status(RequestId id) const {
    return impl_->status(id);
}

bool ConcurrentEngine::step() { return impl_->step(); }
void ConcurrentEngine::start() { impl_->start(); }
void ConcurrentEngine::stop() { impl_->stop(); }
bool ConcurrentEngine::running() const { return impl_->running(); }
ConcurrentMetrics ConcurrentEngine::metrics() const { return impl_->metrics(); }
GroupedConcurrentMetrics ConcurrentEngine::grouped_metrics() const {
    return impl_->grouped_metrics();
}


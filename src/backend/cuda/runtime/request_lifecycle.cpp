#include "engine_internal.hpp"

namespace celeg {
CudaSchedulerDriver::~CudaSchedulerDriver() {
    stop();
    std::lock_guard<std::mutex> lock(mutex_);
    if (prefix_cache_) prefix_cache_->clear();
}

ConcurrentEngine::RequestId CudaSchedulerDriver::submit(
    std::vector<int32_t> prompt, ConcurrentRequestOptions options) {
    if (prompt.empty()) throw std::invalid_argument("request prompt is empty");
    if (prompt.size() > static_cast<size_t>(max_context_))
        throw std::invalid_argument("request prompt exceeds max_context");
    if (options.max_new_tokens <= 0)
        throw std::invalid_argument("max_new_tokens must be positive");
    if (prompt.size() + static_cast<size_t>(options.max_new_tokens) >
        static_cast<size_t>(max_context_)) {
        throw std::invalid_argument("prompt plus max_new_tokens exceeds max_context");
    }
    options.generation.validate();

    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) throw std::runtime_error("engine is stopping");
    const RequestId id = registry_.create(
        std::move(prompt), std::move(options),
        std::chrono::steady_clock::now());
    ++metrics_.submitted;
    metrics_.queued_requests = registry_.queued_count();
    worker_.notify();
    return id;
}

bool CudaSchedulerDriver::cancel(RequestId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    Request* found = registry_.find(id);
    if (!found) return false;
    Request& request = *found;
    if (request.status == RequestStatus::Finished ||
        request.status == RequestStatus::Cancelled ||
        request.status == RequestStatus::Failed) return false;
    request.cancel_requested = true;
    worker_.notify();
    return true;
}

bool CudaSchedulerDriver::release(RequestId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    Request* found = registry_.find(id);
    if (!found) return false;
    if (found->status != RequestStatus::Finished &&
        found->status != RequestStatus::Cancelled &&
        found->status != RequestStatus::Failed) return false;
    return registry_.erase(id);
}

PollResult CudaSchedulerDriver::poll(RequestId id, size_t max_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    Request& request = registry_.at(id);
    PollResult result;
    result.status = request.status;
    result.error = request.error;
    const size_t count = max_tokens == 0
        ? request.output.size()
        : std::min(max_tokens, request.output.size());
    result.tokens.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.tokens.push_back(request.output.front());
        request.output.pop_front();
    }
    result.finished = is_terminal(request.status) && request.output.empty();
    return result;
}

RequestStatus CudaSchedulerDriver::status(RequestId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registry_.at(id).status;
}

CudaSchedulerDriver::Lane* CudaSchedulerDriver::find_free_lane_locked() {
    for (auto& lane : lanes_) if (lane->request_id == 0) return lane.get();
    return nullptr;
}

std::vector<detail::LaneSnapshot>
CudaSchedulerDriver::lane_snapshots_locked() const {
    std::vector<detail::LaneSnapshot> result;
    result.reserve(lanes_.size());
    for (const auto& lane : lanes_) {
        detail::LaneSnapshot snapshot;
        snapshot.lane_index = lane->index;
        snapshot.request_id = lane->request_id;
        if (lane->request_id != 0) {
            const Request& request = registry_.at(lane->request_id);
            snapshot.status = request.status;
            snapshot.priority = request.options.priority;
        }
        result.push_back(snapshot);
    }
    return result;
}

bool CudaSchedulerDriver::admit_requests_locked() {
    bool did_work = false;
    while (!registry_.empty_queue()) {
        Lane* lane = find_free_lane_locked();
        if (!lane) break;
        const auto selected = planner_.next_admission(registry_);
        if (!selected) break;
        const RequestId id = *selected;
        Request& request = registry_.at(id);
        if (request.cancel_requested) {
            registry_.erase_queued(id);
            finish_request_locked(request, RequestStatus::Cancelled);
            did_work = true;
            continue;
        }

        const size_t reserved_tokens =
            engine_options_.scheduler_policy == SchedulerPolicy::GuaranteedNoEvict
                ? request.prompt.size() + static_cast<size_t>(request.options.max_new_tokens)
                : request.prompt.size() + static_cast<size_t>(engine_options_.page_tokens);
        const bool forced_lane = request.options.generation.forced_prefix != nullptr;
        PrefixAcquireResult prefix;
        if (!forced_lane && request.options.prompt_embedding.empty() && prefix_cache_) {
            prefix = prefix_cache_->acquire(request.prompt, reserved_tokens);
        } else {
            prefix.status = PrefixAcquireStatus::Miss;
        }
        if (prefix.status == PrefixAcquireStatus::OutOfMemory) break;

        const bool prefix_hit = prefix.status == PrefixAcquireStatus::Hit;
        std::vector<uint32_t> pages;
        if (prefix_hit) {
            pages = std::move(prefix.pages);
        } else if (prefix_cache_) {
            auto allocated = prefix_cache_->allocate_request_pages(reserved_tokens);
            if (!allocated) break;
            pages = std::move(*allocated);
        }

        try {
            if (!lane->model) {
                CudaModelOptions lane_options = model_options_;
                lane_options.allocate_local_kv_cache =
                    !engine_options_.packed_decode || forced_lane;
                lane->model = std::make_unique<CudaModel>(
                    model_path_, max_context_, lane_options,
                    request.options.generation, runtime_, 0, weight_cache_);
            } else {
                lane->model->session().set_generation_config(request.options.generation);
            }
        } catch (const std::exception& error) {
            if (paged_kv_) paged_kv_->release(pages);
            registry_.erase_queued(id);
            finish_request_locked(request, RequestStatus::Failed, error.what());
            did_work = true;
            continue;
        }

        registry_.erase_queued(id);
        request.pages = std::move(pages);
        request.lane_index = lane->index;
        lane->request_id = id;
        try {
            if (!forced_lane && engine_options_.packed_decode) {
                lane->model->session().release_local_kv_cache();
            }
            lane->model->session().reset(forced_lane);
            if (prefix_hit) {
                if (!prefix.state) {
                    throw std::runtime_error("prefix cache hit has no session state");
                }
                lane->model->persistence().restore_prefix_state(*prefix.state);
                request.prefill_offset = prefix.matched_tokens;
                request.status = request.prefill_offset == request.prompt.size()
                    ? RequestStatus::Decoding : RequestStatus::Prefill;
                request.paged_ready = true;
            } else {
                request.status = RequestStatus::Prefill;
                request.paged_ready = !forced_lane && engine_options_.packed_decode &&
                    packed_executor_ != nullptr;
            }
        } catch (const std::exception& error) {
            finish_request_locked(request, RequestStatus::Failed, error.what());
        }
        did_work = true;
    }
    metrics_.queued_requests = registry_.queued_count();
    metrics_.active_requests = 0;
    for (const auto& lane : lanes_) {
        if (lane->request_id != 0) ++metrics_.active_requests;
    }
    metrics_.logical_pages_used = paged_kv_ ? paged_kv_->used_pages() : 0;
    return did_work;
}

void CudaSchedulerDriver::finish_request_locked(Request& request,
                                             RequestStatus status_value,
                                             std::string error) {
    if (request.lane_index >= 0) {
        Lane& lane = *lanes_.at(static_cast<size_t>(request.lane_index));
        lane.request_id = 0;
        request.lane_index = -1;
    }
    if (!request.pages.empty()) {
        if (paged_kv_) paged_kv_->release(request.pages);
        request.pages.clear();
    }
    request.status = status_value;
    request.error = std::move(error);
    if (status_value == RequestStatus::Finished) ++metrics_.completed;
    if (status_value == RequestStatus::Cancelled) ++metrics_.cancelled;
    if (status_value == RequestStatus::Failed) ++metrics_.failed;
    metrics_.logical_pages_used = paged_kv_ ? paged_kv_->used_pages() : 0;
}

void CudaSchedulerDriver::complete_prefill_locked(Request& request, Lane& lane) {
    request.paged_ready = engine_options_.packed_decode && packed_executor_ != nullptr;
    if (packed_executor_ && prefix_cache_->enabled() &&
        request.options.prompt_embedding.empty()) {
        PrefixState state = lane.model->persistence().export_prefix_state();
        (void)prefix_cache_->insert_or_update(
            request.prompt, request.pages, std::move(state));
    }
    request.status = RequestStatus::Decoding;
}

}

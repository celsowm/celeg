#include "engine_internal.hpp"
// Keep the decode scheduler bound to the concrete CudaModel contract.

namespace celeg {
bool CudaSchedulerDriver::run_decode_work() {
    struct Work { Lane* lane; RequestId id; bool paged_ready; };
    std::vector<Work> work;
    // Phase 1: build work list under a single lock.  Copy paged_ready so
    // the classify phase (phase 3) never needs to re-acquire the lock.
    std::vector<RequestId> page_needed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        int decode_budget = engine_options_.max_batched_tokens;
        const std::vector<int> ordered =
            planner_.order_decode(lane_snapshots_locked());
        for (const int lane_index : ordered) {
            if (decode_budget <= 0) break;
            Lane& lane = *lanes_.at(static_cast<size_t>(lane_index));
            if (lane.request_id == 0) continue;
            Request& request = registry_.at(lane.request_id);
            if (request.status != RequestStatus::Decoding) continue;
            if (request.cancel_requested) {
                finish_request_locked(request, RequestStatus::Cancelled);
                continue;
            }
            if (engine_options_.scheduler_policy ==
                SchedulerPolicy::MaxUtilization) {
                const size_t next_tokens = request.prompt.size() +
                    static_cast<size_t>(request.generated + 1);
                const size_t capacity = request.pages.size() *
                    static_cast<size_t>(engine_options_.page_tokens);
                if (next_tokens > capacity) {
                    page_needed.push_back(request.id);
                }
            }
            work.push_back({&lane, request.id, request.paged_ready});
            --decode_budget;
        }
    }
    // Phase 1b: allocate pages outside the lock, then commit.
    if (!page_needed.empty()) {
        std::vector<std::optional<std::vector<uint32_t>>> allocations;
        std::vector<RequestId> page_failed;
        allocations.reserve(page_needed.size());
        for (size_t i = 0; i < page_needed.size(); ++i) {
            allocations.push_back(prefix_cache_->allocate_request_pages(1));
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < page_needed.size(); ++i) {
            if (allocations[i]) {
                Request& request = registry_.at(page_needed[i]);
                request.pages.insert(request.pages.end(),
                                     allocations[i]->begin(),
                                     allocations[i]->end());
            } else {
                page_failed.push_back(page_needed[i]);
            }
        }
        metrics_.logical_pages_used = paged_kv_->used_pages();
        if (!page_failed.empty()) {
            work.erase(std::remove_if(work.begin(), work.end(),
                [&page_failed](const Work& item) {
                    return std::find(page_failed.begin(), page_failed.end(),
                                     item.id) != page_failed.end();
                }), work.end());
        }
    }

    auto accept_token = [&](const Work& item, int32_t token,
                            bool packed_path) {
        std::lock_guard<std::mutex> lock(mutex_);
        Request& request = registry_.at(item.id);
        if (request.cancel_requested) {
            finish_request_locked(request, RequestStatus::Cancelled);
            return;
        }
        if (request.status != RequestStatus::Decoding) return;
        request.output.push_back(token);
        const auto token_time = std::chrono::steady_clock::now();
        if (request.generated == 0) {
            metrics_.cumulative_ttft_ms +=
                std::chrono::duration<double, std::milli>(
                    token_time - request.submitted_at).count();
            ++metrics_.ttft_samples;
        } else {
            metrics_.cumulative_itl_ms +=
                std::chrono::duration<double, std::milli>(
                    token_time - request.last_token_at).count();
            ++metrics_.itl_samples;
        }
        request.last_token_at = token_time;
        ++request.generated;
        ++metrics_.decoded_tokens;
        if (packed_path) ++metrics_.packed_decode_tokens;
        else ++metrics_.lane_decode_tokens;
        if (is_stop_token(request.options.eos_tokens, token) ||
            request.generated >= request.options.max_new_tokens) {
            finish_request_locked(request, RequestStatus::Finished);
        }
    };

    std::vector<Work> packed_work;
    std::vector<Work> lane_work;
    packed_work.reserve(work.size());
    lane_work.reserve(work.size());
    for (const Work& item : work) {
        std::string reason;
        if (item.paged_ready && packed_executor_ &&
            packed_executor_->eligible(packed_session_context(*item.lane->model), &reason)) {
            packed_work.push_back(item);
        } else {
            lane_work.push_back(item);
        }
    }
    if (packed_work.size() <
        static_cast<size_t>(engine_options_.packed_min_batch)) {
        bool can_fallback = true;
        for (const Work& item : packed_work) {
            if (!item.lane->model->session().local_kv_cache_available()) {
                can_fallback = false;
                break;
            }
        }
        if (can_fallback) {
            lane_work.insert(lane_work.end(), packed_work.begin(), packed_work.end());
            packed_work.clear();
        }
    }

    if (!packed_work.empty()) {
        std::vector<PackedSessionContext> models;
        std::vector<std::vector<uint32_t>> page_tables;
        models.reserve(packed_work.size());
        page_tables.reserve(packed_work.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const Work& item : packed_work) {
                models.push_back(packed_session_context(*item.lane->model));
                page_tables.push_back(registry_.at(item.id).pages);
            }
        }
        try {
            const PackedDecodeMetrics before = packed_executor_->metrics();
            const std::vector<int32_t> tokens =
                packed_executor_->decode(models, page_tables);
            const PackedDecodeMetrics after = packed_executor_->metrics();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                metrics_.packed_decode_steps += after.steps - before.steps;
                metrics_.cumulative_packed_decode_ms +=
                    after.cumulative_ms - before.cumulative_ms;
                metrics_.segmented_paged_decode_steps +=
                    after.segmented_paged_steps - before.segmented_paged_steps;
                metrics_.segmented_paged_decode_tokens +=
                    after.segmented_paged_tokens - before.segmented_paged_tokens;
                metrics_.maximum_packed_batch = std::max<uint64_t>(
                    metrics_.maximum_packed_batch,
                    static_cast<uint64_t>(packed_work.size()));
            }
            for (size_t i = 0; i < packed_work.size(); ++i) {
                accept_token(packed_work[i], tokens[i], true);
            }
        } catch (const std::invalid_argument& error) {
            // A paged request normally has no local KV after prefill import.
            // Only fall back when every row still owns a valid contiguous cache;
            // otherwise surface the page-table/packed error instead of invoking
            // the lane path with null cache pointers.
            bool can_fallback = true;
            for (const Work& item : packed_work) {
                if (!item.lane->model->session().local_kv_cache_available()) {
                    can_fallback = false;
                    break;
                }
            }
            if (can_fallback) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++metrics_.packed_fallback_batches;
                }
                lane_work.insert(lane_work.end(), packed_work.begin(), packed_work.end());
            } else {
                for (const Work& item : packed_work) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    Request& request = registry_.at(item.id);
                    finish_request_locked(request, RequestStatus::Failed,
                                          error.what());
                }
            }
        } catch (const std::exception& error) {
            for (const Work& item : packed_work) {
                std::lock_guard<std::mutex> lock(mutex_);
                Request& request = registry_.at(item.id);
                finish_request_locked(request, RequestStatus::Failed,
                                      error.what());
            }
        }
    }

    // Requests not eligible for the packed kernel continue through the v0.0.9
    // multi-stream lane path. All streams are enqueued before any is joined.
    std::vector<Work> launched;
    launched.reserve(lane_work.size());
    for (const Work& item : lane_work) {
        try {
            item.lane->model->session().decode_async_begin();
            launched.push_back(item);
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            Request& request = registry_.at(item.id);
            finish_request_locked(request, RequestStatus::Failed, error.what());
        }
    }
    for (const Work& item : launched) {
        try {
            const int32_t token = item.lane->model->session().decode_async_finish();
            accept_token(item, token, false);
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            Request& request = registry_.at(item.id);
            finish_request_locked(request, RequestStatus::Failed, error.what());
        }
    }
    return !work.empty();
}

} // namespace celeg

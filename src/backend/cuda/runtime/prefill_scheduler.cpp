#include "engine_internal.hpp"
// Keep the prefill scheduler bound to the concrete CudaModel contract.

namespace celeg {
bool CudaSchedulerDriver::run_prefill_work() {
    struct Work { Lane* lane; RequestId id; size_t begin; size_t count; bool first; bool final; };
    std::vector<Work> work;
    int token_budget = engine_options_.max_batched_tokens;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        int decode_reservation = 0;
        for (const auto& lane_ptr : lanes_) {
            if (lane_ptr->request_id == 0) continue;
            const Request& request = registry_.at(lane_ptr->request_id);
            if (request.status == RequestStatus::Decoding &&
                !request.cancel_requested) ++decode_reservation;
        }
        token_budget = std::max(0, token_budget - decode_reservation);
        const std::vector<int> ordered =
            planner_.order_prefill(lane_snapshots_locked());
        for (const int lane_index : ordered) {
            auto& lane_ptr = lanes_.at(static_cast<size_t>(lane_index));
            if (token_budget <= 0) break;
            Lane& lane = *lane_ptr;
            if (lane.request_id == 0) continue;
            Request& request = registry_.at(lane.request_id);
            if (request.status != RequestStatus::Prefill) continue;
            if (request.cancel_requested) {
                finish_request_locked(request, RequestStatus::Cancelled);
                continue;
            }
            const size_t remaining = request.prompt.size() - request.prefill_offset;
            const size_t count = std::min<size_t>(
                remaining, std::min(engine_options_.prefill_chunk_tokens,
                                    token_budget));
            if (count == 0) continue;
            work.push_back({&lane, request.id, request.prefill_offset, count,
                            request.prefill_offset == 0,
                            count == remaining});
            token_budget -= static_cast<int>(count);
        }
    }
    if (packed_executor_ && engine_options_.ragged_packed_prefill &&
        work.size() >= static_cast<size_t>(engine_options_.ragged_prefill_min_batch)) {
        std::vector<Work> active;
        std::vector<PackedSessionContext> models;
        std::vector<std::vector<uint32_t>> page_tables;
        std::vector<int32_t> tokens;
        std::vector<PackedPrefillRow> rows;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const Work& item : work) {
                Request& request = registry_.at(item.id);
                if (request.status != RequestStatus::Prefill ||
                    request.cancel_requested) continue;
                active.push_back(item);
                models.push_back(packed_session_context(*item.lane->model));
                page_tables.push_back(request.pages);
                rows.push_back(PackedPrefillRow{
                    tokens.size(),
                    item.count,
                    static_cast<uint8_t>(item.final)});
                tokens.insert(tokens.end(),
                              request.prompt.begin() + static_cast<ptrdiff_t>(item.begin),
                              request.prompt.begin() + static_cast<ptrdiff_t>(item.begin + item.count));
            }
        }
        if (!active.empty()) {
            try {
                const PackedDecodeMetrics before = packed_executor_->metrics();
                packed_executor_->prefill(models, page_tables, tokens, rows);
                const PackedDecodeMetrics after = packed_executor_->metrics();
                std::lock_guard<std::mutex> lock(mutex_);
                metrics_.ragged_prefill_steps +=
                    after.ragged_prefill_steps - before.ragged_prefill_steps;
                metrics_.cumulative_ragged_prefill_ms +=
                    after.cumulative_prefill_ms - before.cumulative_prefill_ms;
                metrics_.maximum_ragged_prefill_batch = std::max<uint64_t>(
                    metrics_.maximum_ragged_prefill_batch,
                    static_cast<uint64_t>(active.size()));
                for (size_t row = 0; row < active.size(); ++row) {
                    const Work& item = active[row];
                    Request& request = registry_.at(item.id);
                    if (request.cancel_requested) {
                        finish_request_locked(request, RequestStatus::Cancelled);
                        continue;
                    }
                    if (request.status != RequestStatus::Prefill) continue;
                    request.prefill_offset += item.count;
                    metrics_.prefill_tokens += item.count;
                    metrics_.direct_paged_prefill_tokens += item.count;
                    metrics_.ragged_prefill_tokens += item.count;
                    if (item.final) {
                        complete_prefill_locked(request, *item.lane);
                    }
                }
            } catch (const std::exception& error) {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const Work& item : active) {
                    Request& request = registry_.at(item.id);
                    if (request.status == RequestStatus::Prefill) {
                        finish_request_locked(request, RequestStatus::Failed,
                                              error.what());
                    }
                }
            }
        }
        return !work.empty();
    }
    for (const Work& item : work) {
        try {
            std::vector<int32_t> chunk;
            std::vector<uint32_t> page_table;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                Request& request = registry_.at(item.id);
                chunk.assign(request.prompt.begin() + static_cast<ptrdiff_t>(item.begin),
                             request.prompt.begin() + static_cast<ptrdiff_t>(item.begin + item.count));
                page_table = request.pages;
            }
            if (packed_executor_) {
                item.lane->model->session().prefill_chunk_paged(
                    chunk, item.first, item.final, *paged_kv_, page_table);
            } else if (item.first && item.final) {
                item.lane->model->session().prefill(chunk);
            } else {
                item.lane->model->session().prefill_chunk(chunk, item.first, item.final);
            }
            std::lock_guard<std::mutex> lock(mutex_);
            Request& request = registry_.at(item.id);
            if (request.cancel_requested) {
                finish_request_locked(request, RequestStatus::Cancelled);
                continue;
            }
            request.prefill_offset += item.count;
            metrics_.prefill_tokens += item.count;
            metrics_.lane_prefill_tokens += item.count;
            if (packed_executor_) metrics_.direct_paged_prefill_tokens += item.count;
            if (item.final) {
                complete_prefill_locked(request, *item.lane);
            }
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            Request& request = registry_.at(item.id);
            finish_request_locked(request, RequestStatus::Failed, error.what());
        }
    }
    return !work.empty();
}

} // namespace celeg

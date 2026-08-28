#include "celeg/serve/metal_inference_service.hpp"

#include "celeg/backend/metal/device.hpp"
#include "celeg/backend/metal/model.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace celeg::serve {

namespace {

bool terminal(RequestStatus status) {
    return status == RequestStatus::Finished ||
           status == RequestStatus::Cancelled ||
           status == RequestStatus::Failed;
}

}

struct MetalInferenceService::Impl {
    struct Request {
        RequestId id = 0;
        GenerateRequest input;
        RequestStatus status = RequestStatus::Queued;
        std::vector<int32_t> pending_tokens;
        std::size_t generated_tokens = 0;
        FinishReason finish_reason = FinishReason::None;
        std::string error;
        std::optional<MetalSessionSnapshot> snapshot;
    };

    using PrefixKey = std::vector<int32_t>;
    using PrefixCache = std::map<PrefixKey,
                                 std::shared_ptr<const MetalSessionSnapshot>>;

    Impl(std::string model_path,
         int max_context,
         MetalModelOptions model_options,
         MetalEngineOptions engine_options,
         std::shared_ptr<const RuntimeContext> runtime)
        : device(),
          model(model_path, max_context, model_options, {}, std::move(runtime)),
         session(model.session()),
          model_info(),
          engine_options(std::move(engine_options)) {
        if (this->engine_options.max_active_requests <= 0) {
            throw std::invalid_argument("Metal service max_active_requests must be positive");
        }
        if (this->engine_options.max_batched_tokens <= 0 ||
            this->engine_options.prefill_chunk_tokens <= 0 ||
            this->engine_options.kv_page_tokens <= 0 ||
            (this->engine_options.prefix_cache &&
             this->engine_options.prefix_cache_max_entries == 0)) {
            throw std::invalid_argument("Metal engine limits must be positive");
        }
        device.run_probe();
        model_info.name = std::filesystem::path(model_path).stem().string();
        model_info.backend = "metal";
        model_info.max_context = max_context;
        model_info.limits.max_active_requests = this->engine_options.max_active_requests;
        model_info.limits.max_batched_tokens = this->engine_options.max_batched_tokens;
        model_info.limits.prefill_chunk_tokens = this->engine_options.prefill_chunk_tokens;
    }

    void cache_prefix(const PrefixKey& key, const MetalSessionSnapshot& snapshot) {
        if (!engine_options.prefix_cache || key.empty()) return;
        auto value = std::make_shared<MetalSessionSnapshot>(snapshot);
        const auto found = prefix_cache.find(key);
        if (found != prefix_cache.end()) {
            found->second = std::move(value);
            const auto old = std::find(prefix_lru.begin(), prefix_lru.end(), key);
            if (old != prefix_lru.end()) prefix_lru.erase(old);
        } else {
            prefix_cache.emplace(key, std::move(value));
        }
        prefix_lru.push_back(key);
        while (prefix_lru.size() > engine_options.prefix_cache_max_entries) {
            prefix_cache.erase(prefix_lru.front());
            prefix_lru.pop_front();
        }
    }

    std::shared_ptr<const MetalSessionSnapshot> find_prefix(
        std::span<const int32_t> tokens, size_t& matched) const {
        matched = 0;
        std::shared_ptr<const MetalSessionSnapshot> result;
        for (const auto& [key, snapshot] : prefix_cache) {
            if (key.size() <= matched || key.size() > tokens.size()) continue;
            if (std::equal(key.begin(), key.end(), tokens.begin())) {
                matched = key.size();
                result = snapshot;
            }
        }
        return result;
    }

    void refresh_counts() {
        metrics.active_requests = 0;
        metrics.queued_requests = 0;
        for (const auto& [id, value] : requests) {
            (void)id;
            if (value.status == RequestStatus::Queued) {
                ++metrics.queued_requests;
            } else if (!terminal(value.status)) {
                ++metrics.active_requests;
            }
        }
    }

    void fail(Request& value, const std::exception& error) {
        value.status = RequestStatus::Failed;
        value.finish_reason = FinishReason::Error;
        value.error = error.what();
        ++metrics.failed_requests;
        refresh_counts();
    }

    Request* next_request() {
        if (requests.empty()) return nullptr;
        auto start = requests.upper_bound(last_scheduled_id);
        if (start == requests.end()) start = requests.begin();
        for (auto it = start; it != requests.end(); ++it) {
            if (!terminal(it->second.status)) {
                last_scheduled_id = it->first;
                return &it->second;
            }
        }
        for (auto it = requests.begin(); it != start; ++it) {
            if (!terminal(it->second.status)) {
                last_scheduled_id = it->first;
                return &it->second;
            }
        }
        return nullptr;
    }

    MetalDevice device;
    MetalModel model;
    MetalInferenceSession session;
    ModelInfo model_info;
    MetalEngineOptions engine_options;
    mutable std::mutex mutex;
    std::map<RequestId, Request> requests;
    RequestId next_id = 1;
    RequestId last_scheduled_id = 0;
    PrefixCache prefix_cache;
    std::deque<PrefixKey> prefix_lru;
    ServingMetrics metrics;
    bool running = false;
};

MetalInferenceService::MetalInferenceService(
    std::string model_path,
    int max_context,
    MetalModelOptions model_options,
    MetalEngineOptions engine_options,
    std::shared_ptr<const RuntimeContext> runtime)
    : impl_(std::make_unique<Impl>(std::move(model_path), max_context,
                                   model_options, engine_options,
                                   std::move(runtime))) {}

MetalInferenceService::~MetalInferenceService() = default;

RequestId MetalInferenceService::submit(GenerateRequest request) {
    std::lock_guard lock((*impl_).mutex);
    if (request.prompt_tokens.empty()) throw std::invalid_argument("Metal request prompt is empty");
    if (request.prompt_tokens.size() + request.max_output_tokens >
        static_cast<size_t>((*impl_).model_info.max_context)) {
        throw std::invalid_argument("Metal request exceeds configured context");
    }
    if (!request.prompt_embedding.empty() || !request.images.empty()) {
        throw std::invalid_argument("Metal service does not support multimodal prompts");
    }
    request.generation.validate();
    const RequestId id = (*impl_).next_id++;
    (*impl_).requests.emplace(id, Impl::Request{
        id, std::move(request), RequestStatus::Queued,
        {}, 0, FinishReason::None, {}, std::nullopt});
    ++(*impl_).metrics.submitted_requests;
    (*impl_).refresh_counts();
    return id;
}

GenerateEvent MetalInferenceService::poll(RequestId id, std::size_t max_tokens) {
    std::lock_guard lock((*impl_).mutex);
    const auto found = (*impl_).requests.find(id);
    if (found == (*impl_).requests.end()) {
        throw std::out_of_range("unknown Metal request id");
    }
    Impl::Request& request = found->second;
    GenerateEvent event;
    event.request_id = id;
    event.status = request.status;
    event.error = request.error;
    const size_t count = max_tokens == 0
        ? request.pending_tokens.size()
        : std::min(max_tokens, request.pending_tokens.size());
    event.tokens.assign(request.pending_tokens.begin(), request.pending_tokens.begin() + count);
    request.pending_tokens.erase(request.pending_tokens.begin(),
                                 request.pending_tokens.begin() + count);
    event.finished = terminal(request.status);
    event.finish_reason = request.finish_reason;
    return event;
}

RequestStatus MetalInferenceService::status(RequestId id) const {
    std::lock_guard lock((*impl_).mutex);
    const auto found = (*impl_).requests.find(id);
    if (found == (*impl_).requests.end()) {
        throw std::out_of_range("unknown Metal request id");
    }
    return found->second.status;
}

bool MetalInferenceService::cancel(RequestId id) {
    std::lock_guard lock((*impl_).mutex);
    const auto found = (*impl_).requests.find(id);
    if (found == (*impl_).requests.end()) return false;
    Impl::Request& request = found->second;
    if (terminal(request.status)) return false;
    request.status = RequestStatus::Cancelled;
    request.finish_reason = FinishReason::Cancelled;
    ++(*impl_).metrics.cancelled_requests;
    (*impl_).refresh_counts();
    return true;
}

bool MetalInferenceService::release(RequestId id) {
    std::lock_guard lock((*impl_).mutex);
    const auto found = (*impl_).requests.find(id);
    if (found == (*impl_).requests.end() || !terminal(found->second.status)) return false;
    (*impl_).requests.erase(found);
    (*impl_).refresh_counts();
    return true;
}

ModelInfo MetalInferenceService::model_info() const { return (*impl_).model_info; }

ServingMetrics MetalInferenceService::metrics() const {
    std::lock_guard lock((*impl_).mutex);
    return (*impl_).metrics;
}

bool MetalInferenceService::step() {
    std::lock_guard lock((*impl_).mutex);
    if (!(*impl_).running) return false;
    Impl::Request* request = (*impl_).next_request();
    if (!request) return false;
    try {
        if (request->status == RequestStatus::Queued) {
            const auto started = std::chrono::steady_clock::now();
            size_t cached_tokens = 0;
            const auto cached = (*impl_).find_prefix(
                request->input.prompt_tokens, cached_tokens);
            if (cached) {
                MetalSessionSnapshot snapshot = *cached;
                snapshot.metrics = {};
                (*impl_).model.restore_session_snapshot(std::move(snapshot));
                (*impl_).session.set_generation_config(request->input.generation);
                for (size_t index = cached_tokens;
                     index < request->input.prompt_tokens.size(); ++index) {
                    (*impl_).session.eval_token(request->input.prompt_tokens[index]);
                }
            } else {
                (*impl_).session.reset();
                (*impl_).session.set_generation_config(request->input.generation);
                (*impl_).session.prefill(request->input.prompt_tokens);
            }
            request->snapshot = (*impl_).model.export_session_snapshot();
            (*impl_).cache_prefix(request->input.prompt_tokens, *request->snapshot);
            request->status = RequestStatus::Decoding;
            const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            if (elapsed > 0.0) {
                (*impl_).metrics.prefill_tokens_per_second =
                    static_cast<double>(request->input.prompt_tokens.size()) * 1000.0 / elapsed;
            }
            if (request->input.max_output_tokens == 0) {
                request->status = RequestStatus::Finished;
                request->finish_reason = FinishReason::Length;
                ++(*impl_).metrics.completed_requests;
            }
            (*impl_).refresh_counts();
            return true;
        }
        if (!request->snapshot) {
            throw std::runtime_error("Metal request has no session snapshot");
        }
        (*impl_).model.restore_session_snapshot(std::move(*request->snapshot));
        request->snapshot.reset();
        if (request->generated_tokens >= request->input.max_output_tokens) {
            request->status = RequestStatus::Finished;
            request->finish_reason = FinishReason::Length;
            ++(*impl_).metrics.completed_requests;
            (*impl_).refresh_counts();
            return true;
        }
        const int32_t token = (*impl_).session.decode();
        request->pending_tokens.push_back(token);
        ++request->generated_tokens;
        request->snapshot = (*impl_).model.export_session_snapshot();
        if (is_stop_token(request->input.eos_token_ids, token)) {
            request->status = RequestStatus::Finished;
            request->finish_reason = FinishReason::Stop;
            ++(*impl_).metrics.completed_requests;
        } else if (request->generated_tokens >= request->input.max_output_tokens) {
            request->status = RequestStatus::Finished;
            request->finish_reason = FinishReason::Length;
            ++(*impl_).metrics.completed_requests;
        }
        (*impl_).metrics.decode_tokens_per_second =
            (*impl_).model.metrics().decode_tokens_per_second();
        (*impl_).refresh_counts();
        return true;
    } catch (const std::exception& error) {
        (*impl_).fail(*request, error);
        return false;
    }
}

void MetalInferenceService::start() {
    std::lock_guard lock((*impl_).mutex);
    (*impl_).running = true;
}

void MetalInferenceService::stop() {
    std::lock_guard lock((*impl_).mutex);
    (*impl_).running = false;
}

}

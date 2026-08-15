#include "celeg/detail/runtime/concurrency/request_registry.hpp"

#include <algorithm>
#include <utility>

namespace celeg::detail {

RequestRegistry::RequestId RequestRegistry::create(
    std::vector<int32_t> prompt,
    ConcurrentRequestOptions options,
    std::chrono::steady_clock::time_point submitted_at) {
    const RequestId id = next_request_id_++;
    auto request = std::make_unique<RequestRecord>();
    request->id = id;
    request->options = std::move(options);
    request->prompt = std::move(prompt);
    request->submitted_at = submitted_at;
    requests_.emplace(id, std::move(request));
    admission_queue_.insert({-requests_[id]->options.priority, id});
    return id;
}

RequestRecord* RequestRegistry::find(RequestId id) {
    const auto it = requests_.find(id);
    return it == requests_.end() ? nullptr : it->second.get();
}

const RequestRecord* RequestRegistry::find(RequestId id) const {
    const auto it = requests_.find(id);
    return it == requests_.end() ? nullptr : it->second.get();
}

RequestRecord& RequestRegistry::at(RequestId id) {
    RequestRecord* value = find(id);
    if (!value) throw std::out_of_range("unknown request id");
    return *value;
}

const RequestRecord& RequestRegistry::at(RequestId id) const {
    const RequestRecord* value = find(id);
    if (!value) throw std::out_of_range("unknown request id");
    return *value;
}

std::optional<RequestRegistry::RequestId> RequestRegistry::best_queued() const {
    if (admission_queue_.empty()) return std::nullopt;
    return admission_queue_.begin()->second;
}

void RequestRegistry::erase_queued(RequestId id) {
    const auto* record = find(id);
    if (!record) return;
    admission_queue_.erase({-record->options.priority, id});
}

bool RequestRegistry::erase(RequestId id) {
    const auto* record = find(id);
    if (!record) return false;
    admission_queue_.erase({-record->options.priority, id});
    requests_.erase(id);
    return true;
}

}

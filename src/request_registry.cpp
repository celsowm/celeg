#include "lfm/detail/request_registry.hpp"

#include <algorithm>
#include <utility>

namespace lfm::detail {

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
    admission_queue_.push_back(id);
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

void RequestRegistry::erase_queued(RequestId id) {
    const auto it = std::find(admission_queue_.begin(), admission_queue_.end(), id);
    if (it != admission_queue_.end()) admission_queue_.erase(it);
}

} // namespace lfm::detail

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace celeg {

enum class RequestStatus : uint8_t {
    Queued,
    Prefill,
    Decoding,
    Finished,
    Cancelled,
    Failed,
};

const char* request_status_name(RequestStatus status);

inline constexpr bool is_terminal(RequestStatus status) {
    return status == RequestStatus::Finished ||
           status == RequestStatus::Cancelled ||
           status == RequestStatus::Failed;
}

enum class SchedulerPolicy : uint8_t {
    GuaranteedNoEvict,
    MaxUtilization,
};

class PagedBlockPool {
public:
    PagedBlockPool(size_t pages, int page_tokens);

    std::optional<std::vector<uint32_t>> allocate_tokens(size_t token_count);
    bool retain(const std::vector<uint32_t>& pages);
    void release(const std::vector<uint32_t>& pages);
    size_t free_pages() const;
    uint32_t ref_count(uint32_t page) const;
    size_t total_pages() const { return ref_counts_.size(); }
    size_t used_pages() const { return total_pages() - free_pages(); }
    int page_tokens() const { return page_tokens_; }

private:
    int page_tokens_;
    std::vector<uint32_t> free_;
    std::vector<uint32_t> ref_counts_;
};

} // namespace celeg

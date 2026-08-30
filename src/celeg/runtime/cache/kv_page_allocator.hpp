#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace celeg {

class IKvPageAllocator {
public:
    virtual ~IKvPageAllocator() = default;

    virtual std::optional<std::vector<uint32_t>> allocate_tokens(
        size_t token_count) = 0;
    virtual bool retain(const std::vector<uint32_t>& pages) = 0;
    virtual void release(const std::vector<uint32_t>& pages) = 0;
    virtual std::optional<uint32_t> clone_page_prefix(
        uint32_t source, int used_tokens) = 0;

    virtual size_t total_pages() const = 0;
    virtual size_t used_pages() const = 0;
    virtual int page_tokens() const = 0;
    virtual size_t page_bytes() const = 0;
};

}

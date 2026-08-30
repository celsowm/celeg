#pragma once

#include "celeg/model/program.hpp"
#include "celeg/runtime/cache/expert_policy.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>

namespace celeg {

struct ExpertPayloadRequest {
    ExpertKey key;
    const ExpertPayloadSchema* schema = nullptr;

    void validate() const {
        key.validate();
        if (!schema) throw std::invalid_argument("expert payload schema is required");
        schema->validate();
    }
};

class IExpertSource {
public:
    virtual ~IExpertSource() = default;
    virtual void read(const ExpertPayloadRequest& request,
                      std::span<std::byte> destination) const = 0;
};

class IHostExpertCache {
public:
    virtual ~IHostExpertCache() = default;
    virtual std::size_t resident_bytes() const = 0;
    virtual ExpertCacheMetrics metrics() const = 0;
};

}

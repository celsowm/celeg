#pragma once

#include "celeg/serve/types.hpp"

#include <cstddef>

namespace celeg::serve {

// Canonical inference surface shared by the C API and the HTTP server so
// neither has to branch on backend. CpuInferenceService and
// CudaInferenceService are the only two implementations; callers must not
// reach past this interface into CpuConcurrentEngine / ConcurrentEngine.
class IInferenceService {
public:
    virtual ~IInferenceService() = default;

    virtual RequestId submit(GenerateRequest request) = 0;

    // max_tokens == 0 drains everything currently buffered for the request.
    virtual GenerateEvent poll(RequestId id, std::size_t max_tokens) = 0;

    virtual RequestStatus status(RequestId id) const = 0;
    virtual bool cancel(RequestId id) = 0;

    // Frees the request record. Only valid once the request has reached a
    // terminal status (Finished, Cancelled or Failed); returns false
    // otherwise or if the id is unknown.
    virtual bool release(RequestId id) = 0;

    virtual ModelInfo model_info() const = 0;
    virtual ServingMetrics metrics() const = 0;

    // Drives the underlying scheduler. Intended to be called from a single
    // dispatcher thread, never from the HTTP event loop.
    virtual bool step() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
};

} // namespace celeg::serve

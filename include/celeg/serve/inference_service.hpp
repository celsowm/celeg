#pragma once

#include "celeg/serve/types.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace celeg::serve {

// Canonical inference surface shared by the C API and the HTTP server so
// neither has to branch on backend. CpuInferenceService and
// CudaInferenceService are the only two implementations; callers must not
// reach past this interface into CpuConcurrentEngine / ConcurrentEngine.
class IRequestService {
public:
    virtual ~IRequestService() = default;

    virtual RequestId submit(GenerateRequest request) = 0;

    // max_tokens == 0 drains everything currently buffered for the request.
    virtual GenerateEvent poll(RequestId id, std::size_t max_tokens) = 0;

    virtual RequestStatus status(RequestId id) const = 0;
    virtual bool cancel(RequestId id) = 0;

    // Frees the request record. Only valid once the request has reached a
    // terminal status (Finished, Cancelled or Failed); returns false
    // otherwise or if the id is unknown.
    virtual bool release(RequestId id) = 0;

};

class ISchedulerDriver {
public:
    virtual ~ISchedulerDriver() = default;

    // Drives the underlying scheduler. Intended to be called from a single
    // dispatcher thread, never from the HTTP event loop.
    virtual bool step() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
};

class IServiceDiagnostics {
public:
    virtual ~IServiceDiagnostics() = default;

    virtual ModelInfo model_info() const = 0;
    virtual ServingMetrics metrics() const = 0;
};

// Owns one concrete service while exposing its three independent roles. The
// role pointers are non-owning views into requests; destruction is performed
// exactly once through the request interface.
class ServiceBundle {
public:
    template <typename Service>
    explicit ServiceBundle(std::unique_ptr<Service> service)
        : requests_(std::move(service)) {
        static_assert(std::is_base_of_v<IRequestService, Service> &&
                      std::is_base_of_v<ISchedulerDriver, Service> &&
                      std::is_base_of_v<IServiceDiagnostics, Service>,
                      "ServiceBundle requires one concrete service implementing all roles");
        auto* concrete = static_cast<Service*>(requests_.get());
        scheduler_ = concrete;
        diagnostics_ = concrete;
    }

    IRequestService& requests() { return *requests_; }
    const IRequestService& requests() const { return *requests_; }
    ISchedulerDriver& scheduler() { return *scheduler_; }
    IServiceDiagnostics& diagnostics() { return *diagnostics_; }

private:
    std::unique_ptr<IRequestService> requests_;
    ISchedulerDriver* scheduler_ = nullptr;
    IServiceDiagnostics* diagnostics_ = nullptr;
};

} // namespace celeg::serve

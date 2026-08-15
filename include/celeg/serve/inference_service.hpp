#pragma once

#include "celeg/serve/types.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace celeg::serve {

class IRequestService {
public:
    virtual ~IRequestService() = default;

    virtual RequestId submit(GenerateRequest request) = 0;

    virtual GenerateEvent poll(RequestId id, std::size_t max_tokens) = 0;

    virtual RequestStatus status(RequestId id) const = 0;
    virtual bool cancel(RequestId id) = 0;

    virtual bool release(RequestId id) = 0;

};

class ISchedulerDriver {
public:
    virtual ~ISchedulerDriver() = default;

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

}

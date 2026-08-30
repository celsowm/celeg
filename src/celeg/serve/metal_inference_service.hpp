#pragma once

#include "celeg/backend/metal/runtime_types.hpp"
#include "celeg/runtime/context.hpp"
#include "celeg/serve/inference_service.hpp"

#include <memory>
#include <string>

namespace celeg::serve {

class MetalInferenceService final : public IRequestService,
                                    public ISchedulerDriver,
                                    public IServiceDiagnostics {
public:
    MetalInferenceService(std::string model_path,
                          int max_context,
                          MetalModelOptions model_options = {},
                          MetalEngineOptions engine_options = {},
                          std::shared_ptr<const RuntimeContext> runtime = nullptr);
    ~MetalInferenceService() override;

    RequestId submit(GenerateRequest request) override;
    GenerateEvent poll(RequestId id, std::size_t max_tokens) override;
    RequestStatus status(RequestId id) const override;
    bool cancel(RequestId id) override;
    bool release(RequestId id) override;

    ModelInfo model_info() const override;
    ServingMetrics metrics() const override;

    bool step() override;
    void start() override;
    void stop() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}

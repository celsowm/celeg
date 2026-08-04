#pragma once

#include "celeg/serve/inference_service.hpp"
#include "celeg/serve/request_lifecycle.hpp"
#include "celeg/backend/cpu/concurrent.hpp"

#include <string>

namespace celeg::serve {

class CpuInferenceService final : public IRequestService,
                                  public ISchedulerDriver,
                                  public IServiceDiagnostics {
public:
    CpuInferenceService(const std::string& model_path,
                        int max_context,
                        CpuModelOptions model_options = {},
                        CpuConcurrentEngineOptions engine_options = {},
                        VisualEmbeddingProvider visual_embeddings = {},
                        std::shared_ptr<const RuntimeContext> runtime = nullptr);

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
    CpuConcurrentEngine engine_;
    ModelInfo model_info_;
    RequestLifecycle lifecycle_;
    VisualEmbeddingProvider visual_embeddings_;
};

} // namespace celeg::serve

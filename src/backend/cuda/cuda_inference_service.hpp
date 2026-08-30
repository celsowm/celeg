#pragma once

#include "celeg/serve/inference_service.hpp"
#include "celeg/serve/request_lifecycle.hpp"
#include "runtime_types.hpp"
#include "backend/cuda/concurrency.hpp"

#include <string>

namespace celeg::serve {

class CudaInferenceService final : public serve::IRequestService,
                                   public serve::ISchedulerDriver,
                                   public serve::IServiceDiagnostics {
public:
    CudaInferenceService(std::string model_path,
                         int max_context,
                         CudaModelOptions model_options = {},
                         ConcurrentEngineOptions engine_options = {},
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
    ConcurrentEngine engine_;
    ModelInfo model_info_;
    RequestLifecycle lifecycle_;
    VisualEmbeddingProvider visual_embeddings_;
};

}

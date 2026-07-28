#pragma once

#include "lfm/serve/inference_service.hpp"
#include "lfm/runtime/concurrency.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace lfm::serve {

class CudaInferenceService final : public IInferenceService {
public:
    CudaInferenceService(std::string model_path,
                         int max_context,
                         ModelOptions model_options = {},
                         ConcurrentEngineOptions engine_options = {});

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
    struct RequestMeta {
        std::int32_t eos_token_id = 7;
        bool saw_eos = false;
    };

    ConcurrentEngine engine_;
    ModelInfo model_info_;

    mutable std::mutex meta_mutex_;
    std::unordered_map<RequestId, RequestMeta> meta_;
};

} // namespace lfm::serve

#include "api_internal.hpp"
#include "celeg/serve/cpu_inference_service.hpp"

#ifdef CELEG_API_WITH_CUDA
#include "celeg/backend/cuda/cuda_inference_service.hpp"
#endif

#include <stdexcept>

namespace celeg::api {

std::unique_ptr<celeg::serve::ServiceBundle> create_service_bundle(
    const char* path, const celeg_engine_options& options,
    std::shared_ptr<const celeg::RuntimeContext> runtime) {
    if (!path || !*path) throw std::invalid_argument("engine path is required");
    if (!runtime) throw std::invalid_argument("runtime context is required");
    if (options.backend != options.model.backend) {
        throw std::invalid_argument("engine backend must match model backend");
    }

    if (options.backend == CELEG_BACKEND_CPU) {
        (void)runtime->backends().select_if([](const celeg::IBackendFactory& factory) {
            return factory.supports(celeg::BackendKind::Cpu);
        });
        auto service = std::make_unique<celeg::serve::CpuInferenceService>(
            path, options.model.max_context, cpu_options(options.model),
            cpu_engine_options(options), celeg::VisualEmbeddingProvider{}, runtime);
        return std::make_unique<celeg::serve::ServiceBundle>(std::move(service));
    }
    if (options.backend == CELEG_BACKEND_CUDA) {
#ifdef CELEG_API_WITH_CUDA
        (void)runtime->backends().select_if([](const celeg::IBackendFactory& factory) {
            return factory.supports(celeg::BackendKind::Cuda);
        });
        auto service = std::make_unique<celeg::serve::CudaInferenceService>(
            path, options.model.max_context, cuda_options(options.model),
            cuda_engine_options(options), celeg::VisualEmbeddingProvider{}, runtime);
        return std::make_unique<celeg::serve::ServiceBundle>(std::move(service));
#else
        throw std::invalid_argument("CUDA backend is unavailable in this build");
#endif
    }
    throw std::invalid_argument("unknown backend");
}

} // namespace celeg::api

#include "api_internal.hpp"
#include "celeg/serve/cpu_inference_service.hpp"

#ifdef CELEG_API_WITH_CUDA
#include "celeg/backend/cuda/cuda_inference_service.hpp"
#endif

#include <memory>
#include <stdexcept>
#include <vector>

namespace celeg::api {

namespace {

class CpuBackendOptions final : public IBackendOptions {
public:
    CpuBackendOptions(CpuModelOptions model_value,
                      CpuConcurrentEngineOptions engine_value)
        : model(std::move(model_value)), engine(std::move(engine_value)) {}

    BackendId backend_id() const noexcept override { return "cpu"; }

    const void* type_tag() const noexcept override {
        return type_tag_for<CpuBackendOptions>();
    }

    const void* data() const noexcept override { return this; }

    CpuModelOptions model;
    CpuConcurrentEngineOptions engine;
};

class CpuBackendFactory final : public IBackendFactory {
public:
    std::string_view id() const override { return "cpu"; }
    bool supports(BackendId backend) const override {
        return backend == "cpu";
    }

    std::unique_ptr<serve::ServiceBundle> create(
        const BackendCreateRequest& request) const override {
        if (!request.runtime || !request.options ||
            request.backend_id != id() ||
            request.options->backend_id() != id()) {
            throw std::invalid_argument("CPU backend request has invalid options");
        }
        const auto* options = request.options->as<CpuBackendOptions>();
        if (!options) throw std::invalid_argument("CPU backend options type mismatch");
        auto service = std::make_unique<serve::CpuInferenceService>(
            request.model_path, request.max_context, options->model, options->engine,
            VisualEmbeddingProvider{}, request.runtime);
        return std::make_unique<serve::ServiceBundle>(std::move(service));
    }
};

#ifdef CELEG_API_WITH_CUDA
class CudaBackendOptions final : public IBackendOptions {
public:
    CudaBackendOptions(CudaModelOptions model_value,
                       ConcurrentEngineOptions engine_value)
        : model(std::move(model_value)), engine(std::move(engine_value)) {}

    BackendId backend_id() const noexcept override { return "cuda"; }

    const void* type_tag() const noexcept override {
        return type_tag_for<CudaBackendOptions>();
    }

    const void* data() const noexcept override { return this; }

    CudaModelOptions model;
    ConcurrentEngineOptions engine;
};

class CudaBackendFactory final : public IBackendFactory {
public:
    std::string_view id() const override { return "cuda"; }
    bool supports(BackendId backend) const override {
        return backend == "cuda";
    }

    std::unique_ptr<serve::ServiceBundle> create(
        const BackendCreateRequest& request) const override {
        if (!request.runtime || !request.options ||
            request.backend_id != id() ||
            request.options->backend_id() != id()) {
            throw std::invalid_argument("CUDA backend request has invalid options");
        }
        const auto* options = request.options->as<CudaBackendOptions>();
        if (!options) throw std::invalid_argument("CUDA backend options type mismatch");
        auto service = std::make_unique<serve::CudaInferenceService>(
            request.model_path, request.max_context, options->model, options->engine,
            VisualEmbeddingProvider{}, request.runtime);
        return std::make_unique<serve::ServiceBundle>(std::move(service));
    }
};
#endif

std::shared_ptr<const RuntimeContext> create_backend_runtime(
    const celeg_engine_options& options) {
    RuntimeBuilder builder;
    builder.add_builtins();
    builder.add_backend_factory(std::make_unique<CpuBackendFactory>());
#ifdef CELEG_API_WITH_CUDA
    builder.add_backend_factory(std::make_unique<CudaBackendFactory>());
#else
    if (options.backend == CELEG_BACKEND_CUDA) {
        throw std::invalid_argument("CUDA backend is unavailable in this build");
    }
#endif
    return builder.build_shared();
}

} // namespace

std::unique_ptr<celeg::serve::ServiceBundle> create_service_bundle(
    const char* path, const celeg_engine_options& options) {
    if (!path || !*path) throw std::invalid_argument("engine path is required");
    if (options.backend != options.model.backend) {
        throw std::invalid_argument("engine backend must match model backend");
    }
    const std::shared_ptr<const RuntimeContext> runtime =
        create_backend_runtime(options);
    BackendCreateRequest request;
    request.model_path = path;
    request.max_context = options.model.max_context;
    request.backend_id = options.backend == CELEG_BACKEND_CPU ? "cpu" : "cuda";
    request.runtime = runtime;
    if (options.backend == CELEG_BACKEND_CPU) {
        request.options = std::make_shared<CpuBackendOptions>(
            cpu_options(options.model), cpu_engine_options(options));
    }
#ifdef CELEG_API_WITH_CUDA
    else if (options.backend == CELEG_BACKEND_CUDA) {
        request.options = std::make_shared<CudaBackendOptions>(
            cuda_options(options.model), cuda_engine_options(options));
    }
#endif
    else {
        throw std::invalid_argument("unknown backend");
    }
    const IBackendFactory& factory = runtime->backends().select_best_if(
        [backend_id = request.backend_id](const IBackendFactory& candidate) {
            return candidate.supports(backend_id);
        });
    return factory.create(request);
}

} // namespace celeg::api

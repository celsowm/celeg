#include "api_internal.hpp"
#include "celeg/serve/cpu_inference_service.hpp"

#ifdef CELEG_API_WITH_CUDA
#include "backend/cuda/cuda_inference_service.hpp"
#endif
#ifdef CELEG_API_WITH_METAL
#include "celeg/serve/metal_inference_service.hpp"
#endif

#include <memory>
#include <stdexcept>
#include <vector>

namespace celeg::api {

namespace {

class CpuBackendOptions final : public IBackendOptions {
public:
    static constexpr std::string_view kSchemaId = "cpu/1";

    CpuBackendOptions(CpuModelOptions model_value,
                      CpuConcurrentEngineOptions engine_value)
        : model(std::move(model_value)), engine(std::move(engine_value)) {}

    BackendId backend_id() const noexcept override { return "cpu"; }

    std::string_view schema_id() const noexcept override { return kSchemaId; }

    const void* data() const noexcept override { return this; }

    CpuModelOptions model;
    CpuConcurrentEngineOptions engine;
};

class CpuBackendFactory final : public IAbiBackendFactory {
public:
    std::string_view id() const override { return "cpu"; }
    bool supports(BackendId backend) const override {
        return backend == "cpu";
    }

    std::shared_ptr<const IBackendOptions> decode_options(
        std::span<const std::byte> bytes) const override {
        const auto& source = decode_abi_struct<celeg_cpu_backend_options>(bytes, "CPU backend");
        return std::make_shared<CpuBackendOptions>(
            cpu_options(source.model), cpu_engine_options(source.engine));
    }

    std::unique_ptr<serve::ServiceBundle> create(
        const BackendCreateRequest& request) const override {
        validate_backend_request(request, id());
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
    static constexpr std::string_view kSchemaId = "cuda/1";

    CudaBackendOptions(CudaModelOptions model_value,
                       ConcurrentEngineOptions engine_value)
        : model(std::move(model_value)), engine(std::move(engine_value)) {}

    BackendId backend_id() const noexcept override { return "cuda"; }

    std::string_view schema_id() const noexcept override { return kSchemaId; }

    const void* data() const noexcept override { return this; }

    CudaModelOptions model;
    ConcurrentEngineOptions engine;
};

class CudaBackendFactory final : public IAbiBackendFactory {
public:
    std::string_view id() const override { return "cuda"; }
    bool supports(BackendId backend) const override {
        return backend == "cuda";
    }

    std::shared_ptr<const IBackendOptions> decode_options(
        std::span<const std::byte> bytes) const override {
        const auto& source = decode_abi_struct<celeg_cuda_backend_options>(bytes, "CUDA backend");
        return std::make_shared<CudaBackendOptions>(
            cuda_options(source.model), cuda_engine_options(source.engine));
    }

    std::unique_ptr<serve::ServiceBundle> create(
        const BackendCreateRequest& request) const override {
        validate_backend_request(request, id());
        const auto* options = request.options->as<CudaBackendOptions>();
        if (!options) throw std::invalid_argument("CUDA backend options type mismatch");
        auto service = std::make_unique<serve::CudaInferenceService>(
            request.model_path, request.max_context, options->model, options->engine,
            VisualEmbeddingProvider{}, request.runtime);
        return std::make_unique<serve::ServiceBundle>(std::move(service));
    }
};
#endif

#ifdef CELEG_API_WITH_METAL
class MetalBackendOptions final : public IBackendOptions {
public:
    static constexpr std::string_view kSchemaId = "metal/1";

    MetalBackendOptions(MetalModelOptions model_value,
                        MetalEngineOptions engine_value)
        : model(std::move(model_value)), engine(std::move(engine_value)) {}

    BackendId backend_id() const noexcept override { return "metal"; }
    std::string_view schema_id() const noexcept override { return kSchemaId; }
    const void* data() const noexcept override { return this; }

    MetalModelOptions model;
    MetalEngineOptions engine;
};

class MetalBackendFactory final : public IAbiBackendFactory {
public:
    std::string_view id() const override { return "metal"; }
    bool supports(BackendId backend) const override { return backend == "metal"; }

    std::shared_ptr<const IBackendOptions> decode_options(
        std::span<const std::byte> bytes) const override {
        const auto& source = decode_abi_struct<celeg_metal_backend_options>(bytes, "Metal backend");
        return std::make_shared<MetalBackendOptions>(
            metal_options(source.model), metal_engine_options(source.engine));
    }

    std::unique_ptr<serve::ServiceBundle> create(
        const BackendCreateRequest& request) const override {
        validate_backend_request(request, id());
        const auto* options = request.options->as<MetalBackendOptions>();
        if (!options) throw std::invalid_argument("Metal backend options type mismatch");
        auto service = std::make_unique<serve::MetalInferenceService>(
            request.model_path, request.max_context, options->model, options->engine,
            request.runtime);
        return std::make_unique<serve::ServiceBundle>(std::move(service));
    }
};
#endif

}

std::unique_ptr<celeg::serve::ServiceBundle> create_service_bundle(
    const char* path, const celeg_engine_options& options) {
    if (!path || !*path || !options.backend_id || !*options.backend_id) {
        throw std::invalid_argument("engine path and backend id are required");
    }
    if (!options.backend_options || options.backend_options_size == 0) {
        throw std::invalid_argument("backend options are required");
    }
    RuntimeBuilder builder;
    builder.add_builtins();
    builder.add_backend_factory(std::make_unique<CpuBackendFactory>());
#ifdef CELEG_API_WITH_CUDA
    builder.add_backend_factory(std::make_unique<CudaBackendFactory>());
#endif
#ifdef CELEG_API_WITH_METAL
    builder.add_backend_factory(std::make_unique<MetalBackendFactory>());
#endif
    const std::shared_ptr<const RuntimeContext> runtime = builder.build_shared();
    const IBackendFactory* selected = nullptr;
    try {
        selected = &runtime->backends().select_best_if(
            [&options](const IBackendFactory& candidate) {
                return candidate.supports(options.backend_id);
            });
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("unknown backend: " +
                                    std::string(options.backend_id));
    }
    const IBackendFactory& factory = *selected;
    const auto* decoder = dynamic_cast<const IAbiBackendFactory*>(&factory);
    if (!decoder) throw std::invalid_argument(
        "selected backend does not implement the ABI backend factory capability");
    const auto* raw = static_cast<const std::byte*>(options.backend_options);
    BackendCreateRequest request;
    request.model_path = path;
    request.max_context = options.max_context;
    request.backend_id = options.backend_id;
    request.runtime = runtime;
    request.options = decoder->decode_options(
        std::span<const std::byte>(raw, options.backend_options_size));
    return factory.create(request);
}

}

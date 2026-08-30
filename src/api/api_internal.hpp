#pragma once

#include "celeg/api.h"

#include "celeg/backend/cpu/concurrent.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/numa.hpp"
#include "celeg/backend/cpu/topology.hpp"
#include "celeg/serve/inference_service.hpp"
#include "celeg/text/tokenizer.hpp"
#ifdef CELEG_API_WITH_CUDA
#include "runtime_types.hpp"
#include "backend/cuda/concurrency.hpp"
#endif
#ifdef CELEG_API_WITH_METAL
#include "celeg/backend/metal/runtime_types.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

struct celeg_model {
    celeg_backend backend = CELEG_BACKEND_CPU;
    std::unique_ptr<celeg::CpuModel> cpu;
    std::string error;
};

struct celeg_engine {
    std::unique_ptr<celeg::serve::ServiceBundle> service;
    std::string error;
};

struct celeg_tokenizer {
    std::unique_ptr<celeg::ITokenizer> value;
    std::string error;
};

namespace celeg::api {

extern thread_local std::string global_error;

template <typename Handle, typename Function>
celeg_status protect(Handle* handle, Function&& function) noexcept {
    if (!handle) return CELEG_STATUS_INVALID_ARGUMENT;
    try {
        function();
        handle->error.clear();
        return CELEG_STATUS_OK;
    } catch (const std::length_error& error) {
        handle->error = error.what();
        return CELEG_STATUS_BUFFER_TOO_SMALL;
    } catch (const std::out_of_range& error) {
        handle->error = error.what();
        return CELEG_STATUS_NOT_FOUND;
    } catch (const std::invalid_argument& error) {
        handle->error = error.what();
        return CELEG_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        handle->error = error.what();
        return CELEG_STATUS_CELEG_ERROR;
    } catch (...) {
        handle->error = "unknown runtime error";
        return CELEG_STATUS_CELEG_ERROR;
    }
}

void require_size(uint32_t actual, size_t expected, const char* name);

template <typename T>
const T& decode_abi_struct(std::span<const std::byte> bytes,
                           std::string_view backend) {
    if (bytes.size() < sizeof(T)) {
        throw std::invalid_argument(std::string(backend) + " options are truncated");
    }
    const auto* source = reinterpret_cast<const T*>(bytes.data());
    if (source->struct_size < sizeof(T) || source->struct_size > bytes.size()) {
        throw std::invalid_argument(std::string(backend) +
                                    " options have an invalid struct size");
    }
    return *source;
}

void validate_backend_request(const celeg::BackendCreateRequest& request,
                             celeg::BackendId expected);

celeg::GenerationConfig generation(const celeg_generation_options& source);
celeg::CpuModelOptions cpu_options(const celeg_cpu_model_config& source);
celeg::CpuModelOptions cpu_options(const celeg_cpu_model_options& source);
celeg::CpuConcurrentEngineOptions cpu_engine_options(
    const celeg_cpu_engine_options& source);
std::unique_ptr<celeg::serve::ServiceBundle> create_service_bundle(
    const char* path, const celeg_engine_options& options);
celeg_request_status status(celeg::serve::RequestStatus source);

#ifdef CELEG_API_WITH_CUDA
celeg::CudaModelOptions cuda_options(const celeg_cuda_model_options& source);
celeg::ConcurrentEngineOptions cuda_engine_options(
    const celeg_cuda_engine_options& source);
#endif
#ifdef CELEG_API_WITH_METAL
celeg::MetalModelOptions metal_options(const celeg_metal_model_options& source);
celeg::MetalEngineOptions metal_engine_options(const celeg_metal_engine_options& source);
#endif

}

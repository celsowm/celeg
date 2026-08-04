#include "api_internal.hpp"

#include <algorithm>
#include <vector>

extern "C" {

celeg_engine* celeg_engine_create(const char* path,
                                  const celeg_engine_options* options) {
    if (!path || !*path || !options) {
        celeg::api::global_error = "engine path and options are required";
        return nullptr;
    }
    try {
        celeg::api::require_size(options->struct_size, sizeof(*options), "engine options");
        celeg::api::require_size(options->model.struct_size,
                                 sizeof(options->model), "model options");
        if (options->backend != options->model.backend) {
            celeg::api::global_error = "engine backend must match model backend";
            return nullptr;
        }
        const auto runtime = celeg::create_builtin_runtime_context();
        auto result = std::make_unique<celeg_engine>();
        if (options->backend == CELEG_BACKEND_CPU) {
            (void)runtime->backends().select_if(
                [](const celeg::IBackendFactory& factory) {
                    return factory.supports(celeg::BackendKind::Cpu);
                });
            result->service = std::make_unique<celeg::serve::ServiceBundle>(
                std::make_unique<celeg::serve::CpuInferenceService>(
                    path, options->model.max_context,
                    celeg::api::cpu_options(options->model),
                    celeg::api::cpu_engine_options(*options),
                    celeg::VisualEmbeddingProvider{}, runtime));
        } else if (options->backend == CELEG_BACKEND_CUDA) {
#ifdef CELEG_API_WITH_CUDA
            (void)runtime->backends().select_if(
                [](const celeg::IBackendFactory& factory) {
                    return factory.supports(celeg::BackendKind::Cuda);
                });
            result->service = std::make_unique<celeg::serve::ServiceBundle>(
                std::make_unique<celeg::serve::CudaInferenceService>(
                    path, options->model.max_context,
                    celeg::api::cuda_options(options->model),
                    celeg::api::cuda_engine_options(*options),
                    celeg::VisualEmbeddingProvider{}, runtime));
#else
            celeg::api::global_error = "CUDA backend is unavailable in this build";
            return nullptr;
#endif
        } else {
            celeg::api::global_error = "unknown backend";
            return nullptr;
        }
        return result.release();
    } catch (const std::exception& error) {
        celeg::api::global_error = error.what();
        return nullptr;
    }
}

void celeg_engine_destroy(celeg_engine* engine) { delete engine; }

celeg_status celeg_engine_submit(celeg_engine* engine, const int32_t* tokens,
                                 size_t count,
                                 const celeg_request_options* options,
                                 celeg_request_id* request_id) {
    if (!engine || !tokens || count == 0 || !options || !request_id) {
        return CELEG_STATUS_INVALID_ARGUMENT;
    }
    return celeg::api::protect(engine, [&] {
        celeg::api::require_size(options->struct_size, sizeof(*options), "request options");
        celeg::serve::GenerateRequest request;
        request.prompt_tokens.assign(tokens, tokens + count);
        request.max_output_tokens = options->max_new_tokens;
        request.eos_token_ids = {options->eos_token_id};
        request.priority = options->priority;
        request.generation = celeg::api::generation(options->generation);
        *request_id = engine->service->requests().submit(std::move(request));
    });
}

celeg_status celeg_engine_poll(celeg_engine* engine, celeg_request_id id,
                               int32_t* output, size_t capacity, size_t* count,
                               int* finished) {
    if (!engine || !output || capacity == 0 || !count || !finished) {
        return CELEG_STATUS_INVALID_ARGUMENT;
    }
    return celeg::api::protect(engine, [&] {
        const celeg::serve::GenerateEvent event =
            engine->service->requests().poll(id, capacity);
        std::copy(event.tokens.begin(), event.tokens.end(), output);
        *count = event.tokens.size();
        *finished = event.finished ? 1 : 0;
    });
}

celeg_status celeg_engine_status(celeg_engine* engine, celeg_request_id id,
                                  celeg_request_status* value) {
    if (!value) return CELEG_STATUS_INVALID_ARGUMENT;
    return celeg::api::protect(engine, [&] {
        *value = celeg::api::status(engine->service->requests().status(id));
    });
}

celeg_status celeg_engine_cancel(celeg_engine* engine, celeg_request_id id) {
    return celeg::api::protect(engine, [&] {
        if (!engine->service->requests().cancel(id)) {
            throw std::out_of_range("unknown request id");
        }
    });
}

celeg_status celeg_engine_step(celeg_engine* engine, int* progressed) {
    if (!progressed) return CELEG_STATUS_INVALID_ARGUMENT;
    return celeg::api::protect(engine, [&] {
        *progressed = engine->service->scheduler().step() ? 1 : 0;
    });
}

const char* celeg_engine_last_error(const celeg_engine* engine) {
    return engine ? engine->error.c_str() : celeg::api::global_error.c_str();
}

} // extern "C"

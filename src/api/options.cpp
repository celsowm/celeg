#include "api_internal.hpp"
#include "options_defaults.hpp"

#include <cstring>

extern "C" {

void celeg_cpu_model_options_init(celeg_cpu_model_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->max_context = 4096;
    celeg::api::defaults::generation(options->generation);
    celeg::api::defaults::cpu_model_config(options->cpu);
}

void celeg_engine_options_init(celeg_engine_options* options,
                               const char* backend_id,
                               const void* backend_options,
                               uint32_t backend_options_size) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->backend_id = backend_id;
    options->max_context = 4096;
    options->backend_options = backend_options;
    options->backend_options_size = backend_options_size;
    celeg::api::defaults::generation(options->generation);
}

void celeg_cpu_backend_options_init(celeg_cpu_backend_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    celeg::api::defaults::cpu_model_config(options->model);
    celeg::api::defaults::cpu_engine_options(options->engine);
}

void celeg_cuda_backend_options_init(celeg_cuda_backend_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    celeg::api::defaults::cuda_model_options(options->model);
    celeg::api::defaults::cuda_engine_options(options->engine);
}

void celeg_metal_backend_options_init(celeg_metal_backend_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    celeg::api::defaults::metal_model_options(options->model);
    celeg::api::defaults::metal_engine_options(options->engine);
}

void celeg_request_options_init(celeg_request_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->max_new_tokens = 128;
    options->eos_token_id = 7;
    celeg::api::defaults::generation(options->generation);
}

const char* celeg_backend_capabilities(const char* backend_id) {
    if (!backend_id || !*backend_id) return "backend id is required";

    if (std::strcmp(backend_id, "cpu") == 0) {
        celeg::api::global_error = celeg::detect_cpu_capabilities().summary();
        return celeg::api::global_error.c_str();
    }

    if (std::strcmp(backend_id, "cuda") == 0) {
#ifdef CELEG_API_WITH_CUDA
        celeg::api::global_error =
            "CUDA backend available for celeg_engine_*; celeg_model_* remains CPU-only";
        return celeg::api::global_error.c_str();
#else
        return "CUDA backend unavailable in this build";
#endif
    }

    if (std::strcmp(backend_id, "metal") == 0) {
#ifdef CELEG_API_WITH_METAL
        celeg::api::global_error =
            "Metal backend available for celeg_engine_*; celeg_model_* remains CPU-only";
        return celeg::api::global_error.c_str();
#else
        return "Metal backend unavailable in this build";
#endif
    }

    return "unknown backend";
}

}

#include "lfm/api.h"

#include <stdlib.h>
#include <string.h>

int main(void) {
    runtime_cpu_model_options model;
    runtime_cpu_model_options_init(&model);
    if (model.max_context != 4096 || model.cpu.q4_group_size != 32) return 1;
    runtime_engine_options engine;
    runtime_engine_options_init(&engine, RUNTIME_BACKEND_CPU);
    if (engine.backend_options.cpu.max_active_requests != 16) return 1;
    runtime_request_options request;
    runtime_request_options_init(&request);
    if (request.max_new_tokens != 128) return 1;
    if (!runtime_backend_capabilities(RUNTIME_BACKEND_CPU) ||
        strlen(runtime_backend_capabilities(RUNTIME_BACKEND_CPU)) == 0) return 1;
    if (runtime_model_create("", &model) != 0) return 1;

    {
        const char* gguf = getenv("LFM_GGUF_TEST_FILE");
        if (gguf && *gguf) {
            int32_t token = 1;
            runtime_cpu_model_options_init(&model);
            model.max_context = 16;
            runtime_model* instance = runtime_model_create(gguf, &model);
            if (!instance) return 1;
            if (runtime_model_prefill(instance, &token, 1) != RUNTIME_STATUS_OK) {
                runtime_model_destroy(instance);
                return 1;
            }
            runtime_model_destroy(instance);

            runtime_engine_options_init(&engine, RUNTIME_BACKEND_CPU);
            engine.model.max_context = 16;
            engine.backend_options.cpu.max_active_requests = 1;
            runtime_engine* service = runtime_engine_create(gguf, &engine);
            if (!service) return 1;
            runtime_engine_destroy(service);
        }
    }
    return 0;
}

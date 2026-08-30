#include "celeg/api.h"

#include <stdlib.h>
#include <string.h>

int main(void) {
    celeg_cpu_model_options model;
    celeg_cpu_model_options_init(&model);
    if (model.max_context != 4096 || model.cpu.q4_group_size != 32) return 1;
    celeg_cpu_backend_options backend;
    celeg_cpu_backend_options_init(&backend);
    celeg_engine_options engine;
    celeg_engine_options_init(&engine, "cpu", &backend, (uint32_t)sizeof(backend));
    if (backend.engine.max_active_requests != 16) return 1;
    celeg_request_options request;
    celeg_request_options_init(&request);
    if (request.max_new_tokens != 128) return 1;
    if (!celeg_backend_capabilities("cpu") ||
        strlen(celeg_backend_capabilities("cpu")) == 0) return 1;
    if (strcmp(celeg_backend_capabilities("test-only-unknown"),
               "unknown backend") != 0) return 1;
    if (celeg_model_create("", &model) != NULL) return 1;

    {
        celeg_engine_options unknown;
        celeg_engine_options_init(&unknown, "test-only-unknown",
                                  &backend, (uint32_t)sizeof(backend));
        if (celeg_engine_create("missing-model", &unknown) != NULL) return 1;
        if (strstr(celeg_engine_last_error(NULL), "unknown") == NULL) return 1;
    }

    {
        const char* gguf = getenv("CELEG_GGUF_TEST_FILE");
        if (gguf && *gguf) {
            int32_t token = 1;
            celeg_cpu_model_options_init(&model);
            model.max_context = 16;
            celeg_model* instance = celeg_model_create(gguf, &model);
            if (!instance) return 1;
            if (celeg_model_prefill(instance, &token, 1) != CELEG_STATUS_OK) {
                celeg_model_destroy(instance);
                return 1;
            }
            celeg_model_destroy(instance);

            celeg_cpu_backend_options_init(&backend);
            backend.engine.max_active_requests = 1;
            celeg_engine_options_init(&engine, "cpu", &backend,
                                      (uint32_t)sizeof(backend));
            engine.max_context = 16;
            celeg_engine* service = celeg_engine_create(gguf, &engine);
            if (!service) return 1;
            celeg_engine_destroy(service);

            backend.model.prefill_chunk_threshold = 1;
            celeg_engine_options_init(&engine, "cpu", &backend,
                                      (uint32_t)sizeof(backend));
            engine.max_context = 16;
            celeg_engine* configured_service = celeg_engine_create(gguf, &engine);
            if (!configured_service) return 1;
            celeg_engine_destroy(configured_service);
        }
    }
    return 0;
}

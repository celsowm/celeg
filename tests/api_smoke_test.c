#include "lfm/api.h"

#include <stdlib.h>
#include <string.h>

int main(void) {
    lfm25_cpu_model_options model;
    lfm25_cpu_model_options_init(&model);
    if (model.max_context != 4096 || model.cpu.q4_group_size != 32) return 1;
    lfm25_engine_options engine;
    lfm25_engine_options_init(&engine, LFM25_BACKEND_CPU);
    if (engine.backend_options.cpu.max_active_requests != 16) return 1;
    lfm25_request_options request;
    lfm25_request_options_init(&request);
    if (request.max_new_tokens != 128) return 1;
    if (!lfm25_backend_capabilities(LFM25_BACKEND_CPU) ||
        strlen(lfm25_backend_capabilities(LFM25_BACKEND_CPU)) == 0) return 1;
    if (lfm25_model_create("", &model) != 0) return 1;

    {
        const char* gguf = getenv("LFM_GGUF_TEST_FILE");
        if (gguf && *gguf) {
            int32_t token = 1;
            lfm25_cpu_model_options_init(&model);
            model.max_context = 16;
            lfm25_model* instance = lfm25_model_create(gguf, &model);
            if (!instance) return 1;
            if (lfm25_model_prefill(instance, &token, 1) != LFM25_STATUS_OK) {
                lfm25_model_destroy(instance);
                return 1;
            }
            lfm25_model_destroy(instance);

            lfm25_engine_options_init(&engine, LFM25_BACKEND_CPU);
            engine.model.max_context = 16;
            engine.backend_options.cpu.max_active_requests = 1;
            lfm25_engine* service = lfm25_engine_create(gguf, &engine);
            if (!service) return 1;
            lfm25_engine_destroy(service);
        }
    }
    return 0;
}

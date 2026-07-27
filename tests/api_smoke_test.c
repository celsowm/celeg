#include "lfm/api.h"

#include <string.h>

int main(void) {
    lfm25_model_options model;
    lfm25_model_options_init(&model, LFM25_BACKEND_CPU);
    if (model.max_context != 4096 || model.backend != LFM25_BACKEND_CPU) return 1;
    lfm25_engine_options engine;
    lfm25_engine_options_init(&engine, LFM25_BACKEND_CPU);
    if (engine.backend_options.cpu.max_active_requests != 16) return 1;
    lfm25_request_options request;
    lfm25_request_options_init(&request);
    if (request.max_new_tokens != 128) return 1;
    if (!lfm25_backend_capabilities(LFM25_BACKEND_CPU) ||
        strlen(lfm25_backend_capabilities(LFM25_BACKEND_CPU)) == 0) return 1;
    lfm25_model_options_init(&model, LFM25_BACKEND_CUDA);
    if (lfm25_model_create("", &model) != 0) return 1;
    return 0;
}

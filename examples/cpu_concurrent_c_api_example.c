#include "lfm/cpu_c_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.safetensors\n", argv[0]);
        return 2;
    }
    lfm25_cpu_model_options_v5 model;
    lfm25_cpu_model_options_v5_init(&model);
    model.kv_cache_mode = LFM25_CPU_KV_BF16;

    lfm25_cpu_engine_options_v3 engine_options;
    lfm25_cpu_engine_options_v3_init(&engine_options);
    engine_options.max_active_requests = 4;

    lfm25_cpu_engine* engine = lfm25_cpu_engine_create_v3(
        argv[1], &model, &engine_options);
    if (!engine) {
        fprintf(stderr, "engine creation failed\n");
        return 1;
    }

    const int32_t prompt[] = {1, 2, 3};
    lfm25_cpu_request_options_v1 request;
    lfm25_cpu_request_options_init(&request);
    request.max_new_tokens = 4;
    request.generation.temperature = 0.0f;
    request.generation.top_k = 1;

    lfm25_cpu_request_id id = 0;
    if (lfm25_cpu_engine_submit(engine, prompt, 3, &request, &id) !=
        LFM25_CPU_STATUS_OK) {
        fprintf(stderr, "%s\n", lfm25_cpu_engine_last_error(engine));
        lfm25_cpu_engine_destroy(engine);
        return 1;
    }

    int finished = 0;
    while (!finished) {
        int progressed = 0;
        if (lfm25_cpu_engine_step(engine, &progressed) != LFM25_CPU_STATUS_OK) {
            fprintf(stderr, "%s\n", lfm25_cpu_engine_last_error(engine));
            break;
        }
        int32_t output[16];
        size_t count = 0;
        if (lfm25_cpu_engine_poll(engine, id, output, 16, &count, &finished) !=
            LFM25_CPU_STATUS_OK) {
            fprintf(stderr, "%s\n", lfm25_cpu_engine_last_error(engine));
            break;
        }
        for (size_t i = 0; i < count; ++i) printf("%d\n", output[i]);
    }

    lfm25_cpu_engine_destroy(engine);
    return 0;
}

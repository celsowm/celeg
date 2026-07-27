#include "lfm/api.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.safetensors\n", argv[0]);
        return 2;
    }
    lfm25_engine_options options;
    lfm25_engine_options_init(&options, LFM25_BACKEND_CPU);
    lfm25_engine* engine = lfm25_engine_create(argv[1], &options);
    if (!engine) {
        fprintf(stderr, "%s\n", lfm25_engine_last_error(NULL));
        return 1;
    }
    const int32_t prompt[] = {1, 2, 3};
    lfm25_request_options request;
    lfm25_request_options_init(&request);
    lfm25_request_id id = 0;
    if (lfm25_engine_submit(engine, prompt, 3, &request, &id) != LFM25_STATUS_OK) {
        fprintf(stderr, "%s\n", lfm25_engine_last_error(engine));
        lfm25_engine_destroy(engine);
        return 1;
    }
    int finished = 0;
    while (!finished) {
        int progressed = 0;
        if (lfm25_engine_step(engine, &progressed) != LFM25_STATUS_OK) break;
        int32_t tokens[16];
        size_t count = 0;
        if (lfm25_engine_poll(engine, id, tokens, 16, &count, &finished) != LFM25_STATUS_OK) break;
        for (size_t i = 0; i < count; ++i) printf("%d\n", tokens[i]);
    }
    lfm25_engine_destroy(engine);
    return 0;
}

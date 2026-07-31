#include "lfm/api.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.safetensors\n", argv[0]);
        return 2;
    }
    runtime_engine_options options;
    runtime_engine_options_init(&options, RUNTIME_BACKEND_CPU);
    runtime_engine* engine = runtime_engine_create(argv[1], &options);
    if (!engine) {
        fprintf(stderr, "%s\n", runtime_engine_last_error(NULL));
        return 1;
    }
    const int32_t prompt[] = {1, 2, 3};
    runtime_request_options request;
    runtime_request_options_init(&request);
    runtime_request_id id = 0;
    if (runtime_engine_submit(engine, prompt, 3, &request, &id) != RUNTIME_STATUS_OK) {
        fprintf(stderr, "%s\n", runtime_engine_last_error(engine));
        runtime_engine_destroy(engine);
        return 1;
    }
    int finished = 0;
    while (!finished) {
        int progressed = 0;
        if (runtime_engine_step(engine, &progressed) != RUNTIME_STATUS_OK) break;
        int32_t tokens[16];
        size_t count = 0;
        if (runtime_engine_poll(engine, id, tokens, 16, &count, &finished) != RUNTIME_STATUS_OK) break;
        for (size_t i = 0; i < count; ++i) printf("%d\n", tokens[i]);
    }
    runtime_engine_destroy(engine);
    return 0;
}

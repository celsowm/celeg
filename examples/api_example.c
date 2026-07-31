#include "celeg/api.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.safetensors\n", argv[0]);
        return 2;
    }
    celeg_engine_options options;
    celeg_engine_options_init(&options, CELEG_BACKEND_CPU);
    celeg_engine* engine = celeg_engine_create(argv[1], &options);
    if (!engine) {
        fprintf(stderr, "%s\n", celeg_engine_last_error(NULL));
        return 1;
    }
    const int32_t prompt[] = {1, 2, 3};
    celeg_request_options request;
    celeg_request_options_init(&request);
    celeg_request_id id = 0;
    if (celeg_engine_submit(engine, prompt, 3, &request, &id) != CELEG_STATUS_OK) {
        fprintf(stderr, "%s\n", celeg_engine_last_error(engine));
        celeg_engine_destroy(engine);
        return 1;
    }
    int finished = 0;
    while (!finished) {
        int progressed = 0;
        if (celeg_engine_step(engine, &progressed) != CELEG_STATUS_OK) break;
        int32_t tokens[16];
        size_t count = 0;
        if (celeg_engine_poll(engine, id, tokens, 16, &count, &finished) != CELEG_STATUS_OK) break;
        for (size_t i = 0; i < count; ++i) printf("%d\n", tokens[i]);
    }
    celeg_engine_destroy(engine);
    return 0;
}

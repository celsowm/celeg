#include "celeg/api.h"

#include <stdio.h>

static int fail(const celeg_engine* engine) {
    fprintf(stderr, "%s\n", celeg_engine_last_error(engine));
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;

    celeg_metal_backend_options backend;
    celeg_metal_backend_options_init(&backend);
    celeg_engine_options options;
    celeg_engine_options_init(&options, "metal", &backend, (uint32_t)sizeof(backend));
    options.max_context = 128;
    celeg_engine* engine = celeg_engine_create(argv[1], &options);
    if (!engine) return fail(NULL);

    const int32_t prompt[] = {1, 36309};
    celeg_request_options request;
    celeg_request_options_init(&request);
    request.max_new_tokens = 4;
    request.eos_token_id = -1;
    celeg_request_id id = 0;
    if (celeg_engine_submit(engine, prompt, 2, &request, &id) != CELEG_STATUS_OK) {
        const int result = fail(engine);
        celeg_engine_destroy(engine);
        return result;
    }

    int generated = 0;
    int finished = 0;
    for (int iteration = 0; iteration != 32 && !finished; ++iteration) {
        int progressed = 0;
        if (celeg_engine_step(engine, &progressed) != CELEG_STATUS_OK) {
            const int result = fail(engine);
            celeg_engine_destroy(engine);
            return result;
        }
        int32_t tokens[8];
        size_t count = 0;
        if (celeg_engine_poll(engine, id, tokens, 8, &count, &finished) != CELEG_STATUS_OK) {
            const int result = fail(engine);
            celeg_engine_destroy(engine);
            return result;
        }
        generated += (int)count;
        if (!progressed && !finished) {
            celeg_request_status status;
            if (celeg_engine_status(engine, id, &status) != CELEG_STATUS_OK) {
                const int result = fail(engine);
                celeg_engine_destroy(engine);
                return result;
            }
        }
    }

    celeg_request_status status;
    const int valid = finished && generated == 4 &&
                      celeg_engine_status(engine, id, &status) == CELEG_STATUS_OK &&
                      status == CELEG_REQUEST_COMPLETED;
    celeg_engine_destroy(engine);
    if (!valid) return 1;
    printf("metal C API generated %d tokens\n", generated);
    return 0;
}

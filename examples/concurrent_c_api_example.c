#define _POSIX_C_SOURCE 199309L
#include "lfm/api/cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
static void sleep_ms(int milliseconds) { Sleep((DWORD)milliseconds); }
#else
static void sleep_ms(int milliseconds) {
    struct timespec pause;
    pause.tv_sec = milliseconds / 1000;
    pause.tv_nsec = (long)(milliseconds % 1000) * 1000000L;
    nanosleep(&pause, NULL);
}
#endif

static int terminal(int32_t status) {
    return status == LFM25_REQUEST_FINISHED ||
           status == LFM25_REQUEST_CANCELLED ||
           status == LFM25_REQUEST_FAILED;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL_SAFETENSORS TOKENIZER_JSON\n", argv[0]);
        return 2;
    }

    lfm25_tokenizer* tokenizer = lfm25_tokenizer_create(argv[2]);
    if (!tokenizer) {
        fprintf(stderr, "tokenizer: %s\n", lfm25_tokenizer_last_error(NULL));
        return 1;
    }

    const char* prompts[] = {
        "Explique CUDA em uma frase.",
        "Qual é a capital do Brasil?",
        "Escreva uma saudação curta."
    };
    enum { REQUESTS = 3, TOKEN_CAPACITY = 256 };
    int32_t encoded[REQUESTS][TOKEN_CAPACITY];
    size_t encoded_count[REQUESTS] = {0};

    for (int i = 0; i < REQUESTS; ++i) {
        size_t required = 0;
        lfm25_status status = lfm25_tokenizer_encode(
            tokenizer, prompts[i], 1, encoded[i], TOKEN_CAPACITY, &required);
        if (status != LFM25_STATUS_OK) {
            fprintf(stderr, "encode: %s\n", lfm25_tokenizer_last_error(tokenizer));
            lfm25_tokenizer_destroy(tokenizer);
            return 1;
        }
        encoded_count[i] = required;
    }

    lfm25_engine_options_v3 engine_options;
    lfm25_engine_options_v3_init(&engine_options);
    engine_options.max_active_requests = REQUESTS;
    engine_options.max_batched_tokens = 64;
    engine_options.prefill_chunk_tokens = 32;
    engine_options.model.flags |= LFM25_OPTION_FAST_ATTENTION |
                                  LFM25_OPTION_FUSED_PROJECTIONS |
                                  LFM25_OPTION_FUSED_RESIDUALS;

    lfm25_engine* engine = lfm25_engine_create(argv[1], &engine_options);
    if (!engine) {
        fprintf(stderr, "engine: %s\n", lfm25_engine_last_error(NULL));
        lfm25_tokenizer_destroy(tokenizer);
        return 1;
    }

    lfm25_request_options_v2 request_options;
    lfm25_request_options_init(&request_options);
    request_options.max_new_tokens = 48;
    request_options.eos_token = lfm25_tokenizer_eos_id(tokenizer);

    lfm25_request_id ids[REQUESTS];
    for (int i = 0; i < REQUESTS; ++i) {
        request_options.priority = i == 0 ? 10 : 0;
        if (lfm25_engine_submit(engine, encoded[i], encoded_count[i],
                                &request_options, &ids[i]) != LFM25_STATUS_OK) {
            fprintf(stderr, "submit: %s\n", lfm25_engine_last_error(engine));
            lfm25_engine_destroy(engine);
            lfm25_tokenizer_destroy(tokenizer);
            return 1;
        }
    }

    int done[REQUESTS] = {0};
    int done_count = 0;
    while (done_count < REQUESTS) {
        for (int i = 0; i < REQUESTS; ++i) {
            if (done[i]) continue;
            int32_t tokens[16];
            size_t token_count = 0;
            int32_t status = 0;
            int finished = 0;
            if (lfm25_engine_poll(engine, ids[i], tokens, 16, &token_count,
                                  &status, &finished) != LFM25_STATUS_OK) {
                fprintf(stderr, "poll: %s\n", lfm25_engine_last_error(engine));
                done[i] = 1;
                ++done_count;
                continue;
            }
            if (token_count) {
                char text[4096];
                size_t required = 0;
                if (lfm25_tokenizer_decode(tokenizer, tokens, token_count, 1,
                                           text, sizeof(text), &required) ==
                    LFM25_STATUS_OK) {
                    printf("[%d] %s", i, text);
                    fflush(stdout);
                }
            }
            if (finished || terminal(status)) {
                printf("\n[%d] done status=%d\n", i, status);
                done[i] = 1;
                ++done_count;
            }
        }
        sleep_ms(1);
    }

    lfm25_engine_metrics_v2 metrics = {0};
    metrics.struct_size = sizeof(metrics);
    if (lfm25_engine_get_metrics(engine, &metrics) == LFM25_STATUS_OK) {
        printf("requests=%llu decode_tokens=%llu avg_ttft_ms=%.3f avg_itl_ms=%.3f\n",
               (unsigned long long) metrics.completed,
               (unsigned long long) metrics.decoded_tokens,
               metrics.average_ttft_ms, metrics.average_itl_ms);
    }

    lfm25_packed_metrics_v1 packed = {0};
    packed.struct_size = sizeof(packed);
    if (lfm25_engine_get_packed_metrics(engine, &packed) == LFM25_STATUS_OK) {
        printf("packed_tokens=%llu lane_tokens=%llu max_batch=%llu packed_tps=%.3f\n",
               (unsigned long long) packed.packed_decode_tokens,
               (unsigned long long) packed.lane_decode_tokens,
               (unsigned long long) packed.maximum_packed_batch,
               packed.packed_tokens_per_second);
    }

    lfm25_paged_kv_metrics_v1 paged = {0};
    paged.struct_size = sizeof(paged);
    if (lfm25_engine_get_paged_kv_metrics(engine, &paged) == LFM25_STATUS_OK) {
        printf("physical_pages=%llu/%llu kv_bytes=%llu prefix_hits=%llu prefix_misses=%llu\n",
               (unsigned long long) paged.physical_pages_used,
               (unsigned long long) paged.physical_pages_total,
               (unsigned long long) paged.physical_kv_bytes,
               (unsigned long long) paged.prefix_cache_hits,
               (unsigned long long) paged.prefix_cache_misses);
    }

    lfm25_engine_destroy(engine);
    lfm25_tokenizer_destroy(tokenizer);
    return 0;
}

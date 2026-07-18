#include "lfm/c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int print_token(int32_t token, void* user_data) {
    lfm25_tokenizer* tokenizer = (lfm25_tokenizer*)user_data;
    size_t needed = 0;
    lfm25_status status = lfm25_tokenizer_decode(
        tokenizer, &token, 1, 1, NULL, 0, &needed);
    if (status != LFM25_STATUS_BUFFER_TOO_SMALL) return 1;
    char* text = (char*)malloc(needed);
    if (!text) return 1;
    status = lfm25_tokenizer_decode(
        tokenizer, &token, 1, 1, text, needed, &needed);
    if (status == LFM25_STATUS_OK) {
        fputs(text, stdout);
        fflush(stdout);
    }
    free(text);
    return status == LFM25_STATUS_OK ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR PROMPT\n", argv[0]);
        return 2;
    }
    char weights[4096];
    char tokenizer_path[4096];
    snprintf(weights, sizeof(weights), "%s/model.safetensors", argv[1]);
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", argv[1]);

    lfm25_tokenizer* tokenizer = lfm25_tokenizer_create(tokenizer_path);
    if (!tokenizer) {
        fprintf(stderr, "tokenizer: %s\n", lfm25_tokenizer_last_error(NULL));
        return 1;
    }

    size_t token_count = 0;
    lfm25_status status = lfm25_tokenizer_encode(
        tokenizer, argv[2], 1, NULL, 0, &token_count);
    if (status != LFM25_STATUS_BUFFER_TOO_SMALL) {
        fprintf(stderr, "encode: %s\n", lfm25_tokenizer_last_error(tokenizer));
        lfm25_tokenizer_destroy(tokenizer);
        return 1;
    }
    int32_t* tokens = (int32_t*)malloc(token_count * sizeof(int32_t));
    if (!tokens) return 1;
    status = lfm25_tokenizer_encode(
        tokenizer, argv[2], 1, tokens, token_count, &token_count);
    if (status != LFM25_STATUS_OK) {
        fprintf(stderr, "encode: %s\n", lfm25_tokenizer_last_error(tokenizer));
        free(tokens);
        lfm25_tokenizer_destroy(tokenizer);
        return 1;
    }

    lfm25_model_options_v1 model_options;
    lfm25_generation_options_v1 generation;
    lfm25_model_options_init(&model_options);
    lfm25_generation_options_init(&generation);
    model_options.max_context = 4096;
    model_options.weight_mode = LFM25_WEIGHT_INT4;
    model_options.kv_cache_mode = LFM25_KV_INT8;
    model_options.flags |= LFM25_OPTION_FAST_ATTENTION |
                           LFM25_OPTION_FUSED_PROJECTIONS |
                           LFM25_OPTION_FUSED_RESIDUALS;

    lfm25_model* model = lfm25_model_create(weights, &model_options, &generation);
    if (!model) {
        fprintf(stderr, "model: %s\n", lfm25_last_error(NULL));
        free(tokens);
        lfm25_tokenizer_destroy(tokenizer);
        return 1;
    }
    status = lfm25_prefill(model, tokens, token_count);
    free(tokens);
    if (status != LFM25_STATUS_OK) {
        fprintf(stderr, "prefill: %s\n", lfm25_last_error(model));
        lfm25_model_destroy(model);
        lfm25_tokenizer_destroy(tokenizer);
        return 1;
    }

    int32_t generated = 0;
    status = lfm25_generate(
        model, 128, lfm25_tokenizer_eos_id(tokenizer),
        print_token, tokenizer, &generated);
    fputc('\n', stdout);
    if (status != LFM25_STATUS_OK && status != LFM25_STATUS_STOPPED) {
        fprintf(stderr, "generate: %s\n", lfm25_last_error(model));
    }

    lfm25_model_destroy(model);
    lfm25_tokenizer_destroy(tokenizer);
    return status == LFM25_STATUS_OK ? 0 : 1;
}

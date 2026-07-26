#include "lfm/c_api.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    assert(lfm25_api_version() == LFM25_C_API_VERSION);

    lfm25_model_options_v2 model;
    lfm25_generation_options_v1 generation;
    lfm25_model_options_v2_init(&model);
    lfm25_generation_options_init(&generation);
    assert(model.struct_size == sizeof(model));
    assert(generation.struct_size == sizeof(generation));
    assert(model.kv_cache_mode == LFM25_KV_BF16);

    const char* path = "lfm25_c_api_tokenizer_test.json";
    FILE* file = fopen(path, "wb");
    assert(file != NULL);
    fputs("{\"model\":{\"type\":\"BPE\",\"vocab\":{"
          "\"<|startoftext|>\":0,\"h\":1,\"i\":2,\"hi\":3},"
          "\"merges\":[\"h i\"]},\"added_tokens\":[{\"id\":0,"
          "\"content\":\"<|startoftext|>\",\"special\":true}]}", file);
    fclose(file);

    lfm25_tokenizer* tokenizer = lfm25_tokenizer_create(path);
    assert(tokenizer != NULL);
    size_t required = 0;
    assert(lfm25_tokenizer_encode(tokenizer, "hi", 0, NULL, 0, &required) ==
           LFM25_STATUS_BUFFER_TOO_SMALL);
    assert(required == 1);
    int32_t token = -1;
    assert(lfm25_tokenizer_encode(tokenizer, "hi", 0, &token, 1, &required) ==
           LFM25_STATUS_OK);
    assert(token == 3);
    char text[8];
    assert(lfm25_tokenizer_decode(tokenizer, &token, 1, 0, text,
                                  sizeof(text), &required) == LFM25_STATUS_OK);
    assert(strcmp(text, "hi") == 0);
    lfm25_tokenizer_destroy(tokenizer);
    remove(path);
    return 0;
}

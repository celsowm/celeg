#include "lfm/api/cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    printf("detected CPU: %s\n", lfm25_cpu_capabilities());
    printf("topology: %s\n", lfm25_cpu_topology());
    if (argc < 2) {
        printf("usage: %s model.safetensors\n", argv[0]);
        return 0;
    }
    lfm25_cpu_model_options_v6 options;
    lfm25_cpu_generation_options_v1 generation;
    lfm25_cpu_model_options_init(&options);
    lfm25_cpu_generation_options_init(&generation);
    options.max_context = 128;
    options.affinity = LFM25_CPU_AFFINITY_COMPACT;
    options.kv_cache_mode = LFM25_CPU_KV_BF16;
    options.kv_page_tokens = 32;
    options.prefill_chunk_tokens = 256;
    lfm25_cpu_model* model = lfm25_cpu_model_create(argv[1], &options, &generation);
    if (!model) {
        fprintf(stderr, "could not create CPU model\n");
        return 1;
    }
    int32_t vocab = 0;
    if (lfm25_cpu_model_vocab_size(model, &vocab) == LFM25_CPU_STATUS_OK) {
        printf("vocab_size: %d\n", vocab);
    }
    printf("backend: %s\n", lfm25_cpu_model_backend_description(model));
    printf("pack: %s\n", lfm25_cpu_model_pack_path(model));
    lfm25_cpu_model_destroy(model);
    return 0;
}

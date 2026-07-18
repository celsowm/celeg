#include "lfm/cpu_c_api.h"
#include <assert.h>
#include <stddef.h>

int main(void) {
    assert(LFM25_CPU_C_API_VERSION == 5u);
    assert(offsetof(lfm25_cpu_model_options_v1, struct_size) == 0);
    assert(offsetof(lfm25_cpu_model_options_v2, struct_size) == 0);
    assert(offsetof(lfm25_cpu_model_options_v3, struct_size) == 0);
    assert(offsetof(lfm25_cpu_model_options_v4, struct_size) == 0);
    assert(offsetof(lfm25_cpu_model_options_v5, struct_size) == 0);
    assert(sizeof(lfm25_cpu_model_options_v2) > sizeof(lfm25_cpu_model_options_v1));
    assert(sizeof(lfm25_cpu_model_options_v3) > sizeof(lfm25_cpu_model_options_v2));
    assert(sizeof(lfm25_cpu_model_options_v4) > sizeof(lfm25_cpu_model_options_v3));
    assert(sizeof(lfm25_cpu_model_options_v5) > sizeof(lfm25_cpu_model_options_v4));
    assert(sizeof(lfm25_cpu_engine_options_v2) > sizeof(lfm25_cpu_engine_options_v1));
    assert(sizeof(lfm25_cpu_engine_options_v3) > sizeof(lfm25_cpu_engine_options_v2));
    assert(sizeof(lfm25_cpu_memory_stats_v2) > sizeof(lfm25_cpu_memory_stats_v1));
    assert(sizeof(lfm25_cpu_engine_metrics_v2) > sizeof(lfm25_cpu_engine_metrics_v1));
    assert(sizeof(lfm25_cpu_engine_metrics_v3) > sizeof(lfm25_cpu_engine_metrics_v2));
    assert(LFM25_CPU_ISA_SME2 == 10);
    assert(LFM25_CPU_AFFINITY_SCATTER == 2);
    assert(LFM25_CPU_KV_BF16 == 1);
    assert(LFM25_CPU_NUMA_REPLICATE_WEIGHTS == 2);
    assert(LFM25_CPU_REQUEST_FAILED == 5);
    return 0;
}

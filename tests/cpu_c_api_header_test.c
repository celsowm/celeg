#include "lfm/api/cpu.h"
#include <assert.h>
#include <stddef.h>

int main(void) {
    assert(LFM25_CPU_C_API_VERSION == 6u);
    assert(offsetof(lfm25_cpu_model_options_v6, struct_size) == 0);
    assert(offsetof(lfm25_cpu_engine_options_v3, struct_size) == 0);
    assert(offsetof(lfm25_cpu_memory_stats_v2, struct_size) == 0);
    assert(offsetof(lfm25_cpu_engine_metrics_v3, struct_size) == 0);
    assert(offsetof(lfm25_cpu_request_options_v1, struct_size) == 0);
    assert(LFM25_CPU_ISA_SME2 == 10);
    assert(LFM25_CPU_AFFINITY_SCATTER == 2);
    assert(LFM25_CPU_KV_BF16 == 1);
    assert(LFM25_CPU_NUMA_REPLICATE_WEIGHTS == 2);
    assert(LFM25_CPU_REQUEST_FAILED == 5);
    return 0;
}

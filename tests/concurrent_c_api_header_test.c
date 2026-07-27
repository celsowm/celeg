#include "lfm/api/cuda.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

int main(void) {
    lfm25_engine_options_v3 engine = {0};
    lfm25_request_options_v2 request = {0};
    lfm25_engine_metrics_v2 metrics = {0};
    lfm25_packed_metrics_v1 packed = {0};
    lfm25_packed_metrics_v2 packed_v2 = {0};
    lfm25_paged_kv_metrics_v1 paged = {0};
    lfm25_paged_kv_metrics_v2 paged_v2 = {0};
    lfm25_paged_kv_metrics_v3 paged_v3 = {0};
    engine.struct_size = sizeof(engine);
    request.struct_size = sizeof(request);
    metrics.struct_size = sizeof(metrics);
    packed.struct_size = sizeof(packed);
    packed_v2.struct_size = sizeof(packed_v2);
    paged.struct_size = sizeof(paged);
    paged_v2.struct_size = sizeof(paged_v2);
    paged_v3.struct_size = sizeof(paged_v3);
    assert(LFM25_C_API_VERSION == 6u);
    assert(offsetof(lfm25_engine_options_v3, struct_size) == 0);
    assert(sizeof(lfm25_request_id) == sizeof(uint64_t));
    assert(offsetof(lfm25_packed_metrics_v1, struct_size) == 0);
    assert(packed.struct_size == sizeof(lfm25_packed_metrics_v1));
    assert(offsetof(lfm25_packed_metrics_v2, struct_size) == 0);
    assert(packed_v2.struct_size == sizeof(lfm25_packed_metrics_v2));
    assert(offsetof(lfm25_paged_kv_metrics_v1, struct_size) == 0);
    assert(paged.struct_size == sizeof(lfm25_paged_kv_metrics_v1));
    assert(offsetof(lfm25_paged_kv_metrics_v2, struct_size) == 0);
    assert(paged_v2.struct_size == sizeof(lfm25_paged_kv_metrics_v2));
    assert(offsetof(lfm25_paged_kv_metrics_v3, struct_size) == 0);
    assert(paged_v3.struct_size == sizeof(lfm25_paged_kv_metrics_v3));
    printf("concurrent_c_api_header_test: ok\n");
    return 0;
}

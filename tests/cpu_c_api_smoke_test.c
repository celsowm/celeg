#include "lfm/cpu_c_api.h"
#include <assert.h>
#include <string.h>
int main(void) {
    assert(lfm25_cpu_api_version() == 5u);
    const char* isa = lfm25_cpu_detected_isa();
    const char* caps = lfm25_cpu_capabilities();
    assert(isa && strlen(isa) > 0);
    assert(caps && strlen(caps) > 0);
    assert(lfm25_cpu_topology() && strlen(lfm25_cpu_topology()) > 0);
    lfm25_cpu_model_options_v5 options;
    lfm25_cpu_model_options_v5_init(&options);
    assert(options.max_context == 4096);
    assert(options.kv_cache_mode == LFM25_CPU_KV_BF16);
    assert(options.kv_page_tokens == 32);
    assert(options.prefill_chunk_tokens == 256);
    assert(options.prefill_chunk_threshold == 16);
    lfm25_cpu_engine_options_v2 engine;
    lfm25_cpu_engine_options_v2_init(&engine);
    assert(engine.max_active_requests == 16);
    assert(engine.long_prefill_chunk_tokens == 256);
    assert(engine.long_prefill_threshold == 32);
    lfm25_cpu_request_options_v1 request;
    lfm25_cpu_request_options_init(&request);
    assert(request.max_new_tokens == 128);
    return 0;
}

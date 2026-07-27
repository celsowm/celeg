#include "lfm/api/cuda.h"
#include <assert.h>
#include <stddef.h>

int main(void) {
    assert(LFM25_C_API_VERSION == 6u);
    assert(LFM25_WEIGHT_INT4 == 2);
    assert(LFM25_KV_INT8 == 1);
    assert(sizeof(lfm25_model_options_v2) >= 11u * sizeof(uint32_t));
    assert(offsetof(lfm25_model_options_v2, struct_size) == 0);
    return 0;
}

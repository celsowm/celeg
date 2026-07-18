#include "lfm/c_api.h"
#include <assert.h>
#include <stddef.h>

int main(void) {
    assert(LFM25_C_API_VERSION == 6u);
    assert(LFM25_WEIGHT_INT4 == 2);
    assert(LFM25_KV_INT8 == 1);
    assert(sizeof(lfm25_model_options_v1) >= 11u * sizeof(uint32_t));
    assert(offsetof(lfm25_model_options_v1, struct_size) == 0);
    return 0;
}

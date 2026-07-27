#include "lfm/api.h"

#include <stddef.h>

int main(void) {
    if (offsetof(lfm25_model_options, struct_size) != 0) return 1;
    if (offsetof(lfm25_engine_options, struct_size) != 0) return 1;
    if (offsetof(lfm25_request_options, struct_size) != 0) return 1;
    return 0;
}

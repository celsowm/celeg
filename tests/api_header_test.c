#include "lfm/api.h"

#include <stddef.h>

int main(void) {
    if (offsetof(runtime_cpu_model_options, struct_size) != 0) return 1;
    if (offsetof(runtime_engine_options, struct_size) != 0) return 1;
    if (offsetof(runtime_request_options, struct_size) != 0) return 1;
    return 0;
}

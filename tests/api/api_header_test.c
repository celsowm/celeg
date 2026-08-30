#include "celeg/api.h"

#include <stddef.h>

int main(void) {
    if (offsetof(celeg_cpu_model_options, struct_size) != 0) return 1;
    if (offsetof(celeg_engine_options, struct_size) != 0) return 1;
    if (offsetof(celeg_request_options, struct_size) != 0) return 1;
    return 0;
}

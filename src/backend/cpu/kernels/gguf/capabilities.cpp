#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/model/weights/quantization.hpp"

namespace celeg {

bool gguf_type_is_native_dot(GgmlType type) {
    return ggml_type_support(type).cpu_native_dot;
}

bool gguf_type_dequantizable(GgmlType type) {
    return ggml_type_support(type).cpu_dequantize;
}

}

#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/model/weights/quantization.hpp"

namespace celeg {

bool gguf_type_is_native_dot(GgmlType type) {
    switch (type) {
        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q8_0:
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
            return true;
        default:
            return false;
    }
}

bool gguf_type_dequantizable(GgmlType type) {
    return ggml_row_decoder(type).has_value();
}

}

#include "celeg/detail/model/compiled_model.hpp"

#include <stdexcept>

namespace celeg {

void LinearWeight::validate_storage() const {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("linear weight dimensions must be positive");
    }
    switch (kind) {
        case LinearStorageKind::Bf16:
            if (!bf16 || int8 || int4 || scales) {
                throw std::runtime_error("invalid BF16 linear weight storage");
            }
            break;
        case LinearStorageKind::Int8:
            if (!int8 || int4 || !scales) {
                throw std::runtime_error("invalid INT8 linear weight storage");
            }
            break;
        case LinearStorageKind::Int4:
            if (int8 || !int4 || !scales) {
                throw std::runtime_error("invalid INT4 linear weight storage");
            }
            break;
        case LinearStorageKind::Q4_K:
        case LinearStorageKind::Q6_K:
            if (bf16 || int8 || int4 || scales || gguf_segments.empty()) {
                throw std::runtime_error("invalid GGUF-quantized linear weight storage");
            }
            for (const auto& segment : gguf_segments) {
                if (!segment.blocks || segment.rows <= 0 || segment.cols != cols ||
                    (segment.type != GgmlType::Q4_K && segment.type != GgmlType::Q6_K) ||
                    segment.row_bytes == 0) {
                    throw std::runtime_error("invalid GGUF linear segment");
                }
            }
            break;
    }
}

} // namespace celeg

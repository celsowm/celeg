#include "celeg/detail/model/shared_weights.hpp"

namespace celeg {

size_t SharedModelWeights::memory_bytes() const {
    size_t total = 0;
    for (const auto& [unused_name, weight] : tensors) {
        (void)unused_name;
        for (const auto& segment : weight.gguf_segment_storage) {
            total += segment.bytes();
        }
        total += weight.bf16_storage.bytes() +
                 weight.int8_storage.bytes() +
                 weight.int4_storage.bytes() +
                 weight.gguf_expert_storage.bytes() +
                 weight.scales_storage.bytes();
    }
    return total;
}

} // namespace celeg

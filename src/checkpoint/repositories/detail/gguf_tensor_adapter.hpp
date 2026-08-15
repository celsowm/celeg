#pragma once

#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/tensor.hpp"
#include "gguf_tensor_mapper.hpp"

namespace celeg {

class GgufTensorViewAdapter final {
public:
    static HostTensorView adapt(const GgufTensorView& view,
                                bool rows_rope_permuted);
    static HostTensorView adapt_expert(const GgufTensorView& packed,
                                       const GgufTensorReference& reference);
};

}

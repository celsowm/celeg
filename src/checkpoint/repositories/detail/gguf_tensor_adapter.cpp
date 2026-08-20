#include "gguf_tensor_adapter.hpp"

#include <stdexcept>

namespace celeg {
namespace {

HostTensorView host_view(const GgufTensorView& view) {
    HostTensorView result;
    result.shape = view.shape;
    result.data = view.data;
    result.bytes = view.bytes;
    result.block_encoding = block_encoding_from_ggml_type(view.type);
    switch (view.type) {
        case GgmlType::F32: result.dtype = TensorDType::F32; break;
        case GgmlType::F16: result.dtype = TensorDType::F16; break;
        case GgmlType::BF16: result.dtype = TensorDType::BF16; break;
        default: result.dtype = TensorDType::Quantized; break;
    }
    return result;
}

}

HostTensorView GgufTensorViewAdapter::adapt(const GgufTensorView& view) {
    return host_view(view);
}

HostTensorView GgufTensorViewAdapter::adapt_expert(
    const GgufTensorView& packed, const GgufTensorReference& reference) {
    if (!reference.is_expert_slice() || packed.shape.size() != 3 ||
        packed.shape[0] <= 0 || reference.expert >= packed.shape[0] ||
        packed.bytes % static_cast<size_t>(packed.shape[0]) != 0) {
        throw std::runtime_error("invalid packed GGUF expert tensor: " +
                                 reference.native_name);
    }

    HostTensorView result = host_view(packed);
    const size_t expert_bytes = packed.bytes /
        static_cast<size_t>(packed.shape[0]);
    result.shape.erase(result.shape.begin());
    result.data += static_cast<size_t>(reference.expert) * expert_bytes;
    result.bytes = expert_bytes;
    return result;
}

}

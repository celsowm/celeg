#include "detail.hpp"

namespace celeg {

void MetalModel::Impl::encode_dense_feed_forward(
    id<MTLComputeCommandEncoder> encoder, Layer& layer) {
    encode_matvec(encoder, layer.ffn_gate, normed, gate_up, 0);
    encode_matvec(encoder, layer.ffn_up, normed, gate_up,
                  static_cast<NSUInteger>(layer.intermediate) * sizeof(float));
    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    set_buffer(encoder, gate_up, 0);
    set_buffer(encoder, activated, 1);
    set_bytes(encoder, &intermediate, sizeof(intermediate), 2);
    dispatch(encoder, "celeg_swiglu", intermediate);
    encode_matvec(encoder, layer.ffn_down, activated, operation);
}

}


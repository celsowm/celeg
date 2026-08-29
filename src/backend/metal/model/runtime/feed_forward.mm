#include "detail.hpp"

namespace celeg {

void MetalModel::Impl::encode_dense_feed_forward(
    id<MTLComputeCommandEncoder> encoder, Layer& layer) {
    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    if (!encode_matvec_pair(encoder, layer.ffn_gate, layer.ffn_up, normed,
                            gate_up, intermediate)) {
        encode_matvec(encoder, layer.ffn_gate, normed, gate_up, 0);
        encode_matvec(encoder, layer.ffn_up, normed, gate_up,
                      static_cast<NSUInteger>(layer.intermediate) * sizeof(float));
    }
    set_buffer(encoder, gate_up, 0);
    set_buffer(encoder, activated, 1);
    set_bytes(encoder, &intermediate, sizeof(intermediate), 2);
    dispatch(encoder, "celeg_swiglu", intermediate);
    encode_matvec(encoder, layer.ffn_down, activated, operation);
}

void MetalModel::Impl::encode_dense_feed_forward_batch(
    id<MTLComputeCommandEncoder> encoder, Layer& layer, uint32_t rows) {
    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    if (!encode_matmul_pair(encoder, layer.ffn_gate, layer.ffn_up, batch_normed,
                            batch_gate_up, rows, intermediate)) {
        encode_matmul(encoder, layer.ffn_gate, batch_normed, batch_gate_up, rows,
                      0, 0, intermediate * 2);
        encode_matmul(encoder, layer.ffn_up, batch_normed, batch_gate_up, rows,
                      0, static_cast<NSUInteger>(intermediate) * sizeof(float),
                      intermediate * 2);
    }
    encode_swiglu_batch(encoder, batch_gate_up, batch_activated, rows, intermediate);
    encode_matmul(encoder, layer.ffn_down, batch_activated, batch_operation, rows);
}

}

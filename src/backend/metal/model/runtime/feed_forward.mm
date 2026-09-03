#include "detail.hpp"

#include <algorithm>
namespace celeg {

void MetalModel::Impl::encode_dense_feed_forward(
    id<MTLComputeCommandEncoder> encoder, Layer& layer, bool gate_up_ready) {
    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    if (!gate_up_ready) {
        encode_matvec(encoder, layer.ffn_gate, normed, gate_up, 0);
        encode_matvec(encoder, layer.ffn_up, normed, gate_up,
                      static_cast<NSUInteger>(layer.intermediate) * sizeof(float));
    }
    if (!encode_swiglu_matvec(encoder, layer.ffn_down, gate_up, operation)) {
        set_buffer(encoder, gate_up, 0);
        set_buffer(encoder, activated, 1);
        set_bytes(encoder, &intermediate, sizeof(intermediate), 2);
        dispatch(encoder, "celeg_swiglu", intermediate);
        encode_matvec(encoder, layer.ffn_down, activated, operation);
    }
}

void MetalModel::Impl::encode_dense_feed_forward_batch(
    id<MTLComputeCommandEncoder> encoder, Layer& layer, uint32_t rows) {
    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    encode_matmul(encoder, layer.ffn_gate, batch_normed, batch_gate_up, rows,
                  0, 0, intermediate * 2);
    encode_matmul(encoder, layer.ffn_up, batch_normed, batch_gate_up, rows,
                  0, static_cast<NSUInteger>(intermediate) * sizeof(float),
                  intermediate * 2);

    set_buffer(encoder, batch_gate_up, 0);
    set_buffer(encoder, batch_activated, 1);
    set_bytes(encoder, &rows, sizeof(rows), 2);
    set_bytes(encoder, &intermediate, sizeof(intermediate), 3);
    const std::string_view swiglu_kernel =
        options.numerical_policy == MetalNumericalPolicy::Fast
        ? "celeg_swiglu_batch_2d_relaxed"
        : "celeg_swiglu_batch_2d";
    id<MTLComputePipelineState> swiglu = pipeline(swiglu_kernel);
    [encoder setComputePipelineState:swiglu];
    const NSUInteger threads_x = std::min<NSUInteger>(
        intermediate, swiglu.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(intermediate, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads_x, 1, 1)];
    record_dispatch(swiglu_kernel);

    encode_matmul(encoder, layer.ffn_down, batch_activated, batch_operation, rows);
}

}

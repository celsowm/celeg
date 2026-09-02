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
    constexpr NSUInteger kStaticStageRows = 64;
    constexpr NSUInteger kStaticStageTokens = 128;
    constexpr NSUInteger kStaticStageK = 128;
    constexpr NSUInteger kStaticStageThreads = 128;
    constexpr NSUInteger kStaticStageBytes =
        kStaticStageRows * kStaticStageK * sizeof(uint16_t);
    constexpr const char* kStaticStageKernel =
        "celeg_matmul_tensor_q4k_static_stage128";

    const auto encode_aligned_q4k = [&](const Linear& weight,
                                        id<MTLBuffer> input,
                                        id<MTLBuffer> output,
                                        NSUInteger input_offset,
                                        NSUInteger output_offset,
                                        uint32_t output_stride) -> bool {
        if (weight.storage != LinearStorage::Q4K ||
            !tensor_matmul_available(weight.storage, rows) ||
            (rows % kStaticStageTokens) != 0u ||
            (weight.cols % kStaticStageK) != 0u ||
            (weight.rows % kStaticStageRows) != 0u ||
            device.maxThreadgroupMemoryLength < kStaticStageBytes) {
            return false;
        }

        id<MTLComputePipelineState> state = tensor_pipeline(kStaticStageKernel);
        if (state.maxTotalThreadsPerThreadgroup < kStaticStageThreads ||
            state.staticThreadgroupMemoryLength + kStaticStageBytes >
                device.maxThreadgroupMemoryLength) {
            return false;
        }

        set_buffer(encoder, weight.buffer, 0);
        set_buffer(encoder, input, 1, input_offset);
        set_buffer(encoder, output, 2, output_offset);
        set_bytes(encoder, &rows, sizeof(rows), 3);
        set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
        set_bytes(encoder, &weight.rows, sizeof(weight.rows), 5);
        const uint32_t stride = output_stride == 0 ? weight.rows : output_stride;
        set_bytes(encoder, &stride, sizeof(stride), 6);
        set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 7);

        [encoder setComputePipelineState:state];
        [encoder setThreadgroupMemoryLength:kStaticStageBytes atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(
            weight.rows / kStaticStageRows,
            rows / kStaticStageTokens,
            1)
         threadsPerThreadgroup:MTLSizeMake(kStaticStageThreads, 1, 1)];
        record_dispatch(kStaticStageKernel);
        return true;
    };

    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    if (!encode_aligned_q4k(layer.ffn_gate, batch_normed, batch_gate_up,
                            0, 0, intermediate * 2)) {
        encode_matmul(encoder, layer.ffn_gate, batch_normed, batch_gate_up, rows,
                      0, 0, intermediate * 2);
    }
    if (!encode_aligned_q4k(
            layer.ffn_up, batch_normed, batch_gate_up, 0,
            static_cast<NSUInteger>(intermediate) * sizeof(float),
            intermediate * 2)) {
        encode_matmul(encoder, layer.ffn_up, batch_normed, batch_gate_up, rows,
                      0, static_cast<NSUInteger>(intermediate) * sizeof(float),
                      intermediate * 2);
    }

    set_buffer(encoder, batch_gate_up, 0);
    set_buffer(encoder, batch_activated, 1);
    set_bytes(encoder, &rows, sizeof(rows), 2);
    set_bytes(encoder, &intermediate, sizeof(intermediate), 3);
    id<MTLComputePipelineState> swiglu = pipeline("celeg_swiglu_batch_2d");
    [encoder setComputePipelineState:swiglu];
    const NSUInteger threads_x = std::min<NSUInteger>(
        intermediate, swiglu.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(intermediate, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads_x, 1, 1)];
    record_dispatch("celeg_swiglu_batch_2d");

    if (!encode_aligned_q4k(layer.ffn_down, batch_activated, batch_operation,
                            0, 0, 0)) {
        encode_matmul(encoder, layer.ffn_down, batch_activated, batch_operation, rows);
    }
}

}

#include "detail.hpp"

#include <algorithm>
namespace celeg {

void MetalModel::Impl::encode_dense_feed_forward(
    id<MTLComputeCommandEncoder> encoder, Layer& layer, bool gate_up_ready) {
    const auto tag_last_gpu_dispatch = [&](std::string_view role, const Linear& weight) {
        if (!gpu_counter_samples || gpu_counter_dispatches.empty()) return;
        std::string& name = gpu_counter_dispatches.back();
        name += "[";
        name += role;
        name += " ";
        name += std::to_string(weight.cols);
        name += "->";
        name += std::to_string(weight.rows);
        name += " decode]";
        metal_model_detail::record_dispatch_count(name);
    };

    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    if (!gate_up_ready) {
        begin_parallel_group(encoder);
        encode_matvec(encoder, layer.ffn_gate, normed, gate_up, 0);
        tag_last_gpu_dispatch("ffn_gate", layer.ffn_gate);
        encode_matvec(encoder, layer.ffn_up, normed, gate_up,
                      static_cast<NSUInteger>(layer.intermediate) * sizeof(float));
        tag_last_gpu_dispatch("ffn_up", layer.ffn_up);
        end_parallel_group();
    }
    const bool gated_gelu = layer.ffn_activation == ActivationKind::GeluTanh;
    if (!gated_gelu && encode_swiglu_matvec(encoder, layer.ffn_down, gate_up, operation)) {
        tag_last_gpu_dispatch("ffn_down", layer.ffn_down);
    } else {
        set_buffer(encoder, gate_up, 0);
        set_buffer(encoder, activated, 1);
        set_bytes(encoder, &intermediate, sizeof(intermediate), 2);
        dispatch(encoder, gated_gelu ? "celeg_gated_gelu_tanh" : "celeg_swiglu",
                 intermediate);
        encode_matvec(encoder, layer.ffn_down, activated, operation);
        tag_last_gpu_dispatch("ffn_down", layer.ffn_down);
    }
    if (layer.parallel_intermediate > 0) {
        const uint32_t parallel = static_cast<uint32_t>(layer.parallel_intermediate);
        begin_parallel_group(encoder);
        encode_matvec(encoder, layer.ffn_parallel_gate, normed, gate_up, 0);
        tag_last_gpu_dispatch("ffn_parallel_gate", layer.ffn_parallel_gate);
        encode_matvec(encoder, layer.ffn_parallel_up, normed, gate_up,
                      static_cast<NSUInteger>(layer.parallel_intermediate) * sizeof(float));
        tag_last_gpu_dispatch("ffn_parallel_up", layer.ffn_parallel_up);
        end_parallel_group();
        set_buffer(encoder, gate_up, 0);
        set_buffer(encoder, activated, 1);
        set_bytes(encoder, &parallel, sizeof(parallel), 2);
        dispatch(encoder, gated_gelu ? "celeg_gated_gelu_tanh" : "celeg_swiglu",
                 parallel);
        encode_matvec(encoder, layer.ffn_parallel_down, activated, moe_output);
        tag_last_gpu_dispatch("ffn_parallel_down", layer.ffn_parallel_down);
        encode_weighted_add(encoder, moe_output, operation, layer.ffn_down.rows, 1.0f);
    }
}

void MetalModel::Impl::encode_dense_feed_forward_batch(
    id<MTLComputeCommandEncoder> encoder, Layer& layer, uint32_t rows) {
    const auto tag_last_gpu_dispatch = [&](std::string_view role, const Linear& weight) {
        if (!gpu_counter_samples || gpu_counter_dispatches.empty()) return;
        std::string& name = gpu_counter_dispatches.back();
        name += "[";
        name += role;
        name += " ";
        name += std::to_string(weight.cols);
        name += "->";
        name += std::to_string(weight.rows);
        name += " pp";
        name += std::to_string(rows);
        name += "]";
        metal_model_detail::record_dispatch_count(name);
    };

    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    encode_matmul(encoder, layer.ffn_gate, batch_normed, batch_gate_up, rows,
                  0, 0, intermediate * 2);
    tag_last_gpu_dispatch("ffn_gate", layer.ffn_gate);
    encode_matmul(encoder, layer.ffn_up, batch_normed, batch_gate_up, rows,
                  0, static_cast<NSUInteger>(intermediate) * sizeof(float),
                  intermediate * 2);
    tag_last_gpu_dispatch("ffn_up", layer.ffn_up);

    set_buffer(encoder, batch_gate_up, 0);
    set_buffer(encoder, batch_activated, 1);
    set_bytes(encoder, &rows, sizeof(rows), 2);
    set_bytes(encoder, &intermediate, sizeof(intermediate), 3);
    const bool gated_gelu = layer.ffn_activation == ActivationKind::GeluTanh;
    const bool relaxed = options.numerical_policy == MetalNumericalPolicy::Fast;
    const std::string_view activation_kernel =
        gated_gelu
        ? (relaxed ? "celeg_gated_gelu_tanh_batch_2d_relaxed"
                   : "celeg_gated_gelu_tanh_batch_2d")
        : (relaxed ? "celeg_swiglu_batch_2d_relaxed" : "celeg_swiglu_batch_2d");
    id<MTLComputePipelineState> activation = pipeline(activation_kernel);
    encoder = compute_encoder(encoder);
    order_before_dispatch(encoder);
    [encoder setComputePipelineState:activation];
    const NSUInteger threads_x = std::min<NSUInteger>(
        intermediate, activation.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(intermediate, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads_x, 1, 1)];
    record_dispatch(activation_kernel);

    encode_matmul(encoder, layer.ffn_down, batch_activated, batch_operation, rows);
    tag_last_gpu_dispatch("ffn_down", layer.ffn_down);

    if (layer.parallel_intermediate > 0) {
        const uint32_t parallel = static_cast<uint32_t>(layer.parallel_intermediate);
        encode_matmul(encoder, layer.ffn_parallel_gate, batch_normed, batch_gate_up, rows,
                      0, 0, parallel * 2);
        tag_last_gpu_dispatch("ffn_parallel_gate", layer.ffn_parallel_gate);
        encode_matmul(encoder, layer.ffn_parallel_up, batch_normed, batch_gate_up, rows,
                      0, static_cast<NSUInteger>(parallel) * sizeof(float),
                      parallel * 2);
        tag_last_gpu_dispatch("ffn_parallel_up", layer.ffn_parallel_up);

        set_buffer(encoder, batch_gate_up, 0);
        set_buffer(encoder, batch_activated, 1);
        set_bytes(encoder, &rows, sizeof(rows), 2);
        set_bytes(encoder, &parallel, sizeof(parallel), 3);
        const std::string_view parallel_activation_kernel =
            gated_gelu
            ? (relaxed ? "celeg_gated_gelu_tanh_batch_2d_relaxed"
                       : "celeg_gated_gelu_tanh_batch_2d")
            : (relaxed ? "celeg_swiglu_batch_2d_relaxed" : "celeg_swiglu_batch_2d");
        id<MTLComputePipelineState> parallel_activation =
            pipeline(parallel_activation_kernel);
        encoder = compute_encoder(encoder);
        order_before_dispatch(encoder);
        [encoder setComputePipelineState:parallel_activation];
        const NSUInteger parallel_threads_x = std::min<NSUInteger>(
            parallel, parallel_activation.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(parallel, rows, 1)
           threadsPerThreadgroup:MTLSizeMake(parallel_threads_x, 1, 1)];
        record_dispatch(parallel_activation_kernel);

        encode_matmul(encoder, layer.ffn_parallel_down, batch_activated,
                      batch_parallel_output, rows);
        tag_last_gpu_dispatch("ffn_parallel_down", layer.ffn_parallel_down);
        encode_weighted_add(encoder, batch_parallel_output, batch_operation,
                            layer.ffn_down.rows * rows, 1.0f);
    }
}

}

#include "detail.hpp"
#include "quant_registry.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace celeg {

using metal_model_detail::ns_string;

namespace {

/// @brief Whether the device is Apple M5, resolved once per process.
///
/// The device never changes at runtime, so the `matvec_kernel` selection
/// fast path reads this instead of converting `device.name` per dispatch.
bool device_is_apple_m5(id<MTLDevice> device) {
    static const bool m5 = device != nil &&
        ns_string(device.name).find("Apple M5") != std::string::npos;
    return m5;
}

/// @brief Weight-tile geometry; must match the constants in `tensor.metal`.
constexpr NSUInteger kTensorTileRows = 64;
constexpr NSUInteger kTensorTileTokens = 128;
constexpr NSUInteger kTensorTileK = 64;
constexpr NSUInteger kTensorTileThreads = 128;
constexpr NSUInteger kQ4KStrictStageK = 128;
constexpr NSUInteger kTensorTileBytes =
    kTensorTileRows * kTensorTileK * sizeof(uint16_t);
constexpr NSUInteger kQ4KStrictStageBytes =
    kTensorTileRows * kQ4KStrictStageK * sizeof(uint16_t);
constexpr NSUInteger kGpuCounterSampleCapacity = 4096;
constexpr std::string_view kQ4KStrictKernel =
    "celeg_matmul_tensor_q4k_static_stage128";
constexpr std::string_view kF16RelaxedKernel =
    "celeg_matmul_tensor_f16_fast";
constexpr std::string_view kBF16RelaxedKernel =
    "celeg_matmul_tensor_bf16_fast";
constexpr std::string_view kQ40RelaxedKernel =
    "celeg_matmul_tensor_q4_0_relaxed";
constexpr std::string_view kQ4KRelaxedKernel =
    "celeg_matmul_tensor_q4k_relaxed";
constexpr std::string_view kQ5KRelaxedKernel =
    "celeg_matmul_tensor_q5k_relaxed";
constexpr std::string_view kQ6KRelaxedKernel =
    "celeg_matmul_tensor_q6k_fast";
constexpr std::string_view kQ80RelaxedKernel =
    "celeg_matmul_tensor_q8_0_relaxed";
constexpr std::string_view kF16RelaxedN32Kernel =
    "celeg_matmul_tensor_f16_fast_n32";
constexpr std::string_view kBF16RelaxedN32Kernel =
    "celeg_matmul_tensor_bf16_fast_n32";
constexpr std::string_view kQ40RelaxedN32Kernel =
    "celeg_matmul_tensor_q4_0_relaxed_n32";
constexpr std::string_view kQ4KRelaxedN32Kernel =
    "celeg_matmul_tensor_q4k_relaxed_n32";
constexpr std::string_view kQ5KRelaxedN32Kernel =
    "celeg_matmul_tensor_q5k_relaxed_n32";
constexpr std::string_view kQ6KRelaxedN32Kernel =
    "celeg_matmul_tensor_q6k_fast_n32";
constexpr std::string_view kQ6KStrictFastKernel =
    "celeg_matmul_tensor_q6k_fast_strict";
constexpr std::string_view kQ6KStrictFastN32Kernel =
    "celeg_matmul_tensor_q6k_fast_strict_n32";
constexpr std::string_view kQ80RelaxedN32Kernel =
    "celeg_matmul_tensor_q8_0_relaxed_n32";

bool dense_matvec_rows_experiment_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("CELEG_METAL_DENSE_MATVEC_ROWS");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

/// @brief Concurrent decode encoding only when explicitly requested.
///
/// Serial encoding is the default: A/B showed the concurrent encoder's
/// per-dispatch barriers cost ~65 µs/token of CPU with no remaining GPU
/// overlap gain on current kernels (serial wins wall ~3%, GPU tied).
/// Set CELEG_METAL_CONCURRENT_DECODE=1 to re-enable the concurrent encoder
/// with parallel-group overlap (worth re-measuring for slower kernels).
bool concurrent_decode_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("CELEG_METAL_CONCURRENT_DECODE");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}
}

std::optional<std::string_view> MetalModel::Impl::linear_kernel(
    LinearStorage storage, LinearOperationKind operation) const {
    return quant_linear_kernel(storage, operation);
}

MetalMatvecKernel MetalModel::Impl::matvec_kernel(LinearStorage storage,
                                                  uint32_t rows,
                                                  uint32_t cols) const {
    return quant_matvec_kernel(storage, rows, cols, options, device);
}

MetalMatvecKernel MetalModel::Impl::swiglu_matvec_kernel(LinearStorage storage,
                                                         uint32_t rows,
                                                         uint32_t cols) const {
    return quant_swiglu_matvec_kernel(storage, rows, cols, options, device);
}

id<MTLComputePipelineState> MetalModel::Impl::pipeline(std::string_view name) {
    return pipeline_cache.pipeline(name);
}

id<MTLComputePipelineState> MetalModel::Impl::tensor_pipeline(std::string_view name) {
    return pipeline_cache.tensor_pipeline(name);
}

void MetalModel::Impl::encode_matvec(id<MTLComputeCommandEncoder> encoder,
                                     const Linear& weight,
                                     id<MTLBuffer> input,
                                     id<MTLBuffer> output,
                                     NSUInteger output_offset) {
    set_buffer(encoder, weight.buffer, 0);
    set_buffer(encoder, input, 1);
    set_buffer(encoder, output, 2, output_offset);
    set_bytes(encoder, &weight.rows, sizeof(weight.rows), 3);
    set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
    if (weight.row_bytes != 0) {
        set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 5);
    }
    MetalMatvecKernel selected;
    id<MTLComputePipelineState> state;
    if (weight.cached_matvec_path == 1 && weight.cached_matvec_pipeline != nil) {
        selected = weight.cached_matvec_geometry;
        state = weight.cached_matvec_pipeline;
    } else {
        selected = matvec_kernel(weight.storage, weight.rows, weight.cols);
        if (!selected.name) throw std::runtime_error("unsupported Metal matvec binding");
        state = pipeline(selected.name);
        weight.cached_matvec_pipeline = state;
        weight.cached_matvec_geometry = selected;
        weight.cached_matvec_path = 1;
    }
    encoder = compute_encoder(encoder);
    order_before_dispatch(encoder);
    [encoder setComputePipelineState:state];
    if (selected.threadgroup_floats != 0) {
        [encoder setThreadgroupMemoryLength:selected.threadgroup_floats * sizeof(float)
                                    atIndex:0];
    }
    const NSUInteger groups =
        (weight.rows + selected.rows_per_threadgroup - 1u) / selected.rows_per_threadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(selected.threads, 1, 1)];
    record_dispatch(selected.name);
}

bool MetalModel::Impl::encode_swiglu_matvec(
    id<MTLComputeCommandEncoder> encoder, const Linear& weight,
    id<MTLBuffer> gate_up, id<MTLBuffer> output) {
    MetalMatvecKernel selected;
    id<MTLComputePipelineState> state;
    if (weight.cached_matvec_path == 2 && weight.cached_matvec_pipeline != nil) {
        selected = weight.cached_matvec_geometry;
        state = weight.cached_matvec_pipeline;
    } else {
        selected = swiglu_matvec_kernel(weight.storage, weight.rows, weight.cols);
        if (!selected.name) return false;
        state = pipeline(selected.name);
        weight.cached_matvec_pipeline = state;
        weight.cached_matvec_geometry = selected;
        weight.cached_matvec_path = 2;
    }
    set_buffer(encoder, weight.buffer, 0);
    set_buffer(encoder, gate_up, 1);
    set_buffer(encoder, output, 2);
    set_bytes(encoder, &weight.rows, sizeof(weight.rows), 3);
    set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
    set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 5);
    encoder = compute_encoder(encoder);
    order_before_dispatch(encoder);
    [encoder setComputePipelineState:state];
    if (selected.threadgroup_floats != 0) {
        [encoder setThreadgroupMemoryLength:selected.threadgroup_floats * sizeof(float)
                                    atIndex:0];
    }
    const NSUInteger groups =
        (weight.rows + selected.rows_per_threadgroup - 1u) / selected.rows_per_threadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(selected.threads, 1, 1)];
    record_dispatch(selected.name);
    return true;
}

bool MetalModel::Impl::tensor_matmul_available(LinearStorage storage,
                                               uint32_t rows) const {
    if (rows < 16) return false;
    return quant_tensor_matmul_available(storage, pipeline_cache);
}

bool MetalModel::Impl::fast_tensor_matmul_available(LinearStorage storage) const {
    return quant_fast_tensor_matmul_available(storage, pipeline_cache);
}

void MetalModel::Impl::encode_matmul(id<MTLComputeCommandEncoder> encoder,
                                     const Linear& weight, id<MTLBuffer> input,
                                     id<MTLBuffer> output, uint32_t rows,
                                     NSUInteger input_offset, NSUInteger output_offset,
                                     uint32_t output_stride) {
    set_buffer(encoder, weight.buffer, 0);
    set_buffer(encoder, input, 1, input_offset);
    set_buffer(encoder, output, 2, output_offset);
    set_bytes(encoder, &rows, sizeof(rows), 3);
    set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
    set_bytes(encoder, &weight.rows, sizeof(weight.rows), 5);
    const uint32_t stride = output_stride == 0 ? weight.rows : output_stride;
    set_bytes(encoder, &stride, sizeof(stride), 6);
    const bool dense = weight.row_bytes == 0;
    if (!dense) set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 7);
    const bool tensor = tensor_matmul_available(weight.storage, rows);
    if (tensor) {
        const QuantTensorKernel resolved = quant_select_tensor_kernel(
            weight.storage, rows, weight.rows, weight.cols,
            weight.role.value_or(TensorRole::FfnGate), pipeline_cache, options, device);
        std::string_view selected_kernel = resolved.name;
        if (selected_kernel.empty()) {
            throw std::runtime_error("unsupported Metal tensor matmul binding");
        }
        NSUInteger shared_bytes = resolved.shared_bytes ? resolved.shared_bytes : kTensorTileBytes;
        NSUInteger tile_tokens = resolved.tile_tokens ? resolved.tile_tokens : kTensorTileTokens;
        bool exact_groups = resolved.exact_groups;
        bool custom_tensor = resolved.custom;

        id<MTLComputePipelineState> state = tensor_pipeline(selected_kernel);
        if (custom_tensor &&
            (state.maxTotalThreadsPerThreadgroup < kTensorTileThreads ||
             state.staticThreadgroupMemoryLength + shared_bytes > device.maxThreadgroupMemoryLength)) {
            auto fallback = quant_linear_kernel(weight.storage, LinearOperationKind::MatMulTensor);
            if (!fallback) throw std::runtime_error("unsupported Metal tensor matmul binding");
            selected_kernel = *fallback;
            shared_bytes = kTensorTileBytes;
            exact_groups = false;
            state = tensor_pipeline(selected_kernel);
        }

        encoder = compute_encoder(encoder);
        order_before_dispatch(encoder);
        [encoder setComputePipelineState:state];
        [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
        const NSUInteger row_groups = exact_groups
            ? weight.rows / kTensorTileRows
            : (weight.rows + kTensorTileRows - 1u) / kTensorTileRows;
        const NSUInteger token_groups = exact_groups
            ? rows / kTensorTileTokens
            : (rows + tile_tokens - 1u) / tile_tokens;
        [encoder dispatchThreadgroups:MTLSizeMake(row_groups, token_groups, 1)
             threadsPerThreadgroup:MTLSizeMake(kTensorTileThreads, 1, 1)];
        record_dispatch(selected_kernel);
        return;
    }
    const auto kernel = linear_kernel(weight.storage, LinearOperationKind::MatMul);
    if (!kernel) throw std::runtime_error("unsupported Metal matmul binding");
    const NSUInteger groups = (weight.rows + 7u) / 8u;
    id<MTLComputePipelineState> state = pipeline(*kernel);
    encoder = compute_encoder(encoder);
    order_before_dispatch(encoder);
    [encoder setComputePipelineState:state];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, rows, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    record_dispatch(*kernel);
}

void MetalModel::Impl::encode_embedding(id<MTLComputeCommandEncoder> encoder,
                                        uint32_t width, uint32_t token) {
    set_buffer(encoder, embedding.buffer, 0);
    set_buffer(encoder, hidden, 1);
    set_bytes(encoder, &width, sizeof(width), 2);
    set_bytes(encoder, &token, sizeof(token), 3);
    const auto kernel = linear_kernel(embedding.storage,
                                      LinearOperationKind::Embedding);
    if (!kernel) throw std::runtime_error("unsupported Metal embedding binding");
    dispatch(encoder, *kernel, width);
}

void MetalModel::Impl::encode_embedding_batch(
    id<MTLComputeCommandEncoder> encoder, uint32_t rows,
    const std::vector<int32_t>& tokens) {
    const uint32_t width = static_cast<uint32_t>(model.graph.hidden);
    auto* values = static_cast<uint32_t*>(batch_tokens.contents);
    for (uint32_t index = 0; index < rows; ++index) {
        values[index] = static_cast<uint32_t>(tokens[index]);
    }
    set_buffer(encoder, embedding.buffer, 0);
    set_buffer(encoder, batch_hidden, 1);
    set_bytes(encoder, &width, sizeof(width), 2);
    set_buffer(encoder, batch_tokens, 3);
    const auto kernel = linear_kernel(embedding.storage,
                                      LinearOperationKind::EmbeddingBatch);
    if (!kernel) throw std::runtime_error("unsupported Metal embedding batch binding");
    dispatch(encoder, *kernel, static_cast<NSUInteger>(rows) * width);
}

void MetalModel::Impl::encode_rmsnorm(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                                      id<MTLBuffer> weight, id<MTLBuffer> output, uint32_t width,
                                      float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, weight, 1);
    set_buffer(encoder, output, 2);
    set_bytes(encoder, &width, sizeof(width), 3);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 4);
    dispatch_cooperative(encoder, "celeg_rmsnorm", 1);
}

void MetalModel::Impl::encode_rmsnorm_save(id<MTLComputeCommandEncoder> encoder,
                                           id<MTLBuffer> input, id<MTLBuffer> residual,
                                           id<MTLBuffer> weight, id<MTLBuffer> output,
                                           uint32_t width, float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, residual, 1);
    set_buffer(encoder, weight, 2);
    set_buffer(encoder, output, 3);
    set_bytes(encoder, &width, sizeof(width), 4);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 5);
    dispatch_cooperative(encoder, "celeg_rmsnorm_save", 1);
}

void MetalModel::Impl::encode_residual_rmsnorm(
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
    id<MTLBuffer> residual, id<MTLBuffer> weight, id<MTLBuffer> output,
    id<MTLBuffer> normed, uint32_t width, float multiplier, float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, residual, 1);
    set_buffer(encoder, weight, 2);
    set_buffer(encoder, output, 3);
    set_buffer(encoder, normed, 4);
    set_bytes(encoder, &width, sizeof(width), 5);
    set_bytes(encoder, &multiplier, sizeof(multiplier), 6);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 7);
    dispatch_cooperative(encoder, "celeg_residual_rmsnorm", 1);
}

void MetalModel::Impl::encode_residual_rmsnorm_save(
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
    id<MTLBuffer> residual, id<MTLBuffer> weight, id<MTLBuffer> output,
    id<MTLBuffer> next_residual, id<MTLBuffer> normed, uint32_t width,
    float multiplier, float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, residual, 1);
    set_buffer(encoder, weight, 2);
    set_buffer(encoder, output, 3);
    set_buffer(encoder, next_residual, 4);
    set_buffer(encoder, normed, 5);
    set_bytes(encoder, &width, sizeof(width), 6);
    set_bytes(encoder, &multiplier, sizeof(multiplier), 7);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 8);
    dispatch_cooperative(encoder, "celeg_residual_rmsnorm_save", 1);
}

void MetalModel::Impl::encode_rmsnorm_batch(id<MTLComputeCommandEncoder> encoder,
                                            id<MTLBuffer> input, id<MTLBuffer> weight,
                                            id<MTLBuffer> output, uint32_t rows,
                                            uint32_t width, float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, weight, 1);
    set_buffer(encoder, output, 2);
    set_bytes(encoder, &rows, sizeof(rows), 3);
    set_bytes(encoder, &width, sizeof(width), 4);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 5);
    id<MTLComputePipelineState> state = pipeline("celeg_rmsnorm_batch");
    encoder = compute_encoder(encoder);
    order_before_dispatch(encoder);
    [encoder setComputePipelineState:state];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    record_dispatch("celeg_rmsnorm_batch");
}

void MetalModel::Impl::encode_residual_batch(id<MTLComputeCommandEncoder> encoder,
                                             id<MTLBuffer> input, id<MTLBuffer> residual,
                                             id<MTLBuffer> output, uint32_t count,
                                             float multiplier) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, residual, 1);
    set_buffer(encoder, output, 2);
    set_bytes(encoder, &count, sizeof(count), 3);
    set_bytes(encoder, &multiplier, sizeof(multiplier), 4);
    dispatch(encoder, "celeg_residual_batch", count);
}

void MetalModel::Impl::encode_swiglu_batch(id<MTLComputeCommandEncoder> encoder,
                                           id<MTLBuffer> input, id<MTLBuffer> output,
                                           uint32_t rows, uint32_t width) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, output, 1);
    set_bytes(encoder, &rows, sizeof(rows), 2);
    set_bytes(encoder, &width, sizeof(width), 3);
    dispatch(encoder, "celeg_swiglu_batch", static_cast<NSUInteger>(rows) * width);
}

void MetalModel::Impl::encode_weighted_add(id<MTLComputeCommandEncoder> encoder,
                                           id<MTLBuffer> input, id<MTLBuffer> output,
                                           uint32_t count, float weight) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, output, 1);
    set_bytes(encoder, &count, sizeof(count), 2);
    set_bytes(encoder, &weight, sizeof(weight), 3);
    dispatch(encoder, "celeg_weighted_add", count);
}

}

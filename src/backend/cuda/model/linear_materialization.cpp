#include "linear_materialization.hpp"

#include "celeg/checkpoint/tensor_codec.hpp"
#include "kernels/gguf.cuh"
#include "backend/cuda/weights_loader.hpp"
#include "linear_storage_internal.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

namespace celeg {

std::optional<LinearSource> classify_linear_source(
    const IWeightRepository& repository,
    std::string_view name,
    std::span<const std::int64_t> expected) {
    const std::vector<std::int64_t> expected_vector(expected.begin(), expected.end());
    if (has_packed_int8_matrix(repository, name)) {
        return PackedInt8Source{
            load_packed_int8_matrix(repository, name, expected_vector)};
    }
    if (has_packed_int4_matrix(repository, name)) {
        return PackedInt4Source{
            load_packed_int4_matrix(repository, name, expected_vector)};
    }
    if (has_packed_fp8_matrix(repository, name)) {
        return PackedFp8Source{
            load_packed_fp8_matrix(repository, name, expected_vector)};
    }
    if (has_packed_nvfp4_matrix(repository, name)) {
        return PackedNvfp4Source{
            load_packed_nvfp4_matrix(repository, name, expected_vector)};
    }
    const HostTensorView tensor = repository.tensor(name);
    if (!expected.empty() && !tensor_shape_matches(tensor.shape, expected)) {
        return std::nullopt;
    }
    if (tensor.dtype == TensorDType::Quantized) return GgufSource{tensor};
    if (tensor.dtype == TensorDType::BF16 || tensor.dtype == TensorDType::F16 ||
        tensor.dtype == TensorDType::F32) {
        return DenseSource{tensor};
    }
    return std::nullopt;
}

DeviceWeight materialize_linear(
    const LinearSource& linear_source,
    WeightMode mode,
    std::string_view name,
    int rows,
    int cols,
    CudaMemoryKind memory_kind) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("linear materialization dimensions must be positive");
    }
    DeviceWeight weight(memory_kind);
    weight.shape = {rows, cols};
    std::visit([&](const auto& source) {
        using Source = std::decay_t<decltype(source)>;
        if constexpr (std::is_same_v<Source, PackedInt8Source>) {
            cuda_loader_detail::bind_int8_storage(weight, source.matrix.values,
                                                   source.matrix.scales);
        } else if constexpr (std::is_same_v<Source, PackedInt4Source>) {
            const std::vector<float> values = dequantize_packed_int4(source.matrix);
            std::vector<__nv_bfloat16> dense(values.size());
            for (size_t index = 0; index < values.size(); ++index) {
                dense[index] = __float2bfloat16(values[index]);
            }
            weight.bf16_storage.reset(dense.size());
            CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), dense.data(),
                                  dense.size() * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
            const std::byte* data = reinterpret_cast<const std::byte*>(dense.data());
            if (is_rowwise_quantized_weight_mode(mode)) {
                cuda_loader_detail::quantize_and_bind(
                    weight, data, rows, cols, mode,
                    weight.bf16_storage.data());
            } else {
                weight.linear.storage = Bf16LinearStorage{weight.bf16_storage.data()};
            }
        } else if constexpr (std::is_same_v<Source, PackedFp8Source>) {
            cuda_loader_detail::bind_fp8_storage(weight, source.matrix.values,
                                                  source.matrix.scales);
        } else if constexpr (std::is_same_v<Source, PackedNvfp4Source>) {
            cuda_loader_detail::bind_nvfp4_storage(
                weight, source.matrix.packed, source.matrix.block_scales,
                source.matrix.global_scale, source.matrix.input_global_scale);
        } else if constexpr (std::is_same_v<Source, DenseSource>) {
            const std::size_t count = tensor_element_count(
                source.tensor.shape, name);
            if (source.tensor.dtype == TensorDType::BF16) {
                if (source.tensor.bytes != count * sizeof(__nv_bfloat16)) {
                    throw std::runtime_error(
                        "invalid BF16 linear byte count: " + std::string(name));
                }
                if (is_rowwise_quantized_weight_mode(mode)) {
                    cuda_loader_detail::quantize_and_bind(
                        weight, source.tensor.data, rows, cols, mode);
                } else {
                    weight.bf16_storage.reset(count);
                    CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), source.tensor.data,
                                          source.tensor.bytes, cudaMemcpyHostToDevice));
                    weight.linear.storage = Bf16LinearStorage{weight.bf16_storage.data()};
                }
            } else {
                const std::vector<float> decoded = decode_tensor_f32(
                    source.tensor, source.tensor.shape, name);
                std::vector<__nv_bfloat16> dense(count);
                for (std::size_t index = 0; index < count; ++index) {
                    dense[index] = __float2bfloat16(decoded[index]);
                }
                if (is_rowwise_quantized_weight_mode(mode)) {
                    cuda_loader_detail::quantize_and_bind(
                        weight, reinterpret_cast<const std::byte*>(dense.data()),
                        rows, cols, mode);
                } else {
                    weight.bf16_storage.reset(count);
                    CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), dense.data(),
                                          count * sizeof(__nv_bfloat16),
                                          cudaMemcpyHostToDevice));
                    weight.linear.storage = Bf16LinearStorage{weight.bf16_storage.data()};
                }
            }
        } else if constexpr (std::is_same_v<Source, GgufSource>) {
            const HostTensorView& tensor = source.tensor;
            const GgmlType type = ggml_type_from_block_encoding(tensor.block_encoding);
            const GgmlTypeTrait trait = ggml_type_trait(type);
            if (!ggml_row_decoder(type).has_value() || cols % trait.block_size != 0) {
                throw std::runtime_error(
                    "invalid GGUF linear source: " + std::string(name));
            }
            const std::size_t row_bytes = static_cast<std::size_t>(cols) /
                static_cast<std::size_t>(trait.block_size) * trait.type_size;
            if (tensor.bytes != static_cast<std::size_t>(rows) * row_bytes) {
                throw std::runtime_error(
                    "invalid GGUF linear byte count: " + std::string(name));
            }
            const bool host_dequantization = !cuda_gguf_native_mmq(type);
            if (mode == WeightMode::NativeGguf && !host_dequantization) {
                DeviceBuffer<std::uint8_t> raw(tensor.bytes, memory_kind);
                CELEG_CUDA(cudaMemcpy(raw.data(), tensor.data, tensor.bytes,
                                      cudaMemcpyHostToDevice));
                GgufLinearSegment segment{raw.data(), type, 0, rows,
                                          cols, row_bytes};
                weight.gguf_segment_storage.push_back(std::move(raw));
                weight.linear.storage = GgufLinearStorage{{segment}};
                return;
            }
            if (host_dequantization) {
                std::vector<__nv_bfloat16> host;
                dequantize_gguf_to_bf16(tensor, host);
                weight.bf16_storage.reset(host.size());
                CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), host.data(),
                                      host.size() * sizeof(__nv_bfloat16),
                                      cudaMemcpyHostToDevice));
                if (is_rowwise_quantized_weight_mode(mode)) {
                    cuda_loader_detail::quantize_and_bind(
                        weight, reinterpret_cast<const std::byte*>(host.data()),
                        rows, cols, mode,
                        weight.bf16_storage.data());
                } else {
                    weight.linear.storage = Bf16LinearStorage{weight.bf16_storage.data()};
                }
                return;
            }
            DeviceBuffer<std::uint8_t> raw(tensor.bytes, CudaMemoryKind::Device);
            CELEG_CUDA(cudaMemcpy(raw.data(), tensor.data, tensor.bytes,
                                  cudaMemcpyHostToDevice));
            weight.bf16_storage.reset(static_cast<std::size_t>(rows) * cols);
            launch_gguf_dequant(raw.data(), type, weight.bf16_storage.data(),
                                rows, cols, nullptr);
            CELEG_CUDA(cudaStreamSynchronize(nullptr));
            if (is_rowwise_quantized_weight_mode(mode)) {
                std::vector<__nv_bfloat16> host(weight.bf16_storage.size());
                CELEG_CUDA(cudaMemcpy(host.data(), weight.bf16_storage.data(),
                                      host.size() * sizeof(__nv_bfloat16),
                                      cudaMemcpyDeviceToHost));
                cuda_loader_detail::quantize_and_bind(
                    weight, reinterpret_cast<const std::byte*>(host.data()),
                    rows, cols, mode,
                    weight.bf16_storage.data());
            } else {
                weight.linear.storage = Bf16LinearStorage{weight.bf16_storage.data()};
            }
        }
    }, linear_source);
    return weight;
}

}

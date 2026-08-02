#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include "weight_loader_internal.hpp"

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace celeg {

const LinearWeight* WeightLoader::load_linear_weight(
    const IWeightRepository& repo,
    const std::string& name,
    std::vector<int64_t> expected) {
    if (const auto cached = weights_->tensors.find(name);
        cached != weights_->tensors.end()) {
        if (cached->second.shape != expected) {
            throw std::runtime_error("cached linear shape mismatch for " + name);
        }
        return &cached->second.linear;
    }
    const HostTensorView tensor = repo.tensor(name);
    if (tensor.shape != expected || tensor.shape.size() != 2) {
        throw std::runtime_error("unexpected linear tensor: " + name);
    }
    const int rows = static_cast<int>(tensor.shape[0]);
    const int cols = static_cast<int>(tensor.shape[1]);

    DeviceWeight weight;
    weight.shape = tensor.shape;

    // Native GGUF block-quantized weights: by default they are dequantized to
    // BF16 once here (at load time, off the hot path) so every GEMM runs
    // through the tensor-core cuBLAS/cuBLASLt path. With --weight-mode native
    // the raw blocks are kept on-device in their on-disk super-block layout;
    // the matmul kernels dequantize on the fly, reading ~3x less memory per
    // decode token at the cost of scalar dequant inside the kernel.
    if (tensor.dtype == TensorDType::Quantized) {
        const GgmlType ggml_type = ggml_type_from_block_encoding(tensor.block_encoding);
        if (ggml_type != GgmlType::Q4_K && ggml_type != GgmlType::Q6_K) {
            throw std::runtime_error("unsupported GGUF linear quantization (CUDA supports Q4_K/Q6_K only): " + name);
        }
        const GgmlTypeTrait trait = ggml_type_trait(ggml_type);
        if (cols % trait.block_size != 0) {
            throw std::runtime_error("GGUF linear width is not block-aligned: " + name);
        }
        const size_t expected_bytes =
            static_cast<size_t>(rows) * static_cast<size_t>(cols) /
            trait.block_size * trait.type_size;
        if (tensor.bytes != expected_bytes) {
            throw std::runtime_error(
                "GGUF quantized byte count mismatch for " + name);
        }
        const size_t row_bytes =
            static_cast<size_t>(cols) / trait.block_size * trait.type_size;

        if (weight_mode_ == WeightMode::NativeGguf) {
            DeviceBuffer<uint8_t> raw_blocks(tensor.bytes);
            CELEG_CUDA(cudaMemcpy(raw_blocks.data(), tensor.data,
                                tensor.bytes, cudaMemcpyHostToDevice));
            GgufLinearSegment segment;
            segment.blocks = raw_blocks.data();
            segment.type = ggml_type;
            segment.row_offset = 0;
            segment.rows = rows;
            segment.cols = cols;
            segment.row_bytes = row_bytes;
            weight.gguf_segment_storage.push_back(std::move(raw_blocks));
            weight.linear.gguf_segments.push_back(segment);
            weight.linear.kind =
                ggml_type == GgmlType::Q4_K
                    ? LinearStorageKind::Q4_K
                    : LinearStorageKind::Q6_K;
            weight.linear.rows = rows;
            weight.linear.cols = cols;
            weight.linear.validate_storage();
            auto [it, inserted] =
                weights_->tensors.emplace(name, std::move(weight));
            if (!inserted) throw std::runtime_error("duplicate linear weight: " + name);
            return &it->second.linear;
        }

        DeviceBuffer<uint8_t> raw_blocks(tensor.bytes);
        CELEG_CUDA(cudaMemcpy(raw_blocks.data(), tensor.data,
                            tensor.bytes, cudaMemcpyHostToDevice));
        weight.bf16_storage.reset(static_cast<size_t>(rows) * cols);
        launch_gguf_dequant(raw_blocks.data(), ggml_type,
                           weight.bf16_storage.data(), rows, cols,
                           nullptr);
        CELEG_CUDA(cudaStreamSynchronize(nullptr));

        if (weight_mode_ == WeightMode::Int8 ||
            weight_mode_ == WeightMode::Int4) {
            std::vector<__nv_bfloat16> host_bf16(
                static_cast<size_t>(rows) * cols);
            CELEG_CUDA(cudaMemcpy(host_bf16.data(),
                                weight.bf16_storage.data(),
                                host_bf16.size() * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToHost));
            // Keep the BF16 device buffer as a prefill fallback: the W8A16
            // kernel is a scalar GEMV (m=1 decode), so prefill (m>1) dispatches
            // to BF16 cuBLAS tensor-core GEMM via weight.linear.bf16.
            const std::byte* dense_data =
                reinterpret_cast<const std::byte*>(host_bf16.data());
            const size_t count = static_cast<size_t>(rows) * cols;
            if (weight_mode_ == WeightMode::Int8) {
                Int8RowwisePack pack = quantize_bf16_rows(
                    dense_data, static_cast<size_t>(rows),
                    static_cast<size_t>(cols));
                weight.int8_storage.reset(count);
                weight.scales_storage.reset(pack.scales.size());
                CELEG_CUDA(cudaMemcpy(weight.int8_storage.data(),
                                    pack.values.data(),
                                    pack.values.size() * sizeof(int8_t),
                                    cudaMemcpyHostToDevice));
                CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(),
                                    pack.scales.data(),
                                    pack.scales.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
                weight.linear.kind = LinearStorageKind::Int8;
                weight.linear.int8 = weight.int8_storage.data();
                weight.linear.scales = weight.scales_storage.data();
                weight.linear.bf16 = weight.bf16_storage.data();
            } else {
                Int4RowwisePack pack = quantize_bf16_rows_int4(
                    dense_data, static_cast<size_t>(rows),
                    static_cast<size_t>(cols));
                weight.int4_storage.reset(pack.values.size());
                weight.scales_storage.reset(pack.scales.size());
                CELEG_CUDA(cudaMemcpy(weight.int4_storage.data(),
                                    pack.values.data(),
                                    pack.values.size() * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice));
                CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(),
                                    pack.scales.data(),
                                    pack.scales.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
                weight.linear.kind = LinearStorageKind::Int4;
                weight.linear.int4 = weight.int4_storage.data();
                weight.linear.scales = weight.scales_storage.data();
                weight.linear.bf16 = weight.bf16_storage.data();
            }
            weight.linear.rows = rows;
            weight.linear.cols = cols;
            weight.linear.validate_storage();
            auto [it, inserted] =
                weights_->tensors.emplace(name, std::move(weight));
            if (!inserted) throw std::runtime_error("duplicate linear weight: " + name);
            return &it->second.linear;
        }

        weight.linear.kind = LinearStorageKind::Bf16;
        weight.linear.bf16 = weight.bf16_storage.data();
        weight.linear.rows = rows;
        weight.linear.cols = cols;
        weight.linear.validate_storage();
        auto [it, inserted] =
            weights_->tensors.emplace(name, std::move(weight));
        if (!inserted) throw std::runtime_error("duplicate linear weight: " + name);
        return &it->second.linear;
    }

    if (tensor.dtype != TensorDType::BF16 && tensor.dtype != TensorDType::F16 &&
        tensor.dtype != TensorDType::F32) {
        throw std::runtime_error("unexpected linear tensor dtype: " + name);
    }
    const size_t count = cuda_loader_detail::checked_element_count(tensor.shape);
    const size_t element_bytes = tensor.dtype == TensorDType::F32
        ? sizeof(float)
        : (tensor.dtype == TensorDType::F16 ? sizeof(__half) : sizeof(__nv_bfloat16));
    if (tensor.bytes != count * element_bytes) {
        throw std::runtime_error("invalid linear tensor byte count: " + name);
    }

    std::vector<__nv_bfloat16> dense;
    if (tensor.dtype == TensorDType::F16 || tensor.dtype == TensorDType::F32) {
        dense.resize(count);
        if (tensor.dtype == TensorDType::F16) {
            const __half* src = reinterpret_cast<const __half*>(tensor.data);
            for (size_t i = 0; i < count; ++i) {
                dense[i] = __float2bfloat16(__half2float(src[i]));
            }
        } else {
            const float* src = reinterpret_cast<const float*>(tensor.data);
            for (size_t i = 0; i < count; ++i) dense[i] = __float2bfloat16(src[i]);
        }
    }
    const std::byte* dense_data = tensor.dtype == TensorDType::BF16
        ? tensor.data
        : reinterpret_cast<const std::byte*>(dense.data());

    if (weight_mode_ == WeightMode::Int8) {
        Int8RowwisePack pack = quantize_bf16_rows(
            dense_data, static_cast<size_t>(rows), static_cast<size_t>(cols));
        std::vector<int8_t>& quantized = pack.values;
        std::vector<float>& scales = pack.scales;
        weight.int8_storage.reset(count);
        weight.scales_storage.reset(scales.size());
        CELEG_CUDA(cudaMemcpy(weight.int8_storage.data(), quantized.data(),
                            quantized.size() * sizeof(int8_t),
                            cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                            scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int8;
        weight.linear.int8 = weight.int8_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else if (weight_mode_ == WeightMode::Int4) {
        Int4RowwisePack pack = quantize_bf16_rows_int4(
            dense_data, static_cast<size_t>(rows), static_cast<size_t>(cols));
        weight.int4_storage.reset(pack.values.size());
        weight.scales_storage.reset(pack.scales.size());
        CELEG_CUDA(cudaMemcpy(weight.int4_storage.data(), pack.values.data(),
                            pack.values.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), pack.scales.data(),
                            pack.scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int4;
        weight.linear.int4 = weight.int4_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else {
        weight.bf16_storage.reset(count);
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), dense_data,
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Bf16;
        weight.linear.bf16 = weight.bf16_storage.data();
    }
    weight.linear.rows = rows;
    weight.linear.cols = cols;
    weight.linear.validate_storage();

    auto [it, inserted] = weights_->tensors.emplace(name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate linear weight: " + name);
    return &it->second.linear;
}

const LinearWeight* WeightLoader::load_concat_linear_weight(
    const IWeightRepository& repo,
    const std::string& synthetic_name,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) {
    if (const auto cached = weights_->tensors.find(synthetic_name);
        cached != weights_->tensors.end()) {
        return &cached->second.linear;
    }
    if (parts.empty()) throw std::invalid_argument("concat weight requires parts");
    int64_t common_width = -1;
    int64_t total_rows = 0;
    size_t total_count = 0;
    std::vector<HostTensorView> views;
    views.reserve(parts.size());
    for (const auto& [name, expected] : parts) {
        const HostTensorView tensor = repo.tensor(name);
        if (tensor.shape != expected || tensor.shape.size() != 2) {
            throw std::runtime_error("unexpected concatenated tensor: " + name);
        }
        if (common_width < 0) common_width = tensor.shape[1];
        if (tensor.shape[1] != common_width) {
            throw std::runtime_error("concatenated tensors have different widths");
        }
        if (tensor.dtype == TensorDType::Quantized) {
            views.push_back(tensor);
            total_rows += tensor.shape[0];
            continue;
        }
        if (tensor.dtype != TensorDType::BF16) {
            throw std::runtime_error("unexpected concat tensor dtype: " + name);
        }
        const size_t count = cuda_loader_detail::checked_element_count(tensor.shape);
        if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
            throw std::runtime_error("invalid concatenated tensor bytes: " + name);
        }
        total_count += count;
        total_rows += tensor.shape[0];
        views.push_back(tensor);
    }

    bool any_quantized = false;
    bool all_quantized = true;
    for (const auto& view : views) {
        any_quantized = any_quantized || view.dtype == TensorDType::Quantized;
        all_quantized = all_quantized && view.dtype == TensorDType::Quantized;
    }
    if (any_quantized && !all_quantized) {
        throw std::runtime_error("mixed dense/quantized concat is not supported: " + synthetic_name);
    }

    // GGUF block-quantized concat: by default dequantize every source stream
    // straight into its row-offset slice of one combined BF16 buffer (tensor-
    // core GEMM on BF16 beats the scalar dequantizing GEMV/GEMM kernels).
    // With --weight-mode native, keep each source's raw blocks as a separate
    // GGUF segment so the matmul kernels dequantize on the fly.
    if (views.front().dtype == TensorDType::Quantized) {
        for (const auto& v : views) {
            const GgmlType v_ggml_type = ggml_type_from_block_encoding(v.block_encoding);
            if (v.dtype != TensorDType::Quantized ||
                (v_ggml_type != GgmlType::Q4_K && v_ggml_type != GgmlType::Q6_K)) {
                throw std::runtime_error("mixed dense/unsupported quantized concat is not supported: " + synthetic_name);
            }
        }

        if (weight_mode_ == WeightMode::NativeGguf) {
            DeviceWeight weight;
            weight.shape = {total_rows, common_width};
            int row_offset = 0;
            for (const auto& v : views) {
                const GgmlType v_ggml_type = ggml_type_from_block_encoding(v.block_encoding);
                const GgmlTypeTrait trait = ggml_type_trait(v_ggml_type);
                if (common_width % trait.block_size != 0) {
                    throw std::runtime_error("GGUF concat width is not block-aligned: " + synthetic_name);
                }
                const size_t row_bytes = static_cast<size_t>(common_width / trait.block_size) * trait.type_size;
                const size_t bytes = static_cast<size_t>(v.shape[0]) * row_bytes;
                DeviceBuffer<uint8_t> raw_blocks(bytes);
                CELEG_CUDA(cudaMemcpy(raw_blocks.data(), v.data,
                                    bytes, cudaMemcpyHostToDevice));
                GgufLinearSegment segment;
                segment.blocks = raw_blocks.data();
                segment.type = v_ggml_type;
                segment.row_offset = row_offset;
                segment.rows = static_cast<int>(v.shape[0]);
                segment.cols = static_cast<int>(common_width);
                segment.row_bytes = row_bytes;
                weight.gguf_segment_storage.push_back(std::move(raw_blocks));
                weight.linear.gguf_segments.push_back(segment);
                row_offset += static_cast<int>(v.shape[0]);
            }
            weight.linear.kind = ggml_type_from_block_encoding(views.front().block_encoding) == GgmlType::Q4_K
                ? LinearStorageKind::Q4_K
                : LinearStorageKind::Q6_K;
            weight.linear.rows = static_cast<int>(total_rows);
            weight.linear.cols = static_cast<int>(common_width);
            weight.linear.validate_storage();
            auto [it, inserted] =
                weights_->tensors.emplace(synthetic_name, std::move(weight));
            if (!inserted) throw std::runtime_error("duplicate linear weight: " + synthetic_name);
            return &it->second.linear;
        }

        DeviceWeight weight;
        weight.shape = {total_rows, common_width};
        weight.bf16_storage.reset(static_cast<size_t>(total_rows) * common_width);
        int row_offset = 0;
        for (const auto& v : views) {
            const GgmlType v_ggml_type = ggml_type_from_block_encoding(v.block_encoding);
            const GgmlTypeTrait trait = ggml_type_trait(v_ggml_type);
            if (common_width % trait.block_size != 0) {
                throw std::runtime_error("GGUF concat width is not block-aligned: " + synthetic_name);
            }
            const size_t row_bytes = static_cast<size_t>(common_width / trait.block_size) * trait.type_size;
            const size_t bytes = static_cast<size_t>(v.shape[0]) * row_bytes;
            DeviceBuffer<uint8_t> raw_blocks(bytes);
            CELEG_CUDA(cudaMemcpy(raw_blocks.data(), v.data,
                                bytes, cudaMemcpyHostToDevice));
            launch_gguf_dequant(
                raw_blocks.data(), v_ggml_type,
                weight.bf16_storage.data() +
                    static_cast<size_t>(row_offset) * common_width,
                static_cast<int>(v.shape[0]), static_cast<int>(common_width),
                nullptr);
            CELEG_CUDA(cudaStreamSynchronize(nullptr));
            row_offset += static_cast<int>(v.shape[0]);
        }

        if (weight_mode_ == WeightMode::Int8 ||
            weight_mode_ == WeightMode::Int4) {
            const size_t count = static_cast<size_t>(total_rows) * common_width;
            std::vector<__nv_bfloat16> host_bf16(count);
            CELEG_CUDA(cudaMemcpy(host_bf16.data(),
                                weight.bf16_storage.data(),
                                count * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToHost));
            // Keep BF16 device buffer as prefill fallback (see load_linear_weight).
            const std::byte* dense_data =
                reinterpret_cast<const std::byte*>(host_bf16.data());
            if (weight_mode_ == WeightMode::Int8) {
                Int8RowwisePack pack = quantize_bf16_rows(
                    dense_data, static_cast<size_t>(total_rows),
                    static_cast<size_t>(common_width));
                weight.int8_storage.reset(count);
                weight.scales_storage.reset(pack.scales.size());
                CELEG_CUDA(cudaMemcpy(weight.int8_storage.data(),
                                    pack.values.data(),
                                    pack.values.size() * sizeof(int8_t),
                                    cudaMemcpyHostToDevice));
                CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(),
                                    pack.scales.data(),
                                    pack.scales.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
                weight.linear.kind = LinearStorageKind::Int8;
                weight.linear.int8 = weight.int8_storage.data();
                weight.linear.scales = weight.scales_storage.data();
                weight.linear.bf16 = weight.bf16_storage.data();
            } else {
                Int4RowwisePack pack = quantize_bf16_rows_int4(
                    dense_data, static_cast<size_t>(total_rows),
                    static_cast<size_t>(common_width));
                weight.int4_storage.reset(pack.values.size());
                weight.scales_storage.reset(pack.scales.size());
                CELEG_CUDA(cudaMemcpy(weight.int4_storage.data(),
                                    pack.values.data(),
                                    pack.values.size() * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice));
                CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(),
                                    pack.scales.data(),
                                    pack.scales.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
                weight.linear.kind = LinearStorageKind::Int4;
                weight.linear.int4 = weight.int4_storage.data();
                weight.linear.scales = weight.scales_storage.data();
                weight.linear.bf16 = weight.bf16_storage.data();
            }
            weight.linear.rows = static_cast<int>(total_rows);
            weight.linear.cols = static_cast<int>(common_width);
            weight.linear.validate_storage();
            auto [it, inserted] =
                weights_->tensors.emplace(synthetic_name, std::move(weight));
            if (!inserted) throw std::runtime_error("duplicate linear weight: " + synthetic_name);
            return &it->second.linear;
        }

        weight.linear.kind = LinearStorageKind::Bf16;
        weight.linear.bf16 = weight.bf16_storage.data();
        weight.linear.rows = static_cast<int>(total_rows);
        weight.linear.cols = static_cast<int>(common_width);
        weight.linear.validate_storage();
        auto [it, inserted] =
            weights_->tensors.emplace(synthetic_name, std::move(weight));
        if (!inserted) throw std::runtime_error("duplicate linear weight: " + synthetic_name);
        return &it->second.linear;
    }

    DeviceWeight weight;
    weight.shape = {total_rows, common_width};
    if (weight_mode_ == WeightMode::Int8) {
        std::vector<int8_t> quantized(total_count);
        std::vector<float> scales(static_cast<size_t>(total_rows));
        size_t row_offset = 0;
        size_t quantized_offset = 0;
        for (const auto& view : views) {
            const int rows = static_cast<int>(view.shape[0]);
            const int cols = static_cast<int>(view.shape[1]);
            Int8RowwisePack pack = quantize_bf16_rows(
                view.data, static_cast<size_t>(rows), static_cast<size_t>(cols));
            for (size_t i = 0; i < pack.scales.size(); ++i) {
                scales[row_offset + i] = pack.scales[i];
            }
            for (size_t i = 0; i < pack.values.size(); ++i) {
                quantized[quantized_offset + i] = pack.values[i];
            }
            row_offset += pack.scales.size();
            quantized_offset += pack.values.size();
        }
        weight.int8_storage.reset(total_count);
        weight.scales_storage.reset(scales.size());
        CELEG_CUDA(cudaMemcpy(weight.int8_storage.data(), quantized.data(),
                            quantized.size() * sizeof(int8_t),
                            cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                            scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int8;
        weight.linear.int8 = weight.int8_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else if (weight_mode_ == WeightMode::Int4) {
        std::vector<uint8_t> quantized;
        std::vector<float> scales;
        quantized.reserve(total_count / 2 + 16);
        scales.reserve(static_cast<size_t>(total_rows));
        for (const auto& view : views) {
            const int rows = static_cast<int>(view.shape[0]);
            const int cols = static_cast<int>(view.shape[1]);
            Int4RowwisePack pack = quantize_bf16_rows_int4(
                view.data, static_cast<size_t>(rows), static_cast<size_t>(cols));
            quantized.insert(quantized.end(), pack.values.begin(), pack.values.end());
            scales.insert(scales.end(), pack.scales.begin(), pack.scales.end());
        }
        weight.int4_storage.reset(quantized.size());
        weight.scales_storage.reset(scales.size());
        CELEG_CUDA(cudaMemcpy(weight.int4_storage.data(), quantized.data(),
                            quantized.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                            scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int4;
        weight.linear.int4 = weight.int4_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else {
        weight.bf16_storage.reset(total_count);
        __nv_bfloat16* dest = weight.bf16_storage.data();
        size_t offset = 0;
        for (const auto& view : views) {
            const size_t count = cuda_loader_detail::checked_element_count(view.shape);
            CELEG_CUDA(cudaMemcpy(dest + offset, view.data,
                                count * sizeof(__nv_bfloat16),
                                cudaMemcpyHostToDevice));
            offset += count;
        }
        weight.linear.kind = LinearStorageKind::Bf16;
        weight.linear.bf16 = weight.bf16_storage.data();
    }
    weight.linear.rows = static_cast<int>(total_rows);
    weight.linear.cols = static_cast<int>(common_width);
    weight.linear.validate_storage();

    auto [it, inserted] = weights_->tensors.emplace(synthetic_name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate linear weight: " + synthetic_name);
    return &it->second.linear;
}

} // namespace celeg

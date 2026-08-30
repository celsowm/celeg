#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include "celeg/checkpoint/tensor_codec.hpp"
#include "celeg/checkpoint/packed/int8.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include "celeg/checkpoint/packed/fp8.hpp"
#include "celeg/checkpoint/packed/nvfp4.hpp"
#include "linear_storage_internal.hpp"
#include "linear_materialization.hpp"

#include <cstddef>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace celeg {

const LinearWeight* WeightLoader::commit_linear(std::string_view name,
                                                DeviceWeight&& weight,
                                                int rows, int cols) {
    if (rows <= 0 || cols <= 0 || weight.shape != std::vector<int64_t>{rows, cols}) {
        throw std::runtime_error("final linear weight shape mismatch: " + std::string(name));
    }
    cuda_loader_detail::finish_linear_binding(weight, rows, cols);
    auto [it, inserted] = weights_->tensors.emplace(std::string(name), std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate linear weight: " + std::string(name));
    return &it->second.linear;
}

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
    const WeightMode weight_mode = resolve_weight_mode(name);
    const auto source = classify_linear_source(repo, name, expected);
    if (!source || expected.size() != 2) {
        throw std::runtime_error("unexpected linear tensor: " + name);
    }
    const int rows = static_cast<int>(expected[0]);
    const int cols = static_cast<int>(expected[1]);
    const MaterializationPlan plan = plan_linear_materialization(
        *source, weight_mode, name, rows, cols);
    return commit_linear(name, materialize_linear(plan), rows, cols);
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
    const WeightMode weight_mode = resolve_weight_mode(synthetic_name);
    if (std::all_of(parts.begin(), parts.end(), [&](const auto& part) {
            return has_packed_int8_matrix(repo, part.first);
        })) {
        const int64_t cols = parts.front().second.at(1);
        int64_t rows = 0;
        std::vector<int8_t> values;
        std::vector<float> scales;
        for (const auto& [name, expected] : parts) {
            const PackedInt8Matrix packed = load_packed_int8_matrix(repo, name, expected);
            if (packed.cols != cols) throw std::runtime_error("packed concat width mismatch");
            values.insert(values.end(), packed.values.begin(), packed.values.end());
            scales.insert(scales.end(), packed.scales.begin(), packed.scales.end());
            rows += packed.rows;
        }
        DeviceWeight weight;
        weight.shape = {rows, cols};
        cuda_loader_detail::bind_int8_storage(weight, values, scales);
        return commit_linear(synthetic_name, std::move(weight),
                             static_cast<int>(rows), static_cast<int>(cols));
    }
    if (std::all_of(parts.begin(), parts.end(), [&](const auto& part) {
            return has_packed_fp8_matrix(repo, part.first);
        })) {
        // Per-row (per-output-channel) weight scale, dynamic per-token
        // activation quant shared across all rows: stacking rows from
        // multiple parts needs no scale reconciliation, exactly like the
        // int8 case above.
        const int64_t cols = parts.front().second.at(1);
        int64_t rows = 0;
        std::vector<uint8_t> values;
        std::vector<float> scales;
        for (const auto& [name, expected] : parts) {
            const PackedFp8Matrix packed = load_packed_fp8_matrix(repo, name, expected);
            if (packed.cols != cols) throw std::runtime_error("packed concat width mismatch");
            values.insert(values.end(), packed.values.begin(), packed.values.end());
            scales.insert(scales.end(), packed.scales.begin(), packed.scales.end());
            rows += packed.rows;
        }
        DeviceWeight weight;
        weight.shape = {rows, cols};
        cuda_loader_detail::bind_fp8_storage(weight, values, scales);
        return commit_linear(synthetic_name, std::move(weight),
                             static_cast<int>(rows), static_cast<int>(cols));
    }
    if (std::all_of(parts.begin(), parts.end(), [&](const auto& part) {
            return has_packed_nvfp4_matrix(repo, part.first);
        })) {
        // The per-16-block weight scale varies per row/block, but
        // global_scale and input_global_scale are single per-tensor
        // scalars -- concatenating parts calibrated with different
        // scalars would require re-quantizing one part's fp8 block
        // scales, losing precision the checkpoint doesn't actually need
        // to lose (in practice, a gate/up pair sharing one calibration
        // pass always carries identical scalars; fail loudly rather than
        // silently reconcile if a checkpoint ever violates that).
        const int64_t cols = parts.front().second.at(1);
        int64_t rows = 0;
        std::vector<uint8_t> packed_values;
        std::vector<uint8_t> block_scales;
        std::optional<float> global_scale;
        std::optional<float> input_global_scale;
        for (const auto& [name, expected] : parts) {
            const PackedNvfp4Matrix packed = load_packed_nvfp4_matrix(repo, name, expected);
            if (packed.cols != cols) throw std::runtime_error("packed concat width mismatch");
            if (!global_scale) global_scale = packed.global_scale;
            if (!input_global_scale) input_global_scale = packed.input_global_scale;
            if (packed.global_scale != *global_scale ||
                packed.input_global_scale != *input_global_scale) {
                throw std::runtime_error(
                    "NVFP4 concat parts have mismatched calibration scales: " + name);
            }
            packed_values.insert(packed_values.end(), packed.packed.begin(), packed.packed.end());
            block_scales.insert(block_scales.end(), packed.block_scales.begin(),
                                packed.block_scales.end());
            rows += packed.rows;
        }
        DeviceWeight weight;
        weight.shape = {rows, cols};
        cuda_loader_detail::bind_nvfp4_storage(weight, packed_values, block_scales,
                                               *global_scale, *input_global_scale);
        return commit_linear(synthetic_name, std::move(weight),
                             static_cast<int>(rows), static_cast<int>(cols));
    }
    int64_t common_width = -1;
    int64_t total_rows = 0;
    size_t total_count = 0;
    std::vector<HostTensorView> views;
    views.reserve(parts.size());
    std::vector<std::vector<__nv_bfloat16>> converted_dense;
    converted_dense.reserve(parts.size());
    for (const auto& [name, expected] : parts) {
        const HostTensorView tensor = repo.tensor(name);
        if (!tensor_shape_matches(tensor.shape, expected) || expected.size() != 2) {
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
        if (tensor.dtype != TensorDType::BF16 && tensor.dtype != TensorDType::F16 &&
            tensor.dtype != TensorDType::F32) {
            throw std::runtime_error("unexpected concat tensor dtype: " + name);
        }
        const size_t count = tensor_element_count(tensor.shape, name);
        if (tensor.dtype != TensorDType::BF16) {
            converted_dense.emplace_back(count);
            auto& converted = converted_dense.back();
            const std::vector<float> decoded = decode_tensor_f32(
                tensor, expected, name);
            for (size_t i = 0; i < count; ++i) {
                converted[i] = __float2bfloat16(decoded[i]);
            }
            HostTensorView converted_view = tensor;
            converted_view.dtype = TensorDType::BF16;
            converted_view.data = reinterpret_cast<const std::byte*>(converted.data());
            converted_view.bytes = count * sizeof(__nv_bfloat16);
            views.push_back(std::move(converted_view));
        } else {
            views.push_back(tensor);
        }
        total_count += count;
        total_rows += tensor.shape[0];
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

    if (views.front().dtype == TensorDType::Quantized) {
        bool requires_host_dequantization = false;
        for (const auto& v : views) {
            const GgmlType v_ggml_type = ggml_type_from_block_encoding(v.block_encoding);
            if (v.dtype != TensorDType::Quantized ||
                !ggml_row_decoder(v_ggml_type).has_value()) {
                throw std::runtime_error("mixed dense/unsupported quantized concat is not supported: " +
                                         synthetic_name + " (" + ggml_type_name(v_ggml_type) + ")");
            }
            // One member without an MMQ kernel forces the whole concat onto
            // the host-dequantized path, since the segments must end up in a
            // single storage kind.
            requires_host_dequantization =
                    requires_host_dequantization || !cuda_gguf_native_mmq(v_ggml_type);
        }

        if (requires_host_dequantization) {
            std::vector<__nv_bfloat16> host_bf16(
                static_cast<size_t>(total_rows) * common_width);
            size_t row_offset = 0;
            for (const auto& v : views) {
                std::vector<__nv_bfloat16> decoded;
                dequantize_gguf_to_bf16(v, decoded);
                std::copy(decoded.begin(), decoded.end(), host_bf16.begin() +
                    row_offset * static_cast<size_t>(common_width));
                row_offset += static_cast<size_t>(v.shape[0]);
            }
            DeviceWeight weight;
            weight.shape = {total_rows, common_width};
            weight.bf16_storage.reset(host_bf16.size());
            CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), host_bf16.data(),
                                  host_bf16.size() * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
            if (weight_mode == WeightMode::Int8 || weight_mode == WeightMode::Int4) {
                cuda_loader_detail::quantize_and_bind(
                    weight, reinterpret_cast<const std::byte*>(host_bf16.data()),
                    static_cast<size_t>(total_rows), static_cast<size_t>(common_width),
                    weight_mode, weight.bf16_storage.data());
            } else {
                weight.linear.storage = Bf16LinearStorage{weight.bf16_storage.data()};
            }
            return commit_linear(synthetic_name, std::move(weight),
                                 static_cast<int>(total_rows),
                                 static_cast<int>(common_width));
        }

        if (weight_mode == WeightMode::NativeGguf) {
            DeviceWeight weight;
            weight.shape = {total_rows, common_width};
            weight.linear.storage = GgufLinearStorage{};
            std::vector<GgufLinearSegment>& segments =
                std::get<GgufLinearStorage>(weight.linear.storage).segments;
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
                segments.push_back(segment);
                row_offset += static_cast<int>(v.shape[0]);
            }
            return commit_linear(synthetic_name, std::move(weight),
                                 static_cast<int>(total_rows),
                                 static_cast<int>(common_width));
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

        if (weight_mode == WeightMode::Int8 ||
            weight_mode == WeightMode::Int4) {
            const size_t count = static_cast<size_t>(total_rows) * common_width;
            std::vector<__nv_bfloat16> host_bf16(count);
            CELEG_CUDA(cudaMemcpy(host_bf16.data(),
                                weight.bf16_storage.data(),
                                count * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToHost));
            cuda_loader_detail::quantize_and_bind(
                weight, reinterpret_cast<const std::byte*>(host_bf16.data()),
                static_cast<size_t>(total_rows), static_cast<size_t>(common_width),
                weight_mode, weight.bf16_storage.data());
            return commit_linear(synthetic_name, std::move(weight),
                                 static_cast<int>(total_rows),
                                 static_cast<int>(common_width));
        }

        weight.linear.storage = Bf16LinearStorage{weight.bf16_storage.data()};
        return commit_linear(synthetic_name, std::move(weight),
                             static_cast<int>(total_rows),
                             static_cast<int>(common_width));
    }

    DeviceWeight weight;
    weight.shape = {total_rows, common_width};
    if (weight_mode == WeightMode::Int8) {
        std::vector<int8_t> quantized(total_count);
        std::vector<float> scales(static_cast<size_t>(total_rows));
        size_t row_offset = 0;
        for (const auto& view : views) {
            const size_t rows = static_cast<size_t>(view.shape[0]);
            const size_t cols = static_cast<size_t>(view.shape[1]);
            quantize_bf16_rows_into(
                view.data, rows, cols, quantized, scales, row_offset);
            row_offset += rows;
        }
        cuda_loader_detail::bind_int8_storage(weight, quantized, scales);
    } else if (weight_mode == WeightMode::Int4) {
        const size_t packed_cols =
            (static_cast<size_t>(common_width) + 1) / 2;
        std::vector<uint8_t> quantized(
            static_cast<size_t>(total_rows) * packed_cols);
        std::vector<float> scales(static_cast<size_t>(total_rows));
        size_t row_offset = 0;
        for (const auto& view : views) {
            const size_t rows = static_cast<size_t>(view.shape[0]);
            const size_t cols = static_cast<size_t>(view.shape[1]);
            quantize_bf16_rows_int4_into(
                view.data, rows, cols, quantized, scales, row_offset);
            row_offset += rows;
        }
        cuda_loader_detail::bind_int4_storage(weight, quantized, scales);
    } else {
        weight.bf16_storage.reset(total_count);
        __nv_bfloat16* dest = weight.bf16_storage.data();
        size_t offset = 0;
        for (const auto& view : views) {
            const size_t count = tensor_element_count(view.shape, synthetic_name);
            CELEG_CUDA(cudaMemcpy(dest + offset, view.data,
                                count * sizeof(__nv_bfloat16),
                                cudaMemcpyHostToDevice));
            offset += count;
        }
        weight.linear.storage = Bf16LinearStorage{weight.bf16_storage.data()};
    }
    return commit_linear(synthetic_name, std::move(weight),
                         static_cast<int>(total_rows),
                         static_cast<int>(common_width));
}

}

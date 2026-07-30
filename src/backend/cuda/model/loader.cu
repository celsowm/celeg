#include "lfm/model/weights/loader.hpp"
#include "lfm/model/weights/quantization.hpp"
#include "lfm/runtime/moe/expert_residency.hpp"
#include "lfm/backend/cuda/kernels/gguf.cuh"
#include "lfm/checkpoint/gguf_blocks.hpp"
#include "lfm/checkpoint/tensor_names.hpp"

#include <cstring>
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace lfm {

namespace {

size_t checked_element_count(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("weight shape is empty");
    }
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("weight shape has non-positive dimension");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

// Host-side dequantization of a GGUF Q4_K / Q6_K tensor to row-major BF16. Used
// as a fallback when a concatenation mixes quant formats (e.g. attention q/k/v
// where value projection is Q6_K while query/key are Q4_K) and cannot stay
// packed. The block decode mirrors the device kernel in gguf_kernels.cu.
namespace {
struct Q4KHost { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct Q6KHost { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };

void q4k_decode(const Q4KHost* blk, int col, float& out) {
    const float d = __half2float(blk->d);
    const float dmin = __half2float(blk->dmin);
    const int sub = col >> 5, within = col & 31;
    uint8_t sc, m;
    lfm::gguf_blocks::q4k_scale_min(sub, blk->scales, sc, m);
    const uint8_t* qs = blk->qs + (sub >> 1) * 32;
    const int q = (sub & 1) ? (qs[within] >> 4) : (qs[within] & 0xF);
    out = d * sc * static_cast<float>(q) - dmin * m;
}
void q6k_decode(const Q6KHost* blk, int col, float& out) {
    const float d = __half2float(blk->d);
    const int half = col >> 7, idx = col & 127;
    const int n = idx & 31, grp = idx >> 5;
    const uint8_t* ql = blk->ql + half * 64;
    const uint8_t* qh = blk->qh + half * 32;
    int q;
    if (grp == 0) q = (ql[n] & 0xF) | (((qh[n]) & 3) << 4);
    else if (grp == 1) q = (ql[n + 32] & 0xF) | (((qh[n] >> 2) & 3) << 4);
    else if (grp == 2) q = (ql[n] >> 4) | (((qh[n] >> 4) & 3) << 4);
    else q = (ql[n + 32] >> 4) | (((qh[n] >> 6) & 3) << 4);
    const int is = half * 8 + grp * 2 + (n >> 4);
    out = d * static_cast<float>(blk->scales[is]) * static_cast<float>(q - 32);
}
} // namespace

void dequantize_gguf_to_bf16_impl(const HostTensorView& tensor,
                                  std::vector<__nv_bfloat16>& out) {
    if (tensor.ggml_type != GgmlType::Q4_K && tensor.ggml_type != GgmlType::Q6_K) {
        throw std::runtime_error("unsupported GGUF quantization for CUDA dequantization");
    }
    const int rows = static_cast<int>(tensor.shape[0]);
    const int cols = static_cast<int>(tensor.shape[1]);
    out.resize(static_cast<size_t>(rows) * cols);
    const GgmlTypeTrait trait = ggml_type_trait(tensor.ggml_type);
    const int blocks_per_row = cols / trait.block_size;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * trait.type_size;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_blocks =
            reinterpret_cast<const uint8_t*>(tensor.data) + static_cast<size_t>(r) * row_bytes;
        for (int c = 0; c < cols; ++c) {
            const int b = c / trait.block_size;
            const int within = c % trait.block_size;
            float v = 0.0f;
            if (tensor.ggml_type == GgmlType::Q4_K) {
                q4k_decode(reinterpret_cast<const Q4KHost*>(row_blocks) + b, within, v);
            } else {
                q6k_decode(reinterpret_cast<const Q6KHost*>(row_blocks) + b, within, v);
            }
            out[static_cast<size_t>(r) * cols + c] = __float2bfloat16(v);
        }
    }
}

const __nv_bfloat16* upload_bf16(SharedModelWeights& weights,
                                 const IWeightRepository& repo,
                                 const std::string& name,
                                 const std::vector<int64_t>& expected,
                                 const std::string& cache_key) {
    if (const auto cached = weights.tensors.find(cache_key);
        cached != weights.tensors.end()) {
        if (!expected.empty() && cached->second.shape != expected) {
            throw std::runtime_error("cached weight shape mismatch for " + cache_key);
        }
        return cached->second.bf16_storage.data();
    }
    const HostTensorView tensor = repo.tensor(name);
    if (!expected.empty()) {
        // GGUF convolutional depthwise weights are stored as [hidden, cache]
        // (2D) while the model builder expects [hidden, 1, cache] (3D). The
        // extra unit dimension is implicit, so accept the 2D form.
        bool shape_ok = (tensor.shape == expected);
        if (!shape_ok && tensor.shape.size() == 2 && expected.size() == 3 &&
            tensor.shape[0] == expected[0] &&
            tensor.shape[1] == expected[1] * expected[2] &&
            expected[1] == 1) {
            shape_ok = true;
        }
        if (!shape_ok) {
            throw std::runtime_error("unexpected shape for " + name);
        }
    }
    const size_t count = checked_element_count(tensor.shape);

    DeviceWeight weight;
    weight.shape = tensor.shape;
    weight.bf16_storage.reset(count);

    if (tensor.dtype == TensorDType::BF16) {
        if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
            throw std::runtime_error("invalid BF16 byte count for " + name);
        }
        LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
                            cudaMemcpyHostToDevice));
    } else if (tensor.dtype == TensorDType::F32) {
        if (tensor.bytes != count * sizeof(float)) {
            throw std::runtime_error("invalid F32 byte count for " + name);
        }
        const float* src = reinterpret_cast<const float*>(tensor.data);
        std::vector<__nv_bfloat16> converted(count);
        for (size_t i = 0; i < count; ++i) {
            converted[i] = __float2bfloat16(src[i]);
        }
        LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), converted.data(),
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
    } else if (tensor.dtype == TensorDType::F16) {
        if (tensor.bytes != count * sizeof(__half)) {
            throw std::runtime_error("invalid F16 byte count for " + name);
        }
        const __half* src = reinterpret_cast<const __half*>(tensor.data);
        std::vector<__nv_bfloat16> converted(count);
        for (size_t i = 0; i < count; ++i) {
            converted[i] = __float2bfloat16(__half2float(src[i]));
        }
        LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), converted.data(),
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
    } else {
        throw std::runtime_error(
            "only BF16/F16/F32 source weights are supported; incompatible tensor: " + name);
    }
    auto [it, inserted] = weights.tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate weight: " + cache_key);
    return it->second.bf16_storage.data();
}

} // namespace

void dequantize_gguf_to_bf16(const HostTensorView& tensor,
                             std::vector<__nv_bfloat16>& out) {
    dequantize_gguf_to_bf16_impl(tensor, out);
}

const __nv_bfloat16* WeightLoader::load_weight(
    const IWeightRepository& repo,
    const std::string& name,
    std::vector<int64_t> expected) {
    return upload_bf16(*weights_, repo, name, expected, name);
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
        if (tensor.ggml_type != GgmlType::Q4_K && tensor.ggml_type != GgmlType::Q6_K) {
            throw std::runtime_error("unsupported GGUF linear quantization (CUDA supports Q4_K/Q6_K only): " + name);
        }
        const GgmlTypeTrait trait = ggml_type_trait(tensor.ggml_type);
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
            LFM_CUDA(cudaMemcpy(raw_blocks.data(), tensor.data,
                                tensor.bytes, cudaMemcpyHostToDevice));
            GgufLinearSegment segment;
            segment.blocks = raw_blocks.data();
            segment.type = tensor.ggml_type;
            segment.row_offset = 0;
            segment.rows = rows;
            segment.cols = cols;
            segment.row_bytes = row_bytes;
            weight.gguf_segment_storage.push_back(std::move(raw_blocks));
            weight.linear.gguf_segments.push_back(segment);
            weight.linear.kind =
                tensor.ggml_type == GgmlType::Q4_K
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
        LFM_CUDA(cudaMemcpy(raw_blocks.data(), tensor.data,
                            tensor.bytes, cudaMemcpyHostToDevice));
        weight.bf16_storage.reset(static_cast<size_t>(rows) * cols);
        launch_gguf_dequant(raw_blocks.data(), tensor.ggml_type,
                           weight.bf16_storage.data(), rows, cols,
                           nullptr);
        LFM_CUDA(cudaStreamSynchronize(nullptr));

        if (weight_mode_ == WeightMode::Int8 ||
            weight_mode_ == WeightMode::Int4) {
            std::vector<__nv_bfloat16> host_bf16(
                static_cast<size_t>(rows) * cols);
            LFM_CUDA(cudaMemcpy(host_bf16.data(),
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
                LFM_CUDA(cudaMemcpy(weight.int8_storage.data(),
                                    pack.values.data(),
                                    pack.values.size() * sizeof(int8_t),
                                    cudaMemcpyHostToDevice));
                LFM_CUDA(cudaMemcpy(weight.scales_storage.data(),
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
                LFM_CUDA(cudaMemcpy(weight.int4_storage.data(),
                                    pack.values.data(),
                                    pack.values.size() * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice));
                LFM_CUDA(cudaMemcpy(weight.scales_storage.data(),
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
    const size_t count = checked_element_count(tensor.shape);
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
        LFM_CUDA(cudaMemcpy(weight.int8_storage.data(), quantized.data(),
                            quantized.size() * sizeof(int8_t),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
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
        LFM_CUDA(cudaMemcpy(weight.int4_storage.data(), pack.values.data(),
                            pack.values.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), pack.scales.data(),
                            pack.scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int4;
        weight.linear.int4 = weight.int4_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else {
        weight.bf16_storage.reset(count);
        LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), dense_data,
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
        const size_t count = checked_element_count(tensor.shape);
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
            if (v.dtype != TensorDType::Quantized ||
                (v.ggml_type != GgmlType::Q4_K && v.ggml_type != GgmlType::Q6_K)) {
                throw std::runtime_error("mixed dense/unsupported quantized concat is not supported: " + synthetic_name);
            }
        }

        if (weight_mode_ == WeightMode::NativeGguf) {
            DeviceWeight weight;
            weight.shape = {total_rows, common_width};
            int row_offset = 0;
            for (const auto& v : views) {
                const GgmlTypeTrait trait = ggml_type_trait(v.ggml_type);
                if (common_width % trait.block_size != 0) {
                    throw std::runtime_error("GGUF concat width is not block-aligned: " + synthetic_name);
                }
                const size_t row_bytes = static_cast<size_t>(common_width / trait.block_size) * trait.type_size;
                const size_t bytes = static_cast<size_t>(v.shape[0]) * row_bytes;
                DeviceBuffer<uint8_t> raw_blocks(bytes);
                LFM_CUDA(cudaMemcpy(raw_blocks.data(), v.data,
                                    bytes, cudaMemcpyHostToDevice));
                GgufLinearSegment segment;
                segment.blocks = raw_blocks.data();
                segment.type = v.ggml_type;
                segment.row_offset = row_offset;
                segment.rows = static_cast<int>(v.shape[0]);
                segment.cols = static_cast<int>(common_width);
                segment.row_bytes = row_bytes;
                weight.gguf_segment_storage.push_back(std::move(raw_blocks));
                weight.linear.gguf_segments.push_back(segment);
                row_offset += static_cast<int>(v.shape[0]);
            }
            weight.linear.kind = views.front().ggml_type == GgmlType::Q4_K
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
            const GgmlTypeTrait trait = ggml_type_trait(v.ggml_type);
            if (common_width % trait.block_size != 0) {
                throw std::runtime_error("GGUF concat width is not block-aligned: " + synthetic_name);
            }
            const size_t row_bytes = static_cast<size_t>(common_width / trait.block_size) * trait.type_size;
            const size_t bytes = static_cast<size_t>(v.shape[0]) * row_bytes;
            DeviceBuffer<uint8_t> raw_blocks(bytes);
            LFM_CUDA(cudaMemcpy(raw_blocks.data(), v.data,
                                bytes, cudaMemcpyHostToDevice));
            launch_gguf_dequant(
                raw_blocks.data(), v.ggml_type,
                weight.bf16_storage.data() +
                    static_cast<size_t>(row_offset) * common_width,
                static_cast<int>(v.shape[0]), static_cast<int>(common_width),
                nullptr);
            LFM_CUDA(cudaStreamSynchronize(nullptr));
            row_offset += static_cast<int>(v.shape[0]);
        }

        if (weight_mode_ == WeightMode::Int8 ||
            weight_mode_ == WeightMode::Int4) {
            const size_t count = static_cast<size_t>(total_rows) * common_width;
            std::vector<__nv_bfloat16> host_bf16(count);
            LFM_CUDA(cudaMemcpy(host_bf16.data(),
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
                LFM_CUDA(cudaMemcpy(weight.int8_storage.data(),
                                    pack.values.data(),
                                    pack.values.size() * sizeof(int8_t),
                                    cudaMemcpyHostToDevice));
                LFM_CUDA(cudaMemcpy(weight.scales_storage.data(),
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
                LFM_CUDA(cudaMemcpy(weight.int4_storage.data(),
                                    pack.values.data(),
                                    pack.values.size() * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice));
                LFM_CUDA(cudaMemcpy(weight.scales_storage.data(),
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
        LFM_CUDA(cudaMemcpy(weight.int8_storage.data(), quantized.data(),
                            quantized.size() * sizeof(int8_t),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
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
        LFM_CUDA(cudaMemcpy(weight.int4_storage.data(), quantized.data(),
                            quantized.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
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
            const size_t count = checked_element_count(view.shape);
            LFM_CUDA(cudaMemcpy(dest + offset, view.data,
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

const ExpertLinearWeight* WeightLoader::load_expert_linear_weight(
    const IWeightRepository& repo,
    const std::string& name,
    int experts, int rows_per_expert, int cols) {
    const std::string cache_key = name;
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) {
        return &cached->second;
    }
    if (experts <= 0 || rows_per_expert <= 0 || cols <= 0) {
        throw std::runtime_error("invalid expert weight dimensions for " + name);
    }
    const std::vector<int64_t> expected = {
        static_cast<int64_t>(experts) * rows_per_expert, cols};
    const HostTensorView tensor = repo.tensor(name);
    if (tensor.dtype != TensorDType::BF16) {
        throw std::runtime_error("expert weights must be BF16: " + name);
    }
    if (tensor.shape != expected) {
        throw std::runtime_error("unexpected packed expert shape for " + name);
    }
    const size_t count = checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
        throw std::runtime_error("invalid expert byte count for " + name);
    }

    DeviceWeight weight;
    weight.shape = {experts, rows_per_expert, cols};
    weight.bf16_storage.reset(count);
    LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
                        cudaMemcpyHostToDevice));

    ExpertLinearWeight ew;
    ew.kind = LinearStorageKind::Bf16;
    ew.bf16 = weight.bf16_storage.data();
    ew.experts = experts;
    ew.rows_per_expert = rows_per_expert;
    ew.cols = cols;

    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
    const ExpertLinearWeight& stored = expert_cache_.emplace(cache_key, ew).first->second;
    return &stored;
}

// Packs per-expert gate (w1) and up (w3) tensors into one [experts,
// 2*moe_inter, hidden] buffer. w1 occupies rows [0, moe_inter); w3 occupies
// [moe_inter, 2*moe_inter).
const ExpertLinearWeight* WeightLoader::load_moe_gate_up(
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden) {
    const std::string cache_key = layer_name(layer, "moe.gate_up");
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) {
        return &cached->second;
    }
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE gate_up dimensions for layer " +
                                 std::to_string(layer));
    }
    const size_t rows_per_expert = static_cast<size_t>(2 * moe_intermediate);
    const size_t per_expert = rows_per_expert * static_cast<size_t>(hidden);
    const size_t total = static_cast<size_t>(num_experts) * per_expert;

    DeviceWeight weight;
    weight.shape = {num_experts, static_cast<int>(rows_per_expert), hidden};
    weight.bf16_storage.reset(total);
    __nv_bfloat16* base = weight.bf16_storage.data();

    const size_t moe_inter = static_cast<size_t>(moe_intermediate);
    const size_t hidden_c = static_cast<size_t>(hidden);
    const size_t w_bytes = moe_inter * hidden_c * sizeof(__nv_bfloat16);
    for (int e = 0; e < num_experts; ++e) {
        const std::string w1_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w1.weight");
        const std::string w3_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w3.weight");
        const HostTensorView w1 = repo.tensor(w1_name);
        const HostTensorView w3 = repo.tensor(w3_name);
        if (w1.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w3.shape != std::vector<int64_t>{moe_intermediate, hidden}) {
            throw std::runtime_error("unexpected MoE expert tensor shape for " + w1_name);
        }
        const size_t e_off = static_cast<size_t>(e) * per_expert;
        LFM_CUDA(cudaMemcpy(base + e_off, w1.data, w_bytes, cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(base + e_off + moe_inter * hidden_c, w3.data,
                            w_bytes, cudaMemcpyHostToDevice));
    }

    ExpertLinearWeight ew;
    ew.kind = LinearStorageKind::Bf16;
    ew.bf16 = weight.bf16_storage.data();
    ew.experts = num_experts;
    ew.rows_per_expert = static_cast<int>(rows_per_expert);
    ew.cols = hidden;

    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
    const ExpertLinearWeight& stored = expert_cache_.emplace(cache_key, ew).first->second;
    return &stored;
}

const ExpertLinearWeight* WeightLoader::load_moe_down(
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden) {
    const std::string cache_key = layer_name(layer, "moe.down");
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) {
        return &cached->second;
    }
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE down dimensions for layer " +
                                 std::to_string(layer));
    }
    const size_t rows_per_expert = static_cast<size_t>(hidden);
    const size_t per_expert = rows_per_expert * static_cast<size_t>(moe_intermediate);
    const size_t total = static_cast<size_t>(num_experts) * per_expert;

    DeviceWeight weight;
    weight.shape = {num_experts, hidden, moe_intermediate};
    weight.bf16_storage.reset(total);
    __nv_bfloat16* base = weight.bf16_storage.data();

    const size_t hidden_c = static_cast<size_t>(hidden);
    const size_t moe_inter = static_cast<size_t>(moe_intermediate);
    const size_t w_bytes = hidden_c * moe_inter * sizeof(__nv_bfloat16);
    for (int e = 0; e < num_experts; ++e) {
        const std::string w2_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w2.weight");
        const HostTensorView w2 = repo.tensor(w2_name);
        if (w2.shape != std::vector<int64_t>{hidden, moe_intermediate}) {
            throw std::runtime_error("unexpected MoE expert tensor shape for " + w2_name);
        }
        const size_t e_off = static_cast<size_t>(e) * per_expert;
        LFM_CUDA(cudaMemcpy(base + e_off, w2.data, w_bytes, cudaMemcpyHostToDevice));
    }

    ExpertLinearWeight ew;
    ew.kind = LinearStorageKind::Bf16;
    ew.bf16 = weight.bf16_storage.data();
    ew.experts = num_experts;
    ew.rows_per_expert = hidden;
    ew.cols = moe_intermediate;

    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
    const ExpertLinearWeight& stored = expert_cache_.emplace(cache_key, ew).first->second;
    return &stored;
}

const float* WeightLoader::load_f32_weight(
    const IWeightRepository& repo,
    const std::string& name,
    std::vector<int64_t> expected) {
    if (const auto cached = weights_->tensors.find(name);
        cached != weights_->tensors.end()) {
        if (cached->second.shape != expected) {
            throw std::runtime_error("cached f32 shape mismatch for " + name);
        }
        return cached->second.scales_storage.data();
    }
    const HostTensorView tensor = repo.tensor(name);
    if (tensor.dtype != TensorDType::F32) {
        throw std::runtime_error("expected F32 tensor: " + name);
    }
    if (tensor.shape != expected) {
        throw std::runtime_error("unexpected f32 shape for " + name);
    }
    const size_t count = checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(float)) {
        throw std::runtime_error("invalid f32 byte count for " + name);
    }
    DeviceWeight weight;
    weight.shape = tensor.shape;
    weight.scales_storage.reset(count);
    LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), tensor.data, tensor.bytes,
                        cudaMemcpyHostToDevice));
    auto [it, inserted] = weights_->tensors.emplace(name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate f32 weight: " + name);
    return it->second.scales_storage.data();
}

const LinearWeight* WeightLoader::load_router(
    const IWeightRepository& repo, int layer,
    int num_experts, int hidden) {
    return load_linear_weight(
        repo, layer_name(layer, "feed_forward.gate.weight"),
        {num_experts, hidden});
}

} // namespace lfm



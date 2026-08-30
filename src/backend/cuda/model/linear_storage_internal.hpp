#pragma once

#include "backend/cuda/runtime_types.hpp"
#include "backend/cuda/utils.cuh"
#include "detail/device_weights.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace celeg::cuda_loader_detail {

inline void bind_int8_storage(DeviceWeight& weight,
                              const std::vector<int8_t>& values,
                              const std::vector<float>& scales,
                              const __nv_bfloat16* bf16_fallback = nullptr) {
    weight.int8_storage.reset(values.size());
    weight.scales_storage.reset(scales.size());
    CELEG_CUDA(cudaMemcpy(weight.int8_storage.data(), values.data(),
                          values.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                          scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    weight.linear.storage = Int8LinearStorage{
        weight.int8_storage.data(), weight.scales_storage.data(), bf16_fallback};
}

inline void bind_int4_storage(DeviceWeight& weight,
                              const std::vector<uint8_t>& values,
                              const std::vector<float>& scales,
                              const __nv_bfloat16* bf16_fallback = nullptr) {
    weight.int4_storage.reset(values.size());
    weight.scales_storage.reset(scales.size());
    CELEG_CUDA(cudaMemcpy(weight.int4_storage.data(), values.data(),
                          values.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                          scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    weight.linear.storage = Int4LinearStorage{
        weight.int4_storage.data(), weight.scales_storage.data(), bf16_fallback};
}

inline void bind_fp8_storage(DeviceWeight& weight,
                             const std::vector<uint8_t>& values,
                             const std::vector<float>& scales) {
    weight.fp8_storage.reset(values.size());
    weight.scales_storage.reset(scales.size());
    CELEG_CUDA(cudaMemcpy(weight.fp8_storage.data(), values.data(),
                          values.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                          scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    weight.linear.storage = Fp8LinearStorage{
        reinterpret_cast<const __nv_fp8_e4m3*>(weight.fp8_storage.data()),
        weight.scales_storage.data()};
    weight.linear.kernel = LinearKernelKind::Fp8W8A8;
}

inline void bind_nvfp4_storage(DeviceWeight& weight,
                               const std::vector<uint8_t>& packed,
                               const std::vector<uint8_t>& block_scales,
                               float global_scale,
                               float input_global_scale) {
    weight.nvfp4_packed_storage.reset(packed.size());
    weight.nvfp4_block_scale_storage.reset(block_scales.size());
    CELEG_CUDA(cudaMemcpy(weight.nvfp4_packed_storage.data(), packed.data(),
                          packed.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(weight.nvfp4_block_scale_storage.data(), block_scales.data(),
                          block_scales.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    weight.linear.storage = Nvfp4LinearStorage{
        weight.nvfp4_packed_storage.data(),
        reinterpret_cast<const __nv_fp8_e4m3*>(weight.nvfp4_block_scale_storage.data()),
        global_scale, input_global_scale};
    weight.linear.kernel = LinearKernelKind::Nvfp4W4A4;
}

inline void quantize_and_bind(DeviceWeight& weight,
                              const std::byte* dense_data,
                              size_t rows,
                              size_t cols,
                              WeightMode mode,
                              const __nv_bfloat16* bf16_fallback = nullptr) {
    if (mode == WeightMode::Int8) {
        const Int8RowwisePack pack = quantize_bf16_rows(dense_data, rows, cols);
        bind_int8_storage(weight, pack.values, pack.scales, bf16_fallback);
        return;
    }
    if (mode == WeightMode::Int4) {
        const Int4RowwisePack pack = quantize_bf16_rows_int4(dense_data, rows, cols);
        bind_int4_storage(weight, pack.values, pack.scales, bf16_fallback);
        return;
    }
    throw std::invalid_argument("quantize_and_bind requires Int8 or Int4 weight mode");
}

inline void finish_linear_binding(DeviceWeight& weight, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("linear weight dimensions must be positive");
    }
    weight.linear.rows = rows;
    weight.linear.cols = cols;
}

}

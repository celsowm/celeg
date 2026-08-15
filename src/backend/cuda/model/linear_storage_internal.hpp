#pragma once

#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/detail/model/device_weights.hpp"
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
    weight.linear.kind = LinearStorageKind::Int8;
    weight.linear.int8 = weight.int8_storage.data();
    weight.linear.scales = weight.scales_storage.data();
    weight.linear.bf16 = bf16_fallback;
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
    weight.linear.kind = LinearStorageKind::Int4;
    weight.linear.int4 = weight.int4_storage.data();
    weight.linear.scales = weight.scales_storage.data();
    weight.linear.bf16 = bf16_fallback;
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
    weight.linear.rows = rows;
    weight.linear.cols = cols;
    weight.linear.validate_storage();
}

}

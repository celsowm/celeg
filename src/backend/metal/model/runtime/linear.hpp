#pragma once

#import <Metal/Metal.h>

#include "celeg/model/weights/roles.hpp"
#include "linear_bindings.hpp"

#include <optional>

namespace celeg {

/**
 * @brief Kernel name and launch geometry for one matrix-vector binding.
 */
struct MetalMatvecKernel {
    const char* name = nullptr;
    uint32_t rows_per_threadgroup = 0;
    uint32_t threads = 0;
    uint32_t threadgroup_floats = 0;
};

/**
 * @brief Linear weight with quantized storage and decode memo.
 *
 * Holds the GPU buffer, logical shape and the memoized matvec pipeline
 * resolved from the quant registry. The memo is a pure function of
 * `(storage, rows, cols)` and is single-threaded.
 */
struct MetalLinear {
    id<MTLBuffer> buffer = nil;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t row_bytes = 0;
    metal_model_detail::MetalLinearStorage storage = metal_model_detail::MetalLinearStorage::Float32;
    std::optional<TensorRole> role;
    int layer = -1;
    mutable id<MTLComputePipelineState> cached_matvec_pipeline = nil;
    mutable MetalMatvecKernel cached_matvec_geometry{nullptr, 0, 0, 0};
    mutable uint8_t cached_matvec_path = 0;
};

}

#pragma once

#include "detail.hpp"

#include <optional>
#include <string_view>

namespace celeg {

/**
 * @brief Central registry for quantized kernel selection.
 *
 * All `LinearStorage`-driven switches that previously lived in
 * `pipelines.mm`, `linear_bindings.cpp`, `initialization.mm` and
 * `resources.mm` are consolidated here so a new storage format touches
 * exactly one translation unit. The registry is pure CPU and
 * `Metal`-free except for the device predicate needed for `M5` gates.
 */
std::optional<std::string_view> quant_linear_kernel(
    metal_model_detail::MetalLinearStorage storage,
    LinearOperationKind operation) noexcept;

MetalMatvecKernel quant_matvec_kernel(
    metal_model_detail::MetalLinearStorage storage,
    uint32_t rows,
    uint32_t cols,
    const MetalModelOptions& options,
    id<MTLDevice> device) noexcept;

MetalMatvecKernel quant_swiglu_matvec_kernel(
    metal_model_detail::MetalLinearStorage storage,
    uint32_t rows,
    uint32_t cols,
    const MetalModelOptions& options,
    id<MTLDevice> device) noexcept;

bool quant_tensor_matmul_available(
    metal_model_detail::MetalLinearStorage storage,
    const MetalPipelineCache& cache) noexcept;

bool quant_fast_tensor_matmul_available(
    metal_model_detail::MetalLinearStorage storage,
    const MetalPipelineCache& cache) noexcept;

/**
 * @brief Resolved tensor kernel and launch geometry for `encode_matmul`.
 */
struct QuantTensorKernel {
    std::string_view name{};
    NSUInteger shared_bytes{0};
    NSUInteger tile_tokens{0};
    bool exact_groups{false};
    bool custom{false};
};

QuantTensorKernel quant_select_tensor_kernel(
    metal_model_detail::MetalLinearStorage storage,
    uint32_t rows,
    uint32_t weight_rows,
    uint32_t weight_cols,
    TensorRole role,
    const MetalPipelineCache& cache,
    const MetalModelOptions& options,
    id<MTLDevice> device) noexcept;

/**
 * @brief Library selection for `tensor_pipeline` substring dispatch.
 */
id<MTLLibrary> quant_tensor_library_for(
    std::string_view kernel_name,
    const MetalPipelineCache& cache) noexcept;

}

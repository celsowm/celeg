#pragma once

#import <Metal/Metal.h>

#include <string>
#include <string_view>
#include <unordered_map>

namespace celeg {

/**
 * @brief Cache for Metal compute pipelines.
 *
 * Holds the generic inference library, the tensor libraries for each
 * quantized format and the fast variants, plus the feature flags that
 * describe which libraries successfully compiled. All pipeline lookups
 * are memoized in `pipelines`.
 */
struct MetalPipelineCache {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLLibrary> tensor_library = nil;
    id<MTLLibrary> tensor_fast_dense_library = nil;
    id<MTLLibrary> tensor_fast_q4_0_library = nil;
    id<MTLLibrary> tensor_fast_q4k_library = nil;
    id<MTLLibrary> tensor_fast_q5k_library = nil;
    id<MTLLibrary> tensor_fast_q6k_library = nil;
    id<MTLLibrary> tensor_fast_q8_0_library = nil;
    bool tensor_matmul_f16 = false;
    bool tensor_matmul_bf16 = false;
    bool tensor_matmul_q4_0 = false;
    bool tensor_matmul_q4k = false;
    bool tensor_matmul_q5k = false;
    bool tensor_matmul_q6k = false;
    bool tensor_matmul_q8_0 = false;
    bool tensor_fast_f16 = false;
    bool tensor_fast_bf16 = false;
    bool tensor_fast_q4_0 = false;
    bool tensor_fast_q4k = false;
    bool tensor_fast_q5k = false;
    bool tensor_fast_q6k = false;
    bool tensor_fast_q8_0 = false;
    std::string tensor_compile_error;
    std::string tensor_fast_dense_compile_error;
    std::string tensor_fast_q4_0_compile_error;
    std::string tensor_fast_q4k_compile_error;
    std::string tensor_fast_q5k_compile_error;
    std::string tensor_fast_q6k_compile_error;
    std::string tensor_fast_q8_0_compile_error;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    id<MTLComputePipelineState> pipeline(std::string_view name);
    id<MTLComputePipelineState> tensor_pipeline(std::string_view name);
};

}

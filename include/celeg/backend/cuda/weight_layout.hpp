#pragma once

#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/detail/model/types.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

namespace celeg {

// Strategy interface for weight-format dispatch. The runtime picks one
// concrete IWeightLayout per CudaModelOptions::weight_mode at model-construction
// time and binds it to the live embedding table + scales; the forward pass
// then calls embed_token / embed_batch / embed_token_device through the
// interface without branching on int4/int8/bf16 at every call site
// (Open/Closed Principle). New weight formats (e.g. INT4 groupwise, FP8)
// are added by registering a new IWeightLayout subclass rather than by
// editing model.cu's hot path.
class IWeightLayout {
public:
    virtual ~IWeightLayout() = default;

    // Single-token embedding lookup (decode path). Writes `hidden` BF16
    // elements to `out`.
    virtual void embed_token(int32_t token,
                             __nv_bfloat16* out,
                             int hidden,
                             cudaStream_t stream) = 0;

    // Multi-token embedding lookup (prefill path).
    virtual void embed_batch(const int32_t* tokens,
                             int rows,
                             __nv_bfloat16* out,
                             int hidden,
                             cudaStream_t stream) = 0;

    // Single-token embedding lookup where the token id lives in device
    // memory (CUDA-Graph capture path). The pointer must remain valid for
    // the duration of the kernel launch.
    virtual void embed_token_device(const int32_t* token,
                                    __nv_bfloat16* out,
                                    int hidden,
                                    cudaStream_t stream) = 0;
};

// Factory: returns the IWeightLayout appropriate for `weight_mode`, bound
// to the live embedding table + scales pointers. The table pointer is
// chosen based on the weight mode: for Bf16 it must point at BF16 storage,
// for Int8 at int8 storage, for Int4 at packed uint8 storage.
//
// Concretely:
//   WeightMode::Bf16 -> Bf16WeightLayout (table is __nv_bfloat16*)
//   WeightMode::Int8 -> Int8WeightLayout (table is int8_t*)
//   WeightMode::Int4 -> Int4WeightLayout (table is uint8_t*)
// New weight modes are added by extending this factory (OCP).
std::unique_ptr<IWeightLayout> make_weight_layout(
    WeightMode weight_mode,
    const void* table,
    const float* scales);

std::unique_ptr<IWeightLayout> make_gguf_weight_layout(
    const GgufLinearSegment& segment);

} // namespace celeg

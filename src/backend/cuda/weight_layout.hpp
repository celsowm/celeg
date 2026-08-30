#pragma once

#include "runtime_types.hpp"
#include "detail/linear_weights.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

namespace celeg {

class IWeightLayout {
public:
    virtual ~IWeightLayout() = default;

    virtual void embed_token(int32_t token,
                             __nv_bfloat16* out,
                             int hidden,
                             cudaStream_t stream) = 0;

    virtual void embed_batch(const int32_t* tokens,
                             int rows,
                             __nv_bfloat16* out,
                             int hidden,
                             cudaStream_t stream) = 0;

    virtual void embed_token_device(const int32_t* token,
                                    __nv_bfloat16* out,
                                    int hidden,
                                    cudaStream_t stream) = 0;
};

std::unique_ptr<IWeightLayout> make_weight_layout(
    WeightMode weight_mode,
    const void* table,
    const float* scales);

std::unique_ptr<IWeightLayout> make_gguf_weight_layout(
    const GgufLinearSegment& segment);

}

#pragma once

#include "celeg/model/graph.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace celeg {

enum class CudaQkPositionMode {
    HostScalar,
    DeviceScalar,
    MultiAxisDevice,
};

struct CudaAttentionQkPreparation {
    const AttentionSpec* layout = nullptr;
    __nv_bfloat16* query = nullptr;
    __nv_bfloat16* key = nullptr;
    const __nv_bfloat16* query_norm = nullptr;
    const __nv_bfloat16* key_norm = nullptr;
    float norm_epsilon = 0.0f;
    CudaQkPositionMode position_mode = CudaQkPositionMode::HostScalar;
    int host_position = 0;
    const int* device_position = nullptr;
    int mrope_section0 = 0;
    int mrope_section1 = 0;
    int mrope_section2 = 0;
    bool mrope_interleaved = false;
    cudaStream_t stream = nullptr;
};

struct CudaLatentQkPreparation {
    const AttentionSpec* layout = nullptr;
    __nv_bfloat16* query_rope = nullptr;
    __nv_bfloat16* key_rope = nullptr;
    float fallback_norm_epsilon = 0.0f;
    CudaQkPositionMode position_mode = CudaQkPositionMode::HostScalar;
    int host_position = 0;
    const int* device_position = nullptr;
    cudaStream_t stream = nullptr;
};

void prepare_cuda_attention_qk(const CudaAttentionQkPreparation& preparation);
void prepare_cuda_latent_attention_qk(const CudaLatentQkPreparation& preparation);

}

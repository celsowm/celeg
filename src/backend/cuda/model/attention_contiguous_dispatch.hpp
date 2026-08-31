#pragma once

#include "backend/cuda/attention_capability.hpp"
#include "celeg/model/graph.hpp"

#include <cuda_bf16.h>

namespace celeg {

struct CudaCompiledModel;
struct AttentionLayer;

void dispatch_cuda_standard_attention_contiguous(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    const AttentionCapability& plan,
    __nv_bfloat16* query);

}

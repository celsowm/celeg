#pragma once

#include <cuda_bf16.h>

namespace celeg {

struct AttentionLayer;
struct CudaCompiledModel;

__nv_bfloat16* prepare_cuda_token_attention_gate(
    CudaCompiledModel& model, AttentionLayer& attention, __nv_bfloat16* query);

void apply_cuda_token_attention_gate(
    CudaCompiledModel& model, AttentionLayer& attention);

}

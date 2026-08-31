#pragma once

namespace celeg {

struct AttentionLayer;
struct CudaCompiledModel;

void project_cuda_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention);

}

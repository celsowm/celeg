#pragma once

namespace celeg {

struct AttentionLayer;
struct CudaCompiledModel;

void require_cuda_projected_latent_bindings(const AttentionLayer& attention);

void project_cuda_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention);

}

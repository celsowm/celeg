#pragma once

namespace celeg {

struct AttentionLayer;
struct CompiledLayerProgram;
struct CudaCompiledModel;

void require_cuda_projected_latent_bindings(const AttentionLayer& attention);

void project_cuda_prefill_standard_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention, int rows);
void project_cuda_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention);
void project_cuda_prefill_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention, int rows);

void project_cuda_prefill_standard_attention_output(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows);
void project_cuda_prefill_latent_attention_output(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows);

}

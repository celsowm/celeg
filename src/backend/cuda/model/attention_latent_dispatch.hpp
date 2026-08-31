#pragma once

#include <cstdint>

namespace celeg {

struct AttentionLayer;
struct CudaCompiledModel;
struct PhysicalPagedKvCache;

void dispatch_cuda_latent_attention_contiguous(
    CudaCompiledModel& model, AttentionLayer& attention, AttentionLayer& owner);

void dispatch_cuda_latent_attention_paged(
    CudaCompiledModel& model, AttentionLayer& attention,
    PhysicalPagedKvCache& paged_kv, int slot,
    const uint32_t* device_page_table, int page_table_stride);

void dispatch_cuda_latent_attention_prefill(
    CudaCompiledModel& model, AttentionLayer& attention,
    AttentionLayer& owner, int rows);

}

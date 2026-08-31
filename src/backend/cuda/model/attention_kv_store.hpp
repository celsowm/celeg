#pragma once

#include <cstdint>

namespace celeg {

struct AttentionLayer;
struct CudaCompiledModel;
class PhysicalPagedKvCache;

void store_cuda_latent_kv_contiguous(
    CudaCompiledModel& model, AttentionLayer& attention, AttentionLayer& owner);

void store_cuda_latent_kv_paged(
    CudaCompiledModel& model, AttentionLayer& attention,
    PhysicalPagedKvCache& paged_kv, int slot,
    const uint32_t* device_page_table, int page_table_stride);

}

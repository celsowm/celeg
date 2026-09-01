#pragma once

#include "backend/cuda/attention_capability.hpp"
#include "celeg/model/graph.hpp"
#include "kernels/kernels.cuh"

namespace celeg {

enum class CudaDecodePositionMode {
    HostScalar,
    DeviceCounter,
};

struct CudaDecodeAttentionPolicy {
    AttentionCapability plan{};
    const BlockSparsePattern* block_sparse = nullptr;
};

struct CudaContiguousDecodeDispatch {
    AttentionCapability plan{};
    CudaDecodePositionMode position_mode = CudaDecodePositionMode::HostScalar;
    const BlockSparsePattern* block_sparse = nullptr;
    const __nv_bfloat16* query = nullptr;
    Bf16KvView bf16_kv{};
    Int8KvView int8_kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    AttentionDecodeSegmentation segmentation{};
    const float* alibi_slopes = nullptr;
    RelativePositionBiasDeviceView relative_bias{};
    cudaStream_t stream = nullptr;
};

struct CudaPagedDecodeDispatch {
    AttentionCapability plan{};
    const BlockSparsePattern* block_sparse = nullptr;
    const __nv_bfloat16* query = nullptr;
    Bf16KvPoolView bf16_kv{};
    Int8KvPoolView int8_kv{};
    PagedKvIndex index{};
    PagedKvScaleIndex scale_index{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    GqaGeometry geometry{};
    AttentionSegmentation segmentation{};
    const float* alibi_slopes = nullptr;
    RelativePositionBiasDeviceView relative_bias{};
    cudaStream_t stream = nullptr;
};

CudaDecodeAttentionPolicy plan_cuda_decode_attention(
    const AttentionSpec& layout,
    KvCacheMode kv_format,
    AttentionKvLayout kv_layout,
    AttentionPositionSource position_source,
    bool fast_attention,
    bool segmented_attention,
    int owner_head_dim);

GqaGeometry make_cuda_gqa_geometry(
    const AttentionSpec& layout,
    const AttentionSpec& owner_layout);

GqaBlockSparsePattern lower_cuda_block_sparse_pattern(
    const BlockSparsePattern& pattern);

void dispatch_cuda_contiguous_decode_attention(
    const CudaContiguousDecodeDispatch& dispatch);

void dispatch_cuda_paged_decode_attention(
    const CudaPagedDecodeDispatch& dispatch);

}

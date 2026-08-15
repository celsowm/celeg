#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>


namespace celeg {


struct GqaGeometry {
    int q_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int sliding_window = 0;
};

struct LatentGeometry {
    int query_heads = 0;
    int latent_rank = 0;
    int rotary_width = 0;
    float score_scale = 0.0f;
    int sliding_window = 0;
};

struct AttentionExtent {
    int rows = 1;
    int seq_len = 0;
    const int32_t* position = nullptr;
};

struct AttentionRowStrides {
    int q_width = 0;
    int kv_width = 0;
    int out_width = 0;
};

struct AttentionSegmentation {
    int chunk_tokens = 0;
    int chunks = 0;
    float* partial_max = nullptr;
    float* partial_denom = nullptr;
    float* partial_accum = nullptr;
};


struct Bf16KvView {
    const __nv_bfloat16* keys = nullptr;
    const __nv_bfloat16* values = nullptr;
};

struct Int8KvView {
    const int8_t* keys = nullptr;
    const int8_t* values = nullptr;
    const float* key_scales = nullptr;
    const float* value_scales = nullptr;
};

struct Bf16KvBatchView {
    const __nv_bfloat16* const* keys = nullptr;
    const __nv_bfloat16* const* values = nullptr;
};

struct Int8KvBatchView {
    const int8_t* const* keys = nullptr;
    const int8_t* const* values = nullptr;
    const float* const* key_scales = nullptr;
    const float* const* value_scales = nullptr;
};

struct Bf16KvPoolView {
    const __nv_bfloat16* keys = nullptr;
    const __nv_bfloat16* values = nullptr;
};

struct Int8KvPoolView {
    const int8_t* keys = nullptr;
    const int8_t* values = nullptr;
    const float* key_scales = nullptr;
    const float* value_scales = nullptr;
};

struct PagedKvIndex {
    const uint32_t* page_tables = nullptr;
    int page_table_stride = 0;
    int attention_slot = 0;
    int page_tokens = 0;
    size_t page_vector_elements = 0;
    size_t layer_vector_offset = 0;
};

struct PagedKvScaleIndex {
    size_t page_scale_elements = 0;
    size_t layer_scale_offset = 0;
};


struct GqaContiguousArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    const float* alibi_slopes = nullptr;
    cudaStream_t stream = nullptr;
};

struct GqaContiguousInt8Args {
    const __nv_bfloat16* query = nullptr;
    Int8KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    const float* alibi_slopes = nullptr;
    cudaStream_t stream = nullptr;
};

struct GqaSegmentedArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    AttentionSegmentation segmentation{};
    cudaStream_t stream = nullptr;
};

struct GqaSegmentedInt8Args {
    const __nv_bfloat16* query = nullptr;
    Int8KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    AttentionSegmentation segmentation{};
    cudaStream_t stream = nullptr;
};

struct GqaPrefillFlashArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionRowStrides strides{};
    int rows = 0;
    cudaStream_t stream = nullptr;
};

struct GqaPrefillGemmArgs {
    cublasHandle_t cublas = nullptr;
    const __nv_bfloat16* query = nullptr;
    Bf16KvView kv{};
    __nv_bfloat16* out = nullptr;
    float* scores_scratch = nullptr;
    __nv_bfloat16* probs_scratch = nullptr;
    GqaGeometry geometry{};
    AttentionRowStrides strides{};
    int rows = 0;
    cudaStream_t stream = nullptr;
};

struct GqaBatchPtrArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvBatchView kv{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    GqaGeometry geometry{};
    const float* alibi_slopes = nullptr;
    bool fast = false;
    cudaStream_t stream = nullptr;
};

struct GqaBatchPtrInt8Args {
    const __nv_bfloat16* query = nullptr;
    Int8KvBatchView kv{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    GqaGeometry geometry{};
    const float* alibi_slopes = nullptr;
    bool fast = false;
    cudaStream_t stream = nullptr;
};

struct GqaPagedArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvPoolView kv{};
    PagedKvIndex index{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    GqaGeometry geometry{};
    const float* alibi_slopes = nullptr;
    bool fast = false;
    cudaStream_t stream = nullptr;
};

struct GqaPagedInt8Args {
    const __nv_bfloat16* query = nullptr;
    Int8KvPoolView kv{};
    PagedKvIndex index{};
    PagedKvScaleIndex scale_index{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    GqaGeometry geometry{};
    const float* alibi_slopes = nullptr;
    bool fast = false;
    cudaStream_t stream = nullptr;
};

struct GqaPagedSegmentedArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvPoolView kv{};
    PagedKvIndex index{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    GqaGeometry geometry{};
    AttentionSegmentation segmentation{};
    cudaStream_t stream = nullptr;
};

struct GqaPagedSegmentedInt8Args {
    const __nv_bfloat16* query = nullptr;
    Int8KvPoolView kv{};
    PagedKvIndex index{};
    PagedKvScaleIndex scale_index{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    GqaGeometry geometry{};
    AttentionSegmentation segmentation{};
    cudaStream_t stream = nullptr;
};


struct LatentQueryView {
    const __nv_bfloat16* content = nullptr;
    const __nv_bfloat16* rope = nullptr;
};

struct LatentKvView {
    const __nv_bfloat16* keys = nullptr;
    const __nv_bfloat16* values = nullptr;
    const __nv_bfloat16* key_rope = nullptr;
};

struct LatentKvBatchView {
    const __nv_bfloat16* const* keys = nullptr;
    const __nv_bfloat16* const* values = nullptr;
    const __nv_bfloat16* const* key_rope = nullptr;
};

struct LatentKvPoolView {
    const __nv_bfloat16* keys = nullptr;
    const __nv_bfloat16* values = nullptr;
};

struct LatentContiguousArgs {
    LatentQueryView query{};
    LatentKvView kv{};
    __nv_bfloat16* out = nullptr;
    AttentionExtent extent{};
    const float* alibi_slopes = nullptr;
    LatentGeometry geometry{};
    cudaStream_t stream = nullptr;
};

struct LatentPagedArgs {
    LatentQueryView query{};
    LatentKvPoolView kv{};
    PagedKvIndex index{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    const float* alibi_slopes = nullptr;
    LatentGeometry geometry{};
    cudaStream_t stream = nullptr;
};

struct LatentBatchPtrArgs {
    LatentQueryView query{};
    LatentKvBatchView kv{};
    __nv_bfloat16* out = nullptr;
    const int32_t* positions = nullptr;
    int rows = 0;
    const float* alibi_slopes = nullptr;
    LatentGeometry geometry{};
    cudaStream_t stream = nullptr;
};


struct FactorizedLatentQueryArgs {
    const __nv_bfloat16* query_projection = nullptr;
    const __nv_bfloat16* expansion = nullptr;
    __nv_bfloat16* query_content = nullptr;
    int rows = 0;
    int query_heads = 0;
    int query_nope = 0;
    int query_rope_dim = 0;
    int latent_rank = 0;
    cudaStream_t stream = nullptr;
};

struct FactorizedLatentValueArgs {
    const __nv_bfloat16* latent_output = nullptr;
    const __nv_bfloat16* expansion = nullptr;
    __nv_bfloat16* value_output = nullptr;
    int rows = 0;
    int query_heads = 0;
    int query_nope = 0;
    int value_dim = 0;
    int latent_rank = 0;
    cudaStream_t stream = nullptr;
};

struct FactorizedLatentRopeArgs {
    const __nv_bfloat16* query_projection = nullptr;
    __nv_bfloat16* query_rope = nullptr;
    int rows = 0;
    int query_heads = 0;
    int query_nope = 0;
    int query_rope_dim = 0;
    cudaStream_t stream = nullptr;
};

}

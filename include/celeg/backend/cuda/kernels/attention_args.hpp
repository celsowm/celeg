#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Typed argument aggregates for the CUDA attention launchers.
//
// CUDA attention has a real specialization matrix (see
// `celeg/backend/cuda/attention_capability.hpp`):
//
//     {decode,prefill} x {bf16,int8} x {alibi,none} x {contiguous,paged,batch-ptrs}
//     x {whole,segmented} x {strict,online,flash,gemm}
//
// Those kernels are separately benchmarked and stay separate.  What does not
// have to stay separate is the *shape* of the host-side call: previously every
// launcher took a flat positional list of ten to twenty parameters, many of
// them adjacent same-typed pointers (`const __nv_bfloat16*` key/value),
// adjacent ints (`q_heads, kv_heads, head_dim, sliding_window`,
// `chunk_tokens, chunks`) and adjacent `float*` scratch buffers
// (`partial_max, partial_denom, partial_accum`).  Transposing two of those at a
// call site compiles cleanly and produces wrong numbers.
//
// The aggregates below group parameters by concern, so call sites name what
// they pass.  They are plain aggregates: no constructors, no virtual
// dispatch, nothing that survives into device code.  Every launcher still maps
// one-to-one onto the same kernel it always did.
//
// The axis names deliberately mirror the capability policy header:
// `KvCacheMode` (bf16 vs int8 views), `AttentionKvLayout` (contiguous vs pool
// vs batch pointers), `AttentionPositionSource` (`AttentionExtent::seq_len`
// vs `AttentionExtent::position`), `AttentionPositionBias`
// (`alibi_slopes`) and `AttentionAlgorithm` (which launcher is called).

namespace celeg {

// ---------------------------------------------------------------------------
// Geometry and launch shape
// ---------------------------------------------------------------------------

// Grouped-query attention head geometry.  `sliding_window <= 0` means "attend
// to the whole causal prefix".
struct GqaGeometry {
    int q_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int sliding_window = 0;
};

// Multi-head latent attention geometry (compressed KV, split rope/nope query).
struct LatentGeometry {
    int query_heads = 0;
    int latent_rank = 0;
    int rotary_width = 0;
    float score_scale = 0.0f;
    int sliding_window = 0;
};

// How far the kernel attends, and where that extent comes from.
//
// Exactly one of `seq_len` / `position` is meaningful for a given launcher,
// and which one is fixed by the launcher name, not by this struct:
//
//   * host-scalar launchers (`launch_gqa_decode_strict`, every `_prefill_`
//     entry point) read `seq_len` (decode: KV length; prefill: query rows) and
//     ignore `position`;
//   * device-counter launchers (the `_device` and `_batch` families) read
//     `position` / `positions` and ignore `seq_len`.
//
// `rows` is the query-row count for the bulk entry points; single-token decode
// launchers hard-code one row.
struct AttentionExtent {
    int rows = 1;
    int seq_len = 0;
    const int32_t* position = nullptr;
};

// Interleaved-head row strides, in elements, for the prefill kernels that read
// query/KV straight out of the packed activation buffers instead of a cache.
struct AttentionRowStrides {
    int q_width = 0;
    int kv_width = 0;
    int out_width = 0;
};

// Chunked online-softmax partial state.  `chunks` must equal
// ceil(kv_extent / chunk_tokens); `partial_max`/`partial_denom` hold one float
// per (row, head, chunk) and `partial_accum` `head_dim` floats per that tuple.
struct AttentionSegmentation {
    int chunk_tokens = 0;
    int chunks = 0;
    float* partial_max = nullptr;
    float* partial_denom = nullptr;
    float* partial_accum = nullptr;
};

// ---------------------------------------------------------------------------
// KV cache views, one per (KvCacheMode x AttentionKvLayout) pair
// ---------------------------------------------------------------------------

// Contiguous per-layer BF16 cache.
struct Bf16KvView {
    const __nv_bfloat16* keys = nullptr;
    const __nv_bfloat16* values = nullptr;
};

// Contiguous per-layer INT8 cache with per-(token, kv-head) dequant scales.
struct Int8KvView {
    const int8_t* keys = nullptr;
    const int8_t* values = nullptr;
    const float* key_scales = nullptr;
    const float* value_scales = nullptr;
};

// Device array of per-request contiguous BF16 caches (packed execution).
struct Bf16KvBatchView {
    const __nv_bfloat16* const* keys = nullptr;
    const __nv_bfloat16* const* values = nullptr;
};

// Device array of per-request contiguous INT8 caches (packed execution).
struct Int8KvBatchView {
    const int8_t* const* keys = nullptr;
    const int8_t* const* values = nullptr;
    const float* const* key_scales = nullptr;
    const float* const* value_scales = nullptr;
};

// Physical BF16 page pool shared by every request.
struct Bf16KvPoolView {
    const __nv_bfloat16* keys = nullptr;
    const __nv_bfloat16* values = nullptr;
};

// Physical INT8 page pool plus its scale pool.
struct Int8KvPoolView {
    const int8_t* keys = nullptr;
    const int8_t* values = nullptr;
    const float* key_scales = nullptr;
    const float* value_scales = nullptr;
};

// How a page pool is addressed: the per-request page table plus the pool
// geometry needed to turn (page, token, layer) into an element offset.
struct PagedKvIndex {
    const uint32_t* page_tables = nullptr;
    int page_table_stride = 0;
    int attention_slot = 0;
    int page_tokens = 0;
    size_t page_vector_elements = 0;
    size_t layer_vector_offset = 0;
};

// The same addressing for the INT8 scale pool, whose element size differs from
// the vector pool's.
struct PagedKvScaleIndex {
    size_t page_scale_elements = 0;
    size_t layer_scale_offset = 0;
};

// ---------------------------------------------------------------------------
// Grouped-query attention launcher arguments
// ---------------------------------------------------------------------------

// Contiguous BF16 cache, single stream: strict/online decode (host or device
// position), strict/online prefill, and the fused-ALiBi variants (which are the
// only ones that read `alibi_slopes`; the others require it to be null).
struct GqaContiguousArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    const float* alibi_slopes = nullptr;
    cudaStream_t stream = nullptr;
};

// INT8 twin of `GqaContiguousArgs`.
struct GqaContiguousInt8Args {
    const __nv_bfloat16* query = nullptr;
    Int8KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    const float* alibi_slopes = nullptr;
    cudaStream_t stream = nullptr;
};

// Segmented (chunked online-softmax) attention over a contiguous BF16 cache.
struct GqaSegmentedArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    AttentionSegmentation segmentation{};
    cudaStream_t stream = nullptr;
};

// INT8 twin of `GqaSegmentedArgs`.
struct GqaSegmentedInt8Args {
    const __nv_bfloat16* query = nullptr;
    Int8KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionExtent extent{};
    AttentionSegmentation segmentation{};
    cudaStream_t stream = nullptr;
};

// Tiled ("flash") prefill: query/KV read from the packed activation buffers
// through `strides`, no score matrix materialized.
struct GqaPrefillFlashArgs {
    const __nv_bfloat16* query = nullptr;
    Bf16KvView kv{};
    __nv_bfloat16* out = nullptr;
    GqaGeometry geometry{};
    AttentionRowStrides strides{};
    int rows = 0;
    cudaStream_t stream = nullptr;
};

// Batched-GEMM prefill.  `scores_scratch` needs q_heads*rows*rows floats and
// `probs_scratch` the same element count in BF16.
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

// One decode row per packed request, each against its own contiguous BF16
// cache.  `fast` selects the online kernel over the strict one and is ignored
// by the ALiBi launcher, which has a single kernel.
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

// INT8 twin of `GqaBatchPtrArgs`.
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

// One decode row per packed request against the shared BF16 page pool.
// `fast` is ignored by the ALiBi launcher (single kernel).
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

// INT8 twin of `GqaPagedArgs`.
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

// Segmented paged decode: same addressing as `GqaPagedArgs` plus the chunk
// partial state.  There is no `fast` axis; the segmented kernel is the choice.
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

// INT8 twin of `GqaPagedSegmentedArgs`.
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

// ---------------------------------------------------------------------------
// Latent (compressed-KV) attention launcher arguments
// ---------------------------------------------------------------------------

// The query halves: `content` is the latent-rank projection, `rope` the
// rotary-carrying tail.
struct LatentQueryView {
    const __nv_bfloat16* content = nullptr;
    const __nv_bfloat16* rope = nullptr;
};

// Contiguous per-layer latent cache.
struct LatentKvView {
    const __nv_bfloat16* keys = nullptr;
    const __nv_bfloat16* values = nullptr;
    const __nv_bfloat16* key_rope = nullptr;
};

// Device array of per-request contiguous latent caches (packed execution).
struct LatentKvBatchView {
    const __nv_bfloat16* const* keys = nullptr;
    const __nv_bfloat16* const* values = nullptr;
    const __nv_bfloat16* const* key_rope = nullptr;
};

// Shared latent page pool (the rope tail lives inside the same pool vectors).
struct LatentKvPoolView {
    const __nv_bfloat16* keys = nullptr;
    const __nv_bfloat16* values = nullptr;
};

// Latent attention over a contiguous cache: decode (device position) and
// prefill (host row count).
struct LatentContiguousArgs {
    LatentQueryView query{};
    LatentKvView kv{};
    __nv_bfloat16* out = nullptr;
    AttentionExtent extent{};
    const float* alibi_slopes = nullptr;
    LatentGeometry geometry{};
    cudaStream_t stream = nullptr;
};

// Latent decode over the shared page pool.
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

// Latent decode over per-request contiguous caches.
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

// ---------------------------------------------------------------------------
// Factorized latent projections
// ---------------------------------------------------------------------------
//
// These three helpers previously ended in five or six bare `int` parameters,
// which is exactly the transposition hazard this header exists to remove.

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

} // namespace celeg

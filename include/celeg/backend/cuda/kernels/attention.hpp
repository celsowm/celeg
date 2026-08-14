#pragma once

#include "celeg/backend/cuda/kernels/attention_args.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// CUDA attention launchers.
//
// Each entry point below still maps one-to-one onto a separately benchmarked
// kernel specialization; the specialization matrix is documented in
// `celeg/backend/cuda/attention_capability.hpp`, which is also what decides
// which of these a call site is allowed to reach.  All arguments travel in the
// typed aggregates declared in `attention_args.hpp` so no call site has to get
// a run of same-typed positional parameters in the right order.

namespace celeg {

// ---- Contiguous BF16 cache, strict/online softmax ------------------------
// `_device` reads `args.extent.position`; the others read `args.extent.seq_len`
// (decode: KV length, prefill: query rows).  `args.alibi_slopes` must be null.
void launch_gqa_decode_strict(const GqaContiguousArgs& args);
void launch_gqa_decode_strict_device(const GqaContiguousArgs& args);
void launch_gqa_decode_online(const GqaContiguousArgs& args);
void launch_gqa_decode_online_device(const GqaContiguousArgs& args);
void launch_gqa_prefill_strict(const GqaContiguousArgs& args);
void launch_gqa_prefill_online(const GqaContiguousArgs& args);

// ---- Contiguous BF16 cache, segmented online softmax ---------------------
// Each (row, head, chunk) block handles at most
// `args.segmentation.chunk_tokens` KV positions (causally clamped) instead of
// the single serial per-row loop in `launch_gqa_prefill_online`, then a reduce
// kernel combines the chunk-local online-softmax partials per row/head.  Trades
// one extra kernel plus O(rows * q_heads * chunks) scratch for far more
// block-level parallelism and a much shorter per-block critical path.
void launch_gqa_decode_segmented_device(const GqaSegmentedArgs& args);
void launch_gqa_prefill_segmented(const GqaSegmentedArgs& args);

// ---- Contiguous BF16 cache, prefill fast paths ---------------------------
// Tiled online-softmax prefill; no score matrix is materialized.  Rejects
// head dimensions above `kAttentionFlashMaxHeadDim`.
void launch_gqa_prefill_flash(const GqaPrefillFlashArgs& args);
// Batched-GEMM prefill (see the private attention .cuh implementation for the
// rationale).  `args.strides` are the interleaved-head row strides of the
// packed activation buffers this path reads and writes.
void launch_gqa_prefill_gemm(const GqaPrefillGemmArgs& args);

// ---- Contiguous INT8 cache ----------------------------------------------
void launch_gqa_decode_strict_int8(const GqaContiguousInt8Args& args);
void launch_gqa_decode_online_int8(const GqaContiguousInt8Args& args);
void launch_gqa_decode_strict_int8_device(const GqaContiguousInt8Args& args);
void launch_gqa_decode_online_int8_device(const GqaContiguousInt8Args& args);
void launch_gqa_prefill_strict_int8(const GqaContiguousInt8Args& args);
void launch_gqa_prefill_online_int8(const GqaContiguousInt8Args& args);
void launch_gqa_decode_segmented_int8_device(const GqaSegmentedInt8Args& args);

// ---- Per-request contiguous caches gathered through pointer arrays -------
void launch_gqa_decode_batch_ptrs(const GqaBatchPtrArgs& args);
void launch_gqa_decode_int8_batch_ptrs(const GqaBatchPtrInt8Args& args);

// ---- Shared page pool ----------------------------------------------------
void launch_gqa_decode_paged_batch(const GqaPagedArgs& args);
void launch_gqa_decode_int8_paged_batch(const GqaPagedInt8Args& args);
void launch_gqa_decode_paged_segmented_batch(const GqaPagedSegmentedArgs& args);
void launch_gqa_decode_int8_paged_segmented_batch(const GqaPagedSegmentedInt8Args& args);

// ---- Fused ALiBi family (bias applied inside the score loop) -------------
// These read `args.alibi_slopes`, which must be non-null.  The batch-pointer
// and paged ALiBi launchers have a single kernel each and ignore `args.fast`.
void launch_gqa_decode_alibi_device(const GqaContiguousArgs& args);
void launch_gqa_decode_alibi_int8_device(const GqaContiguousInt8Args& args);
void launch_gqa_prefill_alibi(const GqaContiguousArgs& args);
void launch_gqa_prefill_alibi_int8(const GqaContiguousInt8Args& args);
void launch_gqa_decode_alibi_batch_ptrs(const GqaBatchPtrArgs& args);
void launch_gqa_decode_alibi_int8_batch_ptrs(const GqaBatchPtrInt8Args& args);
void launch_gqa_decode_alibi_paged_batch(const GqaPagedArgs& args);
void launch_gqa_decode_alibi_int8_paged_batch(const GqaPagedInt8Args& args);

// ---- Latent (compressed-KV) attention ------------------------------------
void launch_latent_attention_device(const LatentContiguousArgs& args);
void launch_latent_attention_prefill(const LatentContiguousArgs& args);
void launch_latent_attention_paged_batch(const LatentPagedArgs& args);
void launch_latent_attention_batch_ptrs(const LatentBatchPtrArgs& args);

// ---- Factorized latent projections ---------------------------------------
void launch_factorized_latent_query(const FactorizedLatentQueryArgs& args);
void launch_factorized_latent_value(const FactorizedLatentValueArgs& args);
void launch_factorized_latent_rope(const FactorizedLatentRopeArgs& args);

} // namespace celeg

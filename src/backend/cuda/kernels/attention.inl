// Attention kernels, one leaf per execution strategy. All of these are
// textually included into the single attention.cu translation unit, so the
// helpers they share may keep external linkage; splitting a leaf out into its
// own .cu would require giving those internal linkage first (see
// attention_common.inl).
//
// Include order matters in one place: attention_paged.inl resolves page
// tables through kv_cache.inl's offset helpers, so kv_cache.inl comes first.

#include "kv_cache.inl"

#include "attention_common.inl"
#include "attention_dense.inl"
#include "attention_segmented.inl"
#include "attention_batch_ptrs.inl"
#include "attention_paged.inl"
#include "attention_gemm.inl"

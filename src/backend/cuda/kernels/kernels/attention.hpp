#pragma once

#include "kernels/attention_args.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>


namespace celeg {

struct GqaBlockSparsePattern {
    int block_size = 0;
    int local_blocks = 0;
    int global_blocks = 0;
};

struct GqaDynamicSparsePattern {
    int block_size = 0;
    int max_selected_blocks = 0;
};

void launch_gqa_decode_strict(const GqaContiguousArgs& args);
void launch_gqa_decode_strict_device(const GqaContiguousArgs& args);
void launch_gqa_decode_online(const GqaContiguousArgs& args);
void launch_gqa_decode_online_device(const GqaContiguousArgs& args);
void launch_gqa_decode_block_sparse_device(
    const GqaContiguousArgs& args, GqaBlockSparsePattern pattern);
void launch_gqa_prefill_strict(const GqaContiguousArgs& args);
void launch_gqa_prefill_online(const GqaContiguousArgs& args);
void launch_gqa_prefill_block_sparse(
    const GqaContiguousArgs& args, GqaBlockSparsePattern pattern);
void launch_gqa_prefill_dynamic_sparse(
    const GqaContiguousArgs& args, GqaDynamicSparsePattern pattern);

void launch_gqa_decode_segmented_device(const GqaDecodeSegmentedArgs& args);
void launch_gqa_prefill_segmented(const GqaSegmentedArgs& args);

void launch_gqa_prefill_flash(const GqaPrefillFlashArgs& args);
void launch_gqa_prefill_gemm(const GqaPrefillGemmArgs& args);

void launch_gqa_decode_strict_int8(const GqaContiguousInt8Args& args);
void launch_gqa_decode_online_int8(const GqaContiguousInt8Args& args);
void launch_gqa_decode_strict_int8_device(const GqaContiguousInt8Args& args);
void launch_gqa_decode_online_int8_device(const GqaContiguousInt8Args& args);
void launch_gqa_decode_block_sparse_int8_device(
    const GqaContiguousInt8Args& args, GqaBlockSparsePattern pattern);
void launch_gqa_prefill_strict_int8(const GqaContiguousInt8Args& args);
void launch_gqa_prefill_online_int8(const GqaContiguousInt8Args& args);
void launch_gqa_prefill_block_sparse_int8(
    const GqaContiguousInt8Args& args, GqaBlockSparsePattern pattern);
void launch_gqa_decode_segmented_int8_device(const GqaDecodeSegmentedInt8Args& args);

void launch_gqa_decode_batch_ptrs(const GqaBatchPtrArgs& args);
void launch_gqa_decode_int8_batch_ptrs(const GqaBatchPtrInt8Args& args);

void launch_gqa_decode_paged_batch(const GqaPagedArgs& args);
void launch_gqa_decode_int8_paged_batch(const GqaPagedInt8Args& args);
void launch_gqa_decode_block_sparse_paged(
    const GqaPagedArgs& args, GqaBlockSparsePattern pattern);
void launch_gqa_decode_block_sparse_int8_paged(
    const GqaPagedInt8Args& args, GqaBlockSparsePattern pattern);
void launch_gqa_decode_paged_segmented_batch(const GqaPagedSegmentedArgs& args);
void launch_gqa_decode_int8_paged_segmented_batch(const GqaPagedSegmentedInt8Args& args);

void launch_gqa_decode_alibi_device(const GqaContiguousArgs& args);
void launch_gqa_decode_alibi_int8_device(const GqaContiguousInt8Args& args);
void launch_gqa_prefill_alibi(const GqaContiguousArgs& args);
void launch_gqa_prefill_alibi_int8(const GqaContiguousInt8Args& args);
void launch_gqa_decode_alibi_batch_ptrs(const GqaBatchPtrArgs& args);
void launch_gqa_decode_alibi_int8_batch_ptrs(const GqaBatchPtrInt8Args& args);
void launch_gqa_decode_alibi_paged_batch(const GqaPagedArgs& args);
void launch_gqa_decode_alibi_int8_paged_batch(const GqaPagedInt8Args& args);

void launch_gqa_decode_relative_device(const GqaContiguousArgs& args);
void launch_gqa_decode_relative_int8_device(const GqaContiguousInt8Args& args);
void launch_gqa_prefill_relative(const GqaContiguousArgs& args);
void launch_gqa_prefill_relative_int8(const GqaContiguousInt8Args& args);
void launch_gqa_decode_relative_batch_ptrs(const GqaBatchPtrArgs& args);
void launch_gqa_decode_relative_int8_batch_ptrs(const GqaBatchPtrInt8Args& args);
void launch_gqa_decode_relative_paged_batch(const GqaPagedArgs& args);
void launch_gqa_decode_relative_int8_paged_batch(const GqaPagedInt8Args& args);

void launch_latent_attention_device(const LatentContiguousArgs& args);
void launch_latent_attention_prefill(const LatentContiguousArgs& args);
void launch_latent_attention_paged_batch(const LatentPagedArgs& args);
void launch_latent_attention_batch_ptrs(const LatentBatchPtrArgs& args);

void launch_factorized_latent_query(const FactorizedLatentQueryArgs& args);
void launch_factorized_latent_value(const FactorizedLatentValueArgs& args);
void launch_factorized_latent_rope(const FactorizedLatentRopeArgs& args);

}

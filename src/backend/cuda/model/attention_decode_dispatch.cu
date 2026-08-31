#include "attention_decode_dispatch.hpp"

#include <stdexcept>

namespace celeg {

CudaDecodeAttentionPolicy plan_cuda_decode_attention(
    const AttentionSpec& layout,
    KvCacheMode kv_format,
    AttentionKvLayout kv_layout,
    AttentionPositionSource position_source,
    bool fast_attention,
    bool segmented_attention,
    bool has_alibi,
    int owner_head_dim) {
    const auto* block_sparse = std::get_if<BlockSparsePattern>(&layout.pattern);
    AttentionRequest request;
    request.kv_format = kv_format;
    request.operation = AttentionOperation::Decode;
    request.layout = kv_layout;
    request.position_source = position_source;
    request.bias = has_alibi
        ? AttentionPositionBias::Alibi : AttentionPositionBias::None;
    request.fast_attention = fast_attention && block_sparse == nullptr;
    request.segmented_attention = segmented_attention && block_sparse == nullptr;
    request.head_dim = owner_head_dim;
    return {
        .plan = require_attention_capability(request),
        .block_sparse = block_sparse};
}

GqaGeometry make_cuda_gqa_geometry(
    const AttentionSpec& layout,
    const AttentionSpec& owner_layout) {
    return {
        .q_heads = layout.query_heads,
        .kv_heads = owner_layout.key_value_heads,
        .head_dim = owner_layout.head_dim,
        .sliding_window = layout.sliding_window_size()};
}

GqaBlockSparsePattern lower_cuda_block_sparse_pattern(
    const BlockSparsePattern& pattern) {
    return {
        .block_size = pattern.block_size,
        .local_blocks = pattern.local_blocks,
        .global_blocks = pattern.global_blocks};
}

void dispatch_cuda_contiguous_decode_attention(
    const CudaContiguousDecodeDispatch& dispatch) {
    const bool int8_kv = dispatch.plan.kv_format == KvCacheMode::Int8;
    const bool device_position =
        dispatch.position_mode == CudaDecodePositionMode::DeviceCounter;

    switch (dispatch.plan.algorithm) {
    case AttentionAlgorithm::Alibi:
        if (!device_position) throw UnsupportedAttentionCapability(dispatch.plan);
        if (int8_kv) {
            launch_gqa_decode_alibi_int8_device({
                .query = dispatch.query,
                .kv = dispatch.int8_kv,
                .out = dispatch.out,
                .geometry = dispatch.geometry,
                .extent = dispatch.extent,
                .alibi_slopes = dispatch.alibi_slopes,
                .stream = dispatch.stream});
        } else {
            launch_gqa_decode_alibi_device({
                .query = dispatch.query,
                .kv = dispatch.bf16_kv,
                .out = dispatch.out,
                .geometry = dispatch.geometry,
                .extent = dispatch.extent,
                .alibi_slopes = dispatch.alibi_slopes,
                .stream = dispatch.stream});
        }
        return;

    case AttentionAlgorithm::Segmented:
        if (!device_position) throw UnsupportedAttentionCapability(dispatch.plan);
        if (int8_kv) {
            launch_gqa_decode_segmented_int8_device({
                .query = dispatch.query,
                .kv = dispatch.int8_kv,
                .out = dispatch.out,
                .geometry = dispatch.geometry,
                .extent = dispatch.extent,
                .segmentation = dispatch.segmentation,
                .stream = dispatch.stream});
        } else {
            launch_gqa_decode_segmented_device({
                .query = dispatch.query,
                .kv = dispatch.bf16_kv,
                .out = dispatch.out,
                .geometry = dispatch.geometry,
                .extent = dispatch.extent,
                .segmentation = dispatch.segmentation,
                .stream = dispatch.stream});
        }
        return;

    case AttentionAlgorithm::Online:
        if (int8_kv) {
            const GqaContiguousInt8Args args{
                .query = dispatch.query,
                .kv = dispatch.int8_kv,
                .out = dispatch.out,
                .geometry = dispatch.geometry,
                .extent = dispatch.extent,
                .stream = dispatch.stream};
            if (device_position) launch_gqa_decode_online_int8_device(args);
            else launch_gqa_decode_online_int8(args);
        } else {
            const GqaContiguousArgs args{
                .query = dispatch.query,
                .kv = dispatch.bf16_kv,
                .out = dispatch.out,
                .geometry = dispatch.geometry,
                .extent = dispatch.extent,
                .stream = dispatch.stream};
            if (device_position) launch_gqa_decode_online_device(args);
            else launch_gqa_decode_online(args);
        }
        return;

    case AttentionAlgorithm::Strict:
        if (dispatch.block_sparse) {
            const GqaBlockSparsePattern pattern =
                lower_cuda_block_sparse_pattern(*dispatch.block_sparse);
            if (int8_kv) {
                launch_gqa_decode_block_sparse_int8_device({
                    .query = dispatch.query,
                    .kv = dispatch.int8_kv,
                    .out = dispatch.out,
                    .geometry = dispatch.geometry,
                    .extent = dispatch.extent,
                    .stream = dispatch.stream}, pattern);
            } else {
                launch_gqa_decode_block_sparse_device({
                    .query = dispatch.query,
                    .kv = dispatch.bf16_kv,
                    .out = dispatch.out,
                    .geometry = dispatch.geometry,
                    .extent = dispatch.extent,
                    .stream = dispatch.stream}, pattern);
            }
            return;
        }
        if (int8_kv) {
            const GqaContiguousInt8Args args{
                .query = dispatch.query,
                .kv = dispatch.int8_kv,
                .out = dispatch.out,
                .geometry = dispatch.geometry,
                .extent = dispatch.extent,
                .stream = dispatch.stream};
            if (device_position) launch_gqa_decode_strict_int8_device(args);
            else launch_gqa_decode_strict_int8(args);
        } else {
            const GqaContiguousArgs args{
                .query = dispatch.query,
                .kv = dispatch.bf16_kv,
                .out = dispatch.out,
                .geometry = dispatch.geometry,
                .extent = dispatch.extent,
                .stream = dispatch.stream};
            if (device_position) launch_gqa_decode_strict_device(args);
            else launch_gqa_decode_strict(args);
        }
        return;

    case AttentionAlgorithm::Flash:
    case AttentionAlgorithm::Gemm:
        throw UnsupportedAttentionCapability(dispatch.plan);
    }
}

void dispatch_cuda_paged_decode_attention(
    const CudaPagedDecodeDispatch& dispatch) {
    const bool int8_kv = dispatch.plan.kv_format == KvCacheMode::Int8;

    switch (dispatch.plan.algorithm) {
    case AttentionAlgorithm::Alibi:
        if (int8_kv) {
            launch_gqa_decode_alibi_int8_paged_batch({
                .query = dispatch.query,
                .kv = dispatch.int8_kv,
                .index = dispatch.index,
                .scale_index = dispatch.scale_index,
                .out = dispatch.out,
                .positions = dispatch.positions,
                .rows = dispatch.rows,
                .geometry = dispatch.geometry,
                .alibi_slopes = dispatch.alibi_slopes,
                .stream = dispatch.stream});
        } else {
            launch_gqa_decode_alibi_paged_batch({
                .query = dispatch.query,
                .kv = dispatch.bf16_kv,
                .index = dispatch.index,
                .out = dispatch.out,
                .positions = dispatch.positions,
                .rows = dispatch.rows,
                .geometry = dispatch.geometry,
                .alibi_slopes = dispatch.alibi_slopes,
                .stream = dispatch.stream});
        }
        return;

    case AttentionAlgorithm::Segmented:
        if (int8_kv) {
            launch_gqa_decode_int8_paged_segmented_batch({
                .query = dispatch.query,
                .kv = dispatch.int8_kv,
                .index = dispatch.index,
                .scale_index = dispatch.scale_index,
                .out = dispatch.out,
                .positions = dispatch.positions,
                .rows = dispatch.rows,
                .geometry = dispatch.geometry,
                .segmentation = dispatch.segmentation,
                .stream = dispatch.stream});
        } else {
            launch_gqa_decode_paged_segmented_batch({
                .query = dispatch.query,
                .kv = dispatch.bf16_kv,
                .index = dispatch.index,
                .out = dispatch.out,
                .positions = dispatch.positions,
                .rows = dispatch.rows,
                .geometry = dispatch.geometry,
                .segmentation = dispatch.segmentation,
                .stream = dispatch.stream});
        }
        return;

    case AttentionAlgorithm::Online:
    case AttentionAlgorithm::Strict:
        if (dispatch.block_sparse) {
            if (dispatch.plan.algorithm != AttentionAlgorithm::Strict) {
                throw UnsupportedAttentionCapability(dispatch.plan);
            }
            const GqaBlockSparsePattern pattern =
                lower_cuda_block_sparse_pattern(*dispatch.block_sparse);
            if (int8_kv) {
                launch_gqa_decode_block_sparse_int8_paged({
                    .query = dispatch.query,
                    .kv = dispatch.int8_kv,
                    .index = dispatch.index,
                    .scale_index = dispatch.scale_index,
                    .out = dispatch.out,
                    .positions = dispatch.positions,
                    .rows = dispatch.rows,
                    .geometry = dispatch.geometry,
                    .stream = dispatch.stream}, pattern);
            } else {
                launch_gqa_decode_block_sparse_paged({
                    .query = dispatch.query,
                    .kv = dispatch.bf16_kv,
                    .index = dispatch.index,
                    .out = dispatch.out,
                    .positions = dispatch.positions,
                    .rows = dispatch.rows,
                    .geometry = dispatch.geometry,
                    .stream = dispatch.stream}, pattern);
            }
            return;
        }
        if (int8_kv) {
            launch_gqa_decode_int8_paged_batch({
                .query = dispatch.query,
                .kv = dispatch.int8_kv,
                .index = dispatch.index,
                .scale_index = dispatch.scale_index,
                .out = dispatch.out,
                .positions = dispatch.positions,
                .rows = dispatch.rows,
                .geometry = dispatch.geometry,
                .fast = dispatch.plan.algorithm == AttentionAlgorithm::Online,
                .stream = dispatch.stream});
        } else {
            launch_gqa_decode_paged_batch({
                .query = dispatch.query,
                .kv = dispatch.bf16_kv,
                .index = dispatch.index,
                .out = dispatch.out,
                .positions = dispatch.positions,
                .rows = dispatch.rows,
                .geometry = dispatch.geometry,
                .fast = dispatch.plan.algorithm == AttentionAlgorithm::Online,
                .stream = dispatch.stream});
        }
        return;

    case AttentionAlgorithm::Flash:
    case AttentionAlgorithm::Gemm:
        throw UnsupportedAttentionCapability(dispatch.plan);
    }
}

}

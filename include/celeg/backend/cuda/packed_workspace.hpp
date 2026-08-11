#pragma once

#include "celeg/backend/cuda/packed.hpp"
#include "celeg/backend/cuda/packed_handles.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/model/runtime_types.hpp"

#include <cstddef>
#include <cstdint>

namespace celeg {

class PhysicalPagedKvCache;

// Owns every reusable host/device buffer used by packed decode and ragged
// prefill. The executor owns one workspace for its whole lifetime; request
// execution only fills these buffers and never grows them on the hot path.
struct PackedWorkspace {
    PackedWorkspace(size_t maximum_batch,
                    size_t maximum_prefill_tokens,
                    PhysicalPagedKvCache* paged_kv,
                    const RuntimeTopology& shape);

    void ensure_segmented_workspace(int rows, int chunks);

    size_t maximum_batch = 0;
    size_t maximum_prefill_token_capacity = 0;
    PhysicalPagedKvCache* paged_kv = nullptr;
    RuntimeTopology shape_;
    PackedWorkspaceRequirements requirements_;
    PackedExecutionHandles handles;
    CudaStream& stream;
    CublasHandle& cublas;

    DeviceBuffer<int32_t> positions;
    DeviceBuffer<int32_t> sampled;
    PinnedBuffer<int32_t> sampled_host;
    DeviceBuffer<float> temperatures;
    DeviceBuffer<float> repetition_penalties;
    DeviceBuffer<int32_t> top_k;
    DeviceBuffer<float> top_p;

    DeviceBuffer<__nv_bfloat16> hidden;
    DeviceBuffer<__nv_bfloat16> residual;
    DeviceBuffer<__nv_bfloat16> normed;
    DeviceBuffer<__nv_bfloat16> op_output;
    DeviceBuffer<__nv_bfloat16> qkv_output;
    DeviceBuffer<__nv_bfloat16> mamba_projected;
    DeviceBuffer<__nv_bfloat16> mamba_inner;
    DeviceBuffer<__nv_bfloat16> q;
    DeviceBuffer<__nv_bfloat16> k;
    DeviceBuffer<__nv_bfloat16> v;
    DeviceBuffer<__nv_bfloat16> conv_projected;
    DeviceBuffer<__nv_bfloat16> gated_delta_qkv;
    DeviceBuffer<__nv_bfloat16> gated_delta_z;
    DeviceBuffer<__nv_bfloat16> gated_delta_b;
    DeviceBuffer<__nv_bfloat16> gated_delta_a;
    DeviceBuffer<__nv_bfloat16> gated_delta_output;
    DeviceBuffer<__nv_bfloat16> gate_up;
    DeviceBuffer<__nv_bfloat16> activated;
    DeviceBuffer<__nv_bfloat16> mlp_output;
    DeviceBuffer<__nv_bfloat16> logits;

    DeviceBuffer<float> moe_hidden_float;
    DeviceBuffer<int32_t> moe_sel;
    DeviceBuffer<float> moe_routing_w;
    DeviceBuffer<float> moe_router_scratch;
    DeviceBuffer<__nv_bfloat16> moe_output;
    DeviceBuffer<float> moe_output_accum;
    DeviceBuffer<__nv_bfloat16> moe_gu_scratch;
    DeviceBuffer<__nv_bfloat16> moe_act_scratch;
    DeviceBuffer<float> sampling_scores;
    DeviceBuffer<float> selected_values;
    DeviceBuffer<int32_t> selected_indices;

    PinnedBuffer<int32_t> h_positions;
    PinnedBuffer<float> h_temperatures;
    PinnedBuffer<float> h_repetition_penalties;
    PinnedBuffer<int32_t> h_top_k;
    PinnedBuffer<float> h_top_p;

    PinnedBuffer<__nv_bfloat16*> h_logits;
    PinnedBuffer<__nv_bfloat16*> h_selected_logits;
    PinnedBuffer<uint8_t*> h_seen;
    PinnedBuffer<uint64_t*> h_rng;
    PinnedBuffer<int32_t*> h_sampled_dest;
    PinnedBuffer<int32_t*> h_position_dest;
    DeviceBuffer<__nv_bfloat16*> d_logits;
    DeviceBuffer<__nv_bfloat16*> d_selected_logits;
    DeviceBuffer<uint8_t*> d_seen;
    DeviceBuffer<uint64_t*> d_rng;
    DeviceBuffer<int32_t*> d_sampled_dest;
    DeviceBuffer<int32_t*> d_position_dest;

    PinnedBuffer<__nv_bfloat16*> h_key_bf16;
    PinnedBuffer<__nv_bfloat16*> h_value_bf16;
    PinnedBuffer<int8_t*> h_key_int8;
    PinnedBuffer<int8_t*> h_value_int8;
    PinnedBuffer<float*> h_key_scales;
    PinnedBuffer<float*> h_value_scales;
    PinnedBuffer<__nv_bfloat16*> h_conv_states;
    DeviceBuffer<__nv_bfloat16*> d_key_bf16;
    DeviceBuffer<__nv_bfloat16*> d_value_bf16;
    DeviceBuffer<int8_t*> d_key_int8;
    DeviceBuffer<int8_t*> d_value_int8;
    DeviceBuffer<float*> d_key_scales;
    DeviceBuffer<float*> d_value_scales;
    DeviceBuffer<__nv_bfloat16*> d_conv_states;
    PinnedBuffer<uint32_t> h_page_tables;
    DeviceBuffer<uint32_t> d_page_tables;
    PinnedBuffer<int32_t> h_span_offsets;
    PinnedBuffer<int32_t> h_span_counts;
    PinnedBuffer<int32_t> h_final_rows;
    PinnedBuffer<int32_t> h_selected_final_rows;
    PinnedBuffer<uint8_t*> h_flat_seen;
    DeviceBuffer<int32_t> d_span_offsets;
    DeviceBuffer<int32_t> d_span_counts;
    DeviceBuffer<int32_t> d_final_rows;
    DeviceBuffer<int32_t> d_selected_final_rows;
    DeviceBuffer<uint8_t*> d_flat_seen;
    DeviceBuffer<float> segmented_partial_max;
    DeviceBuffer<float> segmented_partial_denom;
    DeviceBuffer<float> segmented_partial_accum;
    size_t segmented_scalar_capacity = 0;
    size_t segmented_accum_capacity = 0;
};

} // namespace celeg

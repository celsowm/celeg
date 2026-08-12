#include "celeg/backend/cuda/packed_workspace.hpp"

#include "celeg/backend/cuda/paged_kv.hpp"

#include <algorithm>
#include <stdexcept>

namespace celeg {

PackedWorkspace::PackedWorkspace(size_t maximum_batch_value,
                                 size_t maximum_prefill_tokens_value,
                                 PhysicalPagedKvCache* paged_kv_value,
                                 const ExecutionTopology& shape,
                                 int vocab_size)
    : maximum_batch(maximum_batch_value),
      maximum_prefill_token_capacity(maximum_prefill_tokens_value),
      paged_kv(paged_kv_value),
      shape_(shape),
      vocab_size_(vocab_size),
      requirements_(PackedWorkspaceRequirements::derive(
          maximum_batch_value, maximum_prefill_tokens_value,
          paged_kv_value ? static_cast<size_t>(paged_kv_value->max_pages_per_request()) : 0,
          shape_)),
      handles(),
      stream(handles.stream),
      cublas(handles.cublas),
      positions(maximum_prefill_token_capacity),
      sampled(maximum_prefill_token_capacity),
      sampled_host(maximum_batch),
      temperatures(maximum_batch),
      repetition_penalties(maximum_batch),
      top_k(maximum_batch),
      top_p(maximum_batch),
      hidden(maximum_prefill_token_capacity * shape_.hidden),
      residual(maximum_prefill_token_capacity * shape_.hidden),
      normed(maximum_prefill_token_capacity * shape_.hidden),
      op_output(maximum_prefill_token_capacity * shape_.hidden),
      qkv_output(maximum_prefill_token_capacity * requirements_.maximum_projection_width),
      mamba_projected(maximum_prefill_token_capacity *
                      requirements_.maximum_mamba_projection_width),
      mamba_inner(maximum_prefill_token_capacity *
                  requirements_.maximum_mamba_intermediate),
      q(maximum_prefill_token_capacity * requirements_.maximum_projection_width),
      k(maximum_prefill_token_capacity * requirements_.maximum_projection_width),
      v(maximum_prefill_token_capacity * requirements_.maximum_projection_width),
      conv_projected(maximum_prefill_token_capacity * 3 * shape_.hidden),
      gated_delta_qkv(maximum_prefill_token_capacity *
                      static_cast<size_t>(std::max(1, shape_.max_gated_delta_net_qkv_width()))),
      gated_delta_z(maximum_prefill_token_capacity *
                    static_cast<size_t>(std::max(1, shape_.max_gated_delta_net_output_width()))),
      gated_delta_b(maximum_prefill_token_capacity *
                    static_cast<size_t>(std::max(1, shape_.max_gated_delta_net_gate_width()))),
      gated_delta_a(maximum_prefill_token_capacity *
                    static_cast<size_t>(std::max(1, shape_.max_gated_delta_net_gate_width()))),
      gated_delta_output(maximum_prefill_token_capacity *
                         static_cast<size_t>(std::max(1, shape_.max_gated_delta_net_output_width()))),
      gate_up(maximum_prefill_token_capacity * 2 * requirements_.maximum_ffn_intermediate),
      activated(maximum_prefill_token_capacity * requirements_.maximum_ffn_intermediate),
      mlp_output(maximum_prefill_token_capacity * shape_.hidden),
      moe_hidden_float(maximum_prefill_token_capacity * shape_.hidden),
      moe_sel(maximum_prefill_token_capacity * shape_.experts_per_token),
      moe_routing_w(maximum_prefill_token_capacity * shape_.experts_per_token),
      moe_router_scratch(maximum_prefill_token_capacity * shape_.num_experts),
      moe_output(maximum_prefill_token_capacity * shape_.hidden),
      moe_output_accum(maximum_prefill_token_capacity * shape_.hidden),
      moe_gu_scratch(static_cast<size_t>(maximum_prefill_token_capacity) *
                     shape_.experts_per_token * 2 * requirements_.moe_intermediate),
      moe_act_scratch(static_cast<size_t>(maximum_prefill_token_capacity) *
                      shape_.experts_per_token * requirements_.moe_intermediate),
      logits(maximum_prefill_token_capacity * vocab_size_),
      sampling_scores(maximum_batch * vocab_size_),
      selected_values(maximum_batch * static_cast<size_t>(kMaxTopK)),
      selected_indices(maximum_batch * static_cast<size_t>(kMaxTopK)),
      h_positions(maximum_prefill_token_capacity),
      h_temperatures(maximum_batch),
      h_repetition_penalties(maximum_batch),
      h_top_k(maximum_batch),
      h_top_p(maximum_batch),
      h_logits(maximum_batch),
      h_selected_logits(maximum_batch),
      h_seen(maximum_batch),
      h_rng(maximum_batch),
      h_sampled_dest(maximum_batch),
      h_position_dest(maximum_batch),
      d_logits(maximum_batch),
      d_selected_logits(maximum_batch),
      d_seen(maximum_batch),
      d_rng(maximum_batch),
      d_sampled_dest(maximum_batch),
      d_position_dest(maximum_batch),
      h_key_bf16(requirements_.layer_slots),
      h_value_bf16(requirements_.layer_slots),
      h_key_int8(requirements_.layer_slots),
      h_value_int8(requirements_.layer_slots),
      h_key_scales(requirements_.layer_slots),
      h_value_scales(requirements_.layer_slots),
      h_conv_states(requirements_.layer_slots),
      d_key_bf16(requirements_.layer_slots),
      d_value_bf16(requirements_.layer_slots),
      d_key_int8(requirements_.layer_slots),
      d_value_int8(requirements_.layer_slots),
      d_key_scales(requirements_.layer_slots),
      d_value_scales(requirements_.layer_slots),
      d_conv_states(requirements_.layer_slots),
      h_page_tables(requirements_.page_table_entries),
      d_page_tables(requirements_.page_table_entries),
      h_span_offsets(maximum_batch),
      h_span_counts(maximum_batch),
      h_final_rows(maximum_batch),
      h_selected_final_rows(maximum_batch),
      h_flat_seen(maximum_prefill_token_capacity),
      d_span_offsets(maximum_batch),
      d_span_counts(maximum_batch),
      d_final_rows(maximum_batch),
      d_selected_final_rows(maximum_batch),
      d_flat_seen(maximum_prefill_token_capacity) {
    std::fill_n(h_logits.data(), maximum_batch, nullptr);
    std::fill_n(h_selected_logits.data(), maximum_batch, nullptr);
    std::fill_n(h_seen.data(), maximum_batch, nullptr);
    std::fill_n(h_rng.data(), maximum_batch, nullptr);
    std::fill_n(h_sampled_dest.data(), maximum_batch, nullptr);
    std::fill_n(h_position_dest.data(), maximum_batch, nullptr);
    std::fill_n(h_key_bf16.data(), requirements_.layer_slots, nullptr);
    std::fill_n(h_value_bf16.data(), requirements_.layer_slots, nullptr);
    std::fill_n(h_key_int8.data(), requirements_.layer_slots, nullptr);
    std::fill_n(h_value_int8.data(), requirements_.layer_slots, nullptr);
    std::fill_n(h_key_scales.data(), requirements_.layer_slots, nullptr);
    std::fill_n(h_value_scales.data(), requirements_.layer_slots, nullptr);
    std::fill_n(h_conv_states.data(), requirements_.layer_slots, nullptr);
}

void PackedWorkspace::ensure_segmented_workspace(int rows, int chunks) {
    const size_t scalar_count = static_cast<size_t>(rows) *
        shape_.maximum_attention_query_heads() * static_cast<size_t>(chunks);
    const size_t accum_count = scalar_count * shape_.maximum_attention_head_dim();
    if (scalar_count > segmented_scalar_capacity) {
        segmented_partial_max.reset(scalar_count);
        segmented_partial_denom.reset(scalar_count);
        segmented_scalar_capacity = scalar_count;
    }
    if (accum_count > segmented_accum_capacity) {
        segmented_partial_accum.reset(accum_count);
        segmented_accum_capacity = accum_count;
    }
}

} // namespace celeg

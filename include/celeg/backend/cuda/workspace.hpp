#pragma once

#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"
#include "celeg/backend/cuda/utils.cuh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace celeg {

// Concrete storage owned by one CUDA model execution.  Keeping transient
// buffers and expert-residency state here prevents the model facade from
// becoming the owner of backend mechanics while retaining direct data access
// in the kernel hot path.
struct CudaWorkspace {
    DeviceBuffer<__nv_bfloat16> hidden_;
    DeviceBuffer<__nv_bfloat16> residual_;
    DeviceBuffer<__nv_bfloat16> normed_;
    DeviceBuffer<__nv_bfloat16> op_output_;
    DeviceBuffer<__nv_bfloat16> qkv_output_;
    DeviceBuffer<__nv_bfloat16> conv_projected_;
    DeviceBuffer<__nv_bfloat16> gate_up_;
    DeviceBuffer<__nv_bfloat16> activated_;
    DeviceBuffer<__nv_bfloat16> mlp_output_;
    DeviceBuffer<__nv_bfloat16> logits_;
    DeviceBuffer<__nv_bfloat16> rope_cos_;
    DeviceBuffer<__nv_bfloat16> rope_sin_;
    DeviceBuffer<float> attention_partial_max_;
    DeviceBuffer<float> attention_partial_denom_;
    DeviceBuffer<float> attention_partial_accum_;
    int attention_chunks_ = 0;

    DeviceBuffer<uint32_t> paged_page_table_;
    DeviceBuffer<int32_t> paged_prefill_tokens_;

    DeviceBuffer<int32_t> prefill_tokens_;
    DeviceBuffer<__nv_bfloat16> prefill_hidden_;
    DeviceBuffer<__nv_bfloat16> prefill_residual_;
    DeviceBuffer<__nv_bfloat16> prefill_normed_;
    DeviceBuffer<__nv_bfloat16> prefill_op_output_;
    DeviceBuffer<__nv_bfloat16> prefill_qkv_;
    DeviceBuffer<__nv_bfloat16> prefill_q_;
    DeviceBuffer<__nv_bfloat16> prefill_k_;
    DeviceBuffer<__nv_bfloat16> prefill_v_;
    DeviceBuffer<__nv_bfloat16> prefill_conv_projected_;
    DeviceBuffer<__nv_bfloat16> prefill_gate_up_;
    DeviceBuffer<__nv_bfloat16> prefill_activated_;
    DeviceBuffer<__nv_bfloat16> prefill_mlp_output_;
    DeviceBuffer<float> prefill_attn_partial_max_;
    DeviceBuffer<float> prefill_attn_partial_denom_;
    DeviceBuffer<float> prefill_attn_partial_accum_;
    DeviceBuffer<float> prefill_attn_scores_;
    DeviceBuffer<__nv_bfloat16> prefill_attn_probs_;

    DeviceBuffer<float> moe_hidden_float_;
    DeviceBuffer<int> moe_sel_;
    DeviceBuffer<float> moe_routing_w_;
    DeviceBuffer<float> moe_router_scratch_;
    DeviceBuffer<float> moe_output_accum_;
    DeviceBuffer<__nv_bfloat16> moe_output_;
    DeviceBuffer<__nv_bfloat16> moe_gu_scratch_;
    DeviceBuffer<__nv_bfloat16> moe_act_scratch_;

    DeviceBuffer<float> moe_pf_hidden_float_;
    DeviceBuffer<int> moe_pf_sel_;
    DeviceBuffer<int> moe_pf_sel_masked_;
    DeviceBuffer<std::uint8_t> expert_active_dev_;
    DeviceBuffer<float> moe_pf_routing_w_;
    DeviceBuffer<float> moe_pf_router_scratch_;
    DeviceBuffer<float> moe_pf_output_accum_;
    DeviceBuffer<__nv_bfloat16> moe_pf_output_;
    DeviceBuffer<__nv_bfloat16> moe_pf_gu_scratch_;
    DeviceBuffer<__nv_bfloat16> moe_pf_act_scratch_;
    std::vector<DeviceBuffer<float>> moe_router_float_;

    ExpertOffloadPlan expert_offload_plan_;
    HostExpertStore host_expert_store_;
    std::vector<ExpertLayerCache*> expert_caches_;
    std::vector<std::vector<ExpertLocation>> expert_catalog_;
    std::unique_ptr<CudaStream> expert_transfer_stream_;
    CudaEvent ffn_done_event_;
    CudaEvent promote_done_event_;
    CudaEvent prefetch_done_event_;
    CudaEvent router_done_event_;
    int* moe_sel_host_ = nullptr;
    size_t moe_sel_host_cap_ = 0;
    float* moe_route_scores_host_ = nullptr;
    size_t moe_route_scores_cap_ = 0;
    std::vector<int> prefetch_idx_;
    std::vector<int> prefetch_ranked_;
    std::vector<float> prefetch_scores_;
    std::vector<int> cold_expert_host_;
    std::vector<float> cold_scores_host_;
};

} // namespace celeg

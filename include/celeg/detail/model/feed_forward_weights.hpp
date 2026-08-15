#pragma once


#include "celeg/detail/model/expert_weights.hpp"
#include "celeg/detail/model/linear_weights.hpp"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/model/program.hpp"

#include <cstddef>
#include <variant>

namespace celeg {

struct DenseFfnWeights {
    const LinearWeight* w13 = nullptr;
    const LinearWeight* w2 = nullptr;
};

struct MoeFfnWeights {
    const LinearWeight* router = nullptr;
    const float* expert_bias = nullptr;
    const ExpertLinearWeight* gate_up = nullptr;
    const ExpertLinearWeight* down = nullptr;
    const float* router_float = nullptr;

    const __nv_bfloat16* const* gate_up_ptrs = nullptr;
    const __nv_bfloat16* const* down_ptrs = nullptr;

    const LinearWeight* shared_w13 = nullptr;
    const LinearWeight* shared_w2 = nullptr;
    const LinearWeight* shared_gate = nullptr;

    bool offloaded() const { return gate_up_ptrs != nullptr; }
};

using FeedForwardWeights = std::variant<DenseFfnWeights, MoeFfnWeights>;

inline DenseFfnWeights* as_dense_ffn(FeedForwardWeights& ff) {
    return std::get_if<DenseFfnWeights>(&ff);
}
inline const DenseFfnWeights* as_dense_ffn(const FeedForwardWeights& ff) {
    return std::get_if<DenseFfnWeights>(&ff);
}
inline MoeFfnWeights* as_moe_ffn(FeedForwardWeights& ff) {
    return std::get_if<MoeFfnWeights>(&ff);
}
inline const MoeFfnWeights* as_moe_ffn(const FeedForwardWeights& ff) {
    return std::get_if<MoeFfnWeights>(&ff);
}
inline bool is_moe_ffn(const FeedForwardWeights& ff) {
    return std::holds_alternative<MoeFfnWeights>(ff);
}

inline celeg::MoeRouterConfig moe_router_config(const MoeLayerProgram& semantics) {
    celeg::MoeRouterConfig cfg;
    cfg.num_experts = semantics.router.expert_count;
    cfg.experts_per_token = semantics.router.experts_per_token;
    cfg.normalize_topk = semantics.router.normalization == MoeNormalizationKind::SumSelected;
    cfg.softmax = semantics.router.score == MoeRouterScoreKind::SoftmaxLogits;
    cfg.use_expert_bias = semantics.router.has_expert_bias;
    cfg.routed_scaling_factor = semantics.router.routed_scaling;
    if (const auto* grouped = std::get_if<MoeGroupedTopKSelectionSpec>(
            &semantics.router.selection)) {
        cfg.group_count = grouped->group_count;
        cfg.experts_per_group = grouped->experts_per_group;
        cfg.groups_per_token = grouped->groups_per_token;
        cfg.group_score_top_k = grouped->group_score_top_k;
    }
    return cfg;
}

inline celeg::MoeFfnDevice moe_ffn_device(const MoeFfnWeights& moe,
                                          const MoeLayerProgram& semantics) {
    celeg::MoeFfnDevice fdev;
    fdev.num_experts = semantics.router.expert_count;
    fdev.inter = semantics.routed.mlp.intermediate_size;
    fdev.hidden_dim = semantics.routed.mlp.hidden_size;
    if (moe.offloaded()) {
        fdev.gate_up_ptrs = moe.gate_up_ptrs;
        fdev.down_ptrs = moe.down_ptrs;
    } else {
        if (moe.gate_up->kind == LinearStorageKind::Q4_K ||
            moe.gate_up->kind == LinearStorageKind::Q6_K) {
            fdev.gate_up_gguf = moe.gate_up->gguf_blocks;
            fdev.down_gguf = moe.down->gguf_blocks;
            fdev.gate_up_gguf_type = moe.gate_up->gguf_type;
            fdev.down_gguf_type = moe.down->gguf_type;
            fdev.expert_gate_up_row_bytes = moe.gate_up->gguf_row_bytes;
            fdev.expert_down_row_bytes = moe.down->gguf_row_bytes;
            fdev.expert_gate_up_byte_stride = moe.gate_up->gguf_expert_stride;
            fdev.expert_down_byte_stride = moe.down->gguf_expert_stride;
        } else {
            fdev.gate_up = moe.gate_up->bf16;
            fdev.down = moe.down->bf16;
            fdev.expert_gate_up_stride =
                static_cast<size_t>(2) * semantics.routed.mlp.intermediate_size *
                    semantics.routed.mlp.hidden_size;
            fdev.expert_down_stride =
                static_cast<size_t>(semantics.routed.mlp.hidden_size) *
                    semantics.routed.mlp.intermediate_size;
        }
    }
    return fdev;
}

}

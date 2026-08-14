#pragma once

#include "celeg/model/inference.hpp"

#include <vector>

namespace celeg::inference_detail {

struct CanonicalInferenceContext {
    explicit CanonicalInferenceContext(const InferenceInput& source)
        : input(source) {}

    const InferenceInput& input;
    CanonicalModelFacts facts;
    const TensorInventoryEntry* embedding = nullptr;
    std::vector<int> intermediate_sizes;

    int layer_count = 0;
    int dense_start = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool has_moe = false;
    int moe_intermediate = 0;
    int shared_expert_intermediate = 0;
    bool normalize_topk = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
    bool moe_router_softmax = false;
    int routing_groups = 0;
    int total_routing_groups = 0;
    int groups_per_token = 0;
    int group_score_top_k = 0;
    int experts_per_group = 0;
};

CanonicalInferenceContext initialize_canonical_facts(const InferenceInput& input);
void infer_layer_semantics(CanonicalInferenceContext& context);
void bind_canonical_tensors(CanonicalInferenceContext& context);

} // namespace celeg::inference_detail

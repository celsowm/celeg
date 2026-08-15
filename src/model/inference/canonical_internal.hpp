#pragma once

#include "celeg/model/inference.hpp"

#include <optional>
#include <vector>

namespace celeg::inference_detail {

struct CanonicalMoeFacts {
    int num_experts = 0;
    int experts_per_token = 0;
    int intermediate_size = 0;
    MoeSelectionSpec selection = MoeTopKSelectionSpec{};
    std::optional<SharedExpertSpec> shared;
    bool normalize_topk = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
    bool router_softmax = false;
};

struct CanonicalInferenceContext {
    explicit CanonicalInferenceContext(const InferenceInput& source)
        : input(source) {}

    const InferenceInput& input;
    CanonicalModelFacts facts;
    const TensorInventoryEntry* embedding = nullptr;
    std::vector<int> intermediate_sizes;

    int layer_count = 0;
    int dense_start = 0;
    std::optional<CanonicalMoeFacts> moe;
};

CanonicalInferenceContext initialize_canonical_facts(const InferenceInput& input);
void infer_layer_semantics(CanonicalInferenceContext& context);
void bind_canonical_tensors(CanonicalInferenceContext& context);

}

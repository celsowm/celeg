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

    // Physical layer count from the checkpoint's own layer-count metadata
    // (e.g. num_hidden_layers). Equal to layer_count unless the checkpoint
    // declares a layer-repeat schedule (e.g. "num_loops"), in which case
    // layer_count = physical_layer_count * repeat_count and each physical
    // layer's weights are reused across multiple virtual layers.
    int physical_layer_count = 0;

    // Maps a virtual layer index (used for graph/KV-cache/binding identity)
    // to the physical layer index whose weights it should bind to (used for
    // tensor-name candidate generation only).
    int physical_layer(int layer) const {
        return physical_layer_count > 0 ? layer % physical_layer_count : layer;
    }
};

CanonicalInferenceContext initialize_canonical_facts(const InferenceInput& input);
void resolve_canonical_layers(CanonicalInferenceContext& context);

}

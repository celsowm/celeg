#include "celeg/model/inference.hpp"

#include "celeg/model/graph_builder.hpp"

#include <stdexcept>

namespace celeg {


ModelGraph GraphSynthesizer::synthesize(const CanonicalModelFacts& facts) const {
    facts.validate();
    ResolvedModel intermediate;
    intermediate.topology = facts.topology;
    build_dense_transformer_graph(intermediate);
    return intermediate.graph;
}

WeightPlan WeightPlanSynthesizer::synthesize(const CanonicalModelFacts& facts) const {
    facts.validate();
    WeightPlan result;
    result.requests.reserve(facts.bindings.values.size());
    for (const auto& binding : facts.bindings.values) {
        result.requests.push_back({binding.role, binding.layer, binding.expert,
                                   binding.shape, binding.source_name,
                                   binding.physical_layer});
    }
    return result;
}

ResolvedModel ResolutionAssembler::assemble(const CanonicalModelFacts& facts) const {
    facts.validate();
    ResolvedModel result;
    result.topology = facts.topology;
    result.graph = GraphSynthesizer{}.synthesize(facts);
    // Keep the semantic gate granularity aligned with the canonical physical
    // binding.  This is still checkpoint-neutral: the resolver may discover
    // the granularity from a tensor shape, while graph lowering must preserve
    // that fact through the topology/graph round trip.
    for (size_t layer = 0; layer < result.graph.layers.size(); ++layer) {
        auto* attention = std::get_if<AttentionSpec>(&result.graph.layers[layer].mixer);
        if (!attention || !attention->output_gate.enabled() ||
            attention->output_gate.packed_with_query) continue;
        const auto* binding = facts.bindings.find(TensorRole::AttentionGate,
                                                  static_cast<int>(layer));
        if (!binding || binding->shape.size() != 2) continue;
        if (binding->shape[0] == attention->query_heads) {
            attention->output_gate.granularity = AttentionGateGranularity::HeadWise;
        } else if (binding->shape[0] == attention->query_width()) {
            attention->output_gate.granularity = AttentionGateGranularity::ElementWise;
        } else {
            throw std::invalid_argument("attention gate binding width does not match graph geometry");
        }
    }
    result.weight_plan = WeightPlanSynthesizer{}.synthesize(facts);
    result.capabilities = {true, true, false, facts.tied_embeddings};
    result.provenance.architecture_id = facts.resolution_mode;
    result.provenance.source_format = facts.source_format;
    result.provenance.checkpoint_profile_id = facts.resolution_mode;
    derive_runtime_topology_from_graph(result.topology, result.graph);
    result.topology.validate();
    result.provenance.identity = facts.fingerprint();
    result.validate();
    return result;
}

} // namespace celeg

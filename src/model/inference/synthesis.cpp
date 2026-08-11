#include "celeg/model/inference.hpp"

#include "celeg/model/graph_builder.hpp"

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

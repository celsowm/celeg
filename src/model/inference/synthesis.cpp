#include "celeg/model/inference.hpp"

#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weight_plan.hpp"

#include <stdexcept>
#include <utility>

namespace celeg {
namespace {

class CanonicalBindingNamingPolicy final : public ITensorNamingPolicy {
public:
    explicit CanonicalBindingNamingPolicy(const TensorRoleBindings& bindings)
        : bindings_(bindings) {}

    std::vector<std::string> candidates(const TensorRequest& request) const override {
        const TensorRoleBinding* binding = bindings_.find(
            request.role, request.layer, request.expert);
        return binding == nullptr ? std::vector<std::string>{}
                                  : std::vector<std::string>{binding->source_name};
    }

private:
    const TensorRoleBindings& bindings_;
};

} // namespace


ModelGraph GraphSynthesizer::synthesize(const CanonicalModelFacts& facts) const {
    facts.validate();
    ResolvedModel intermediate;
    build_dense_transformer_graph(intermediate, facts.topology);
    intermediate.graph.validate();
    return intermediate.graph;
}

ResolvedModel ResolutionAssembler::assemble(const CanonicalModelFacts& facts) const {
    facts.validate();
    ResolvedModel result;
    result.topology = facts.topology;
    result.graph = GraphSynthesizer{}.synthesize(facts);
    result.capabilities = {true, true, false, facts.tied_embeddings};
    result.provenance.architecture_id = facts.resolution_mode;
    result.provenance.source_format = facts.source_format;
    result.provenance.checkpoint_profile_id = facts.resolution_mode;
    result.topology = compose_runtime_topology(std::move(result.topology), result.graph);
    result.topology.validate();
    CanonicalBindingNamingPolicy naming(facts.bindings);
    build_weight_plan_from_graph(result, naming);
    result.provenance.identity = facts.fingerprint();
    result.validate();
    return result;
}

} // namespace celeg

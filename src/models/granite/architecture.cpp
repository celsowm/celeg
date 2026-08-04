#include "celeg/detail/model/builtin_architectures.hpp"
#include "celeg/detail/models/granite_topology.hpp"
#include "celeg/model/weight_plan.hpp"
#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weights/roles.hpp"
#include "naming_policy.hpp"

#include <stdexcept>

namespace celeg::detail {
namespace {

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const auto policy = std::make_shared<const GraniteTensorNamingPolicy>();
    return policy;
}

class GraniteArchitecture final : public IArchitecture {
public:
    std::string_view id() const override { return "granite"; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const std::string type = metadata.architecture_type();
        return {type == "granite", type == "granite" ? 100 : 0,
                type == "granite" ? "Granite checkpoint" : "not Granite"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("Granite architecture cannot resolve checkpoint");
        }
        const auto& source = checkpoint.metadata;
        RuntimeTopology t = resolve_granite_topology(source);

        ResolvedModel result;
        result.topology = t;
        result.architecture_id = "granite";
        result.source_format = source.is_gguf() ? "gguf" : "safetensors";
        result.checkpoint_profile_id = "granite";
        result.chat_profile_id = "granite-instruct";
        result.profile = {"granite", "", {}, result.chat_profile_id};
        result.identity = "granite-" + t.fingerprint();
        result.capabilities = {true, true, false, true};
        build_dense_transformer_graph(result);
        build_dense_weight_plan(result, *naming_policy());
        return result;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_granite_architecture() {
    return std::make_unique<GraniteArchitecture>();
}

void register_granite_architecture(ArchitectureCatalog& catalog) {
    catalog.add(make_granite_architecture());
}

} // namespace celeg::detail

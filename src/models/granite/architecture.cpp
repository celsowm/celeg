#include "celeg/detail/model/builtin_architectures.hpp"
#include "celeg/detail/models/granite_topology.hpp"
#include "celeg/model/weight_plan.hpp"
#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weights/roles.hpp"

#include <stdexcept>

namespace celeg::detail {
namespace {

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const GraniteTensorNamingPolicy policy;
    return std::shared_ptr<const ITensorNamingPolicy>(&policy, [](const ITensorNamingPolicy*) {});
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
        const bool gguf = source.is_gguf();
        RuntimeTopology t = resolve_granite_topology(source);

        ResolvedModel result;
        result.is_gguf = gguf;
        result.topology = t;
        result.architecture_id = "granite";
        result.checkpoint_profile_id = "granite";
        result.chat_profile_id = "granite-instruct";
        result.profile = {"granite", "", {}, result.chat_profile_id};
        result.identity = "granite-" + t.fingerprint();
        result.tensor_naming = naming_policy();
        result.capabilities = {true, true, false, true};
        const AttentionSpec& attention = t.attention_layouts.front();
        result.definition.dimensions = {t.hidden, t.intermediate, t.num_hidden_layers,
            attention.query_heads, attention.key_value_heads, attention.head_dim, t.vocab_size,
            t.max_position_embeddings};
        result.definition.rope = {PositionalEncodingKind::Rope, attention.rope_theta, {}};
        result.definition.numerics = {t.norm_eps, t.embedding_multiplier,
            t.attention_multiplier, 1.0f, t.residual_multiplier, t.logits_divisor};
        result.definition.tokens = {t.bos_token_id, t.eos_token_id, t.pad_token_id};
        result.definition.architecture = "granite";
        result.definition.source_format = gguf ? "gguf" : "safetensors";
        result.definition.validate();
        build_dense_transformer_graph(result);
        build_dense_weight_plan(result);
        return result;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_granite_architecture() {
    return std::make_unique<GraniteArchitecture>();
}

} // namespace celeg::detail

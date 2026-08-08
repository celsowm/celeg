#include "celeg/models/nanbeige/architecture.hpp"

#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weight_plan.hpp"
#include "naming_policy.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace celeg::detail {
namespace {

bool is_nanbeige42(const CheckpointMetadata& metadata) {
    return metadata.architecture_type() == "nanbeige" &&
        metadata.integer_or("hidden_size", 0) == 3072 &&
        metadata.integer_or("intermediate_size", 0) == 10752 &&
        metadata.integer_or("num_hidden_layers", 0) == 22 &&
        metadata.integer_or("num_attention_heads", 0) == 48 &&
        metadata.integer_or("num_key_value_heads", 0) == 8 &&
        metadata.integer_or("vocab_size", 0) == 166144;
}

int required(const CheckpointMetadata& metadata, std::string_view key) {
    return static_cast<int>(metadata.integer_for(key, key));
}

RuntimeTopology resolve_topology(const CheckpointMetadata& metadata) {
    if (metadata.contains("ngram_vocab_size") || metadata.contains("num_ngram_layers") ||
        metadata.contains("use_hyper_connections") || metadata.contains("depth_attention")) {
        throw std::runtime_error("Nanbeige variants with n-gram, hyper-connection or depth attention are unsupported");
    }
    RuntimeTopology topology;
    const int physical_layers = required(metadata, "num_hidden_layers");
    const int loops = static_cast<int>(metadata.integer_or("num_loops", 2));
    if (loops != 2) throw std::runtime_error("Nanbeige42 requires exactly two layer loops");
    topology.hidden = required(metadata, "hidden_size");
    topology.intermediate = required(metadata, "intermediate_size");
    topology.dense_intermediate = topology.intermediate;
    topology.max_feed_forward_intermediate = topology.intermediate;
    topology.num_hidden_layers = physical_layers * loops;
    topology.vocab_size = required(metadata, "vocab_size");
    topology.max_position_embeddings = required(metadata, "max_position_embeddings");
    topology.token_policy.bos_token_id = static_cast<int>(metadata.integer_or("bos_token_id", 166100));
    topology.token_policy.eos_token_ids = {static_cast<int>(metadata.integer_or("eos_token_id", 166101))};
    topology.token_policy.pad_token_id = static_cast<int>(metadata.integer_or("pad_token_id", 0));
    topology.numerical_policy.norm_eps = static_cast<float>(metadata.number_for_or(
        "rms_norm_eps", "rms_norm_eps", 1.0e-5));
    topology.numerical_policy.attention_multiplier = 1.0f / std::sqrt(128.0f);
    topology.rope_type = "default";

    topology.checkpoint_layer_for_layer.resize(static_cast<size_t>(topology.num_hidden_layers));
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        topology.checkpoint_layer_for_layer[static_cast<size_t>(layer)] = layer % physical_layers;
    }
    topology.norm_after_layers = {physical_layers - 1};
    topology.mixer_kinds.assign(static_cast<size_t>(topology.num_hidden_layers), MixerKind::Attention);
    topology.feed_forward_kinds.assign(static_cast<size_t>(topology.num_hidden_layers), FeedForwardKind::Dense);
    topology.attention_layer_count = topology.num_hidden_layers;
    topology.layer_for_attention_slot.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.attention_slot_for_layer.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.feed_forward_intermediates.assign(static_cast<size_t>(topology.num_hidden_layers), topology.intermediate);
    topology.feed_forward_activations.assign(static_cast<size_t>(topology.num_hidden_layers), ActivationKind::SwiGLU);
    const double rope_theta = metadata.number_for_or("rope_theta", "rope.freq_base", 70000000.0);
    topology.attention_layouts.assign(static_cast<size_t>(topology.num_hidden_layers),
        AttentionSpec{48, 8, 128, false, AttentionMaskKind::Causal, 0, rope_theta,
                      1.0, {}, topology.numerical_policy.attention_multiplier});
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        topology.layer_for_attention_slot[static_cast<size_t>(layer)] = layer;
        topology.attention_slot_for_layer[static_cast<size_t>(layer)] = layer;
    }
    topology.validate();
    return topology;
}

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const auto policy = std::make_shared<const Nanbeige42TensorNamingPolicy>();
    return policy;
}

class Nanbeige42Architecture final : public IArchitecture {
public:
    std::string_view id() const override { return "nanbeige42"; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const bool supported = is_nanbeige42(metadata);
        return {supported, supported ? 220 : 0,
                supported ? "Nanbeige4.2-3B checkpoint" : "not Nanbeige4.2-3B"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("Nanbeige4.2 architecture cannot resolve checkpoint");
        }
        ArchitectureResolutionStages stages;
        stages.topology = [](const CheckpointView& view) { return resolve_topology(view.metadata); };
        stages.graph = [](ResolvedModel& model, const CheckpointView&) { build_dense_transformer_graph(model); };
        stages.weights = [](ResolvedModel& model, const CheckpointView&) {
            build_dense_weight_plan(model, *naming_policy());
        };
        stages.capabilities = {true, true, false, false};
        stages.provenance.architecture_id = "nanbeige42";
        stages.provenance.source_format = checkpoint.metadata.is_gguf() ? "gguf" : "safetensors";
        stages.provenance.checkpoint_profile_id = "nanbeige42";
        stages.provenance.chat_profile_id = "nanbeige42-instruct";
        stages.provenance.profile = {"nanbeige42", "", {}, stages.provenance.chat_profile_id};
        ResolvedModel result = resolve_architecture_stages(checkpoint, std::move(stages));
        result.provenance.identity = "nanbeige42-" + result.topology.fingerprint();
        return result;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_nanbeige42_architecture() {
    return std::make_unique<Nanbeige42Architecture>();
}

void register_nanbeige42_architecture(ArchitectureCatalog& catalog) {
    catalog.add(make_nanbeige42_architecture());
}

} // namespace celeg::detail

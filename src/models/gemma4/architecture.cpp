#include "celeg/detail/model/builtin_architectures.hpp"

#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weights/roles.hpp"
#include "naming_policy.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace celeg::detail {
namespace {

constexpr std::string_view kTextPrefix = "text_config.";

std::string text_key(std::string_view key) {
    return std::string(kTextPrefix) + std::string(key);
}

int required_int(const CheckpointMetadata& metadata, std::string_view key) {
    return static_cast<int>(metadata.integer(text_key(key)));
}

int optional_int(const CheckpointMetadata& metadata, std::string_view key, int fallback) {
    const std::string flattened = text_key(key);
    return metadata.contains(flattened)
        ? static_cast<int>(metadata.integer(flattened)) : fallback;
}

double optional_number(const CheckpointMetadata& metadata, std::string_view key,
                       double fallback) {
    const std::string flattened = text_key(key);
    return metadata.contains(flattened) ? metadata.number(flattened) : fallback;
}

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const Gemma4TensorNamingPolicy policy;
    return std::shared_ptr<const ITensorNamingPolicy>(&policy,
        [](const ITensorNamingPolicy*) {});
}

void add_request(ResolvedModel& model, TensorRequest request) {
    if (model.tensor_naming) {
        const auto names = model.tensor_naming->candidates(request);
        if (!names.empty()) model.tensor_bindings.source_names.push_back(names.front());
    }
    model.weight_plan.requests.push_back(std::move(request));
}

int group_for_type(std::string_view layer_type) {
    return layer_type == "sliding_attention" ? 0 : 1;
}

RuntimeTopology decode_topology(const CheckpointMetadata& metadata) {
    if (metadata.is_gguf()) {
        throw std::runtime_error("Gemma 4 native SafeTensors support does not accept GGUF");
    }
    if (metadata.architecture_type() != "gemma4" ||
        metadata.string_or(text_key("model_type"), {}) != "gemma4_text") {
        throw std::runtime_error("Gemma 4 requires model_type=gemma4 and text_config.model_type=gemma4_text");
    }

    RuntimeTopology topology;
    topology.hidden = required_int(metadata, "hidden_size");
    topology.intermediate = required_int(metadata, "intermediate_size");
    topology.dense_intermediate = topology.intermediate;
    topology.max_feed_forward_intermediate = topology.intermediate;
    topology.num_hidden_layers = required_int(metadata, "num_hidden_layers");
    const int query_heads = required_int(metadata, "num_attention_heads");
    const int local_kv_heads = required_int(metadata, "num_key_value_heads");
    const int local_head_dim = required_int(metadata, "head_dim");
    topology.vocab_size = required_int(metadata, "vocab_size");
    topology.max_position_embeddings = required_int(metadata, "max_position_embeddings");
    topology.conv_cache = 0;
    topology.conv_dim = 0;
    topology.norm_eps = static_cast<float>(optional_number(metadata, "rms_norm_eps", 1.0e-6));
    topology.rope_type = metadata.string_or(
        text_key("rope_parameters.sliding_attention.rope_type"), "default");
    topology.bos_token_id = optional_int(metadata, "bos_token_id", 2);
    topology.eos_token_id = optional_int(metadata, "eos_token_id", 1);
    topology.pad_token_id = optional_int(metadata, "pad_token_id", 0);
    topology.embedding_multiplier = std::sqrt(static_cast<float>(topology.hidden));
    topology.attention_multiplier = 1.0f;
    topology.residual_multiplier = 1.0f;
    topology.logits_divisor = 1.0f;
    topology.final_logit_softcap = static_cast<float>(optional_number(
        metadata, "final_logit_softcapping", 0.0));
    topology.has_split_attention_norms = true;
    topology.has_per_layer_input = optional_int(metadata, "hidden_size_per_layer_input", 0) > 0;
    topology.per_layer_input_size = optional_int(metadata, "hidden_size_per_layer_input", 0);

    const std::string schedule_key = text_key("layer_types");
    std::vector<std::string> layer_types;
    if (metadata.contains(schedule_key)) {
        layer_types = metadata.strings(schedule_key);
    } else {
        layer_types.resize(static_cast<size_t>(topology.num_hidden_layers));
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            layer_types[static_cast<size_t>(layer)] =
                ((layer + 1) % 6 == 0 || layer == topology.num_hidden_layers - 1)
                ? "full_attention" : "sliding_attention";
        }
    }
    if (static_cast<int>(layer_types.size()) != topology.num_hidden_layers) {
        throw std::runtime_error("Gemma 4 layer_types length does not match num_hidden_layers");
    }

    const int global_head_dim = optional_int(metadata, "global_head_dim", 512);
    const int global_kv_heads = optional_int(metadata, "num_global_key_value_heads",
                                             local_kv_heads);
    const int sliding_window = optional_int(metadata, "sliding_window", 512);
    const int shared_start = topology.num_hidden_layers -
        optional_int(metadata, "num_kv_shared_layers", 0);
    if (shared_start < 0 || shared_start > topology.num_hidden_layers) {
        throw std::runtime_error("Gemma 4 num_kv_shared_layers is out of range");
    }

    topology.mixer_kinds.assign(static_cast<size_t>(topology.num_hidden_layers),
                                MixerKind::Attention);
    topology.layer_types = topology.mixer_kinds;
    topology.attention_slot_for_layer.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.layer_for_attention_slot.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.attention_layouts.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.feed_forward_intermediates.assign(
        static_cast<size_t>(topology.num_hidden_layers), topology.intermediate);
    topology.feed_forward_activations.assign(
        static_cast<size_t>(topology.num_hidden_layers), ActivationKind::GeluTanh);
    topology.attention_layer_count = topology.num_hidden_layers;
    topology.conv_layer_count = 0;

    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        const bool full = layer_types[static_cast<size_t>(layer)] == "full_attention";
        if (!full && layer_types[static_cast<size_t>(layer)] != "sliding_attention") {
            throw std::runtime_error("Gemma 4 has an unsupported attention layer type");
        }
        topology.attention_slot_for_layer[static_cast<size_t>(layer)] = layer;
        topology.layer_for_attention_slot[static_cast<size_t>(layer)] = layer;
        const std::string type = full ? "full_attention" : "sliding_attention";
        const std::string rope_prefix = text_key(
            std::string("rope_parameters.") + type + ".");
        const double theta = metadata.number_or(rope_prefix + "rope_theta",
                                                full ? 1.0e6 : 1.0e4);
        const double rotary_fraction = metadata.number_or(
            rope_prefix + "partial_rotary_factor", 1.0);
        const bool is_shared = layer >= shared_start && shared_start > 0;
        const int group = group_for_type(type);
        bool publishes = false;
        if (!is_shared && shared_start > 0) {
            publishes = true;
            for (int later = layer + 1; later < shared_start; ++later) {
                if (layer_types[static_cast<size_t>(later)] == type) {
                    publishes = false;
                    break;
                }
            }
        }
        AttentionSpec spec;
        spec.query_heads = query_heads;
        spec.key_value_heads = full ? global_kv_heads : local_kv_heads;
        spec.head_dim = full ? global_head_dim : local_head_dim;
        spec.query_key_norm = true;
        spec.mask = full ? AttentionMaskKind::Causal : AttentionMaskKind::SlidingCausal;
        spec.sliding_window = full ? 0 : sliding_window;
        spec.rope_theta = theta;
        spec.rotary_fraction = rotary_fraction;
        spec.kv_sharing = (is_shared || publishes)
            ? KvSharingSpec{group, publishes} : KvSharingSpec{};
        topology.attention_layouts[static_cast<size_t>(layer)] = spec;
        if (spec.kv_sharing.shared()) ++topology.shared_kv_group_count;
    }
    // Count each group once rather than once per owner/consumer layer.
    topology.shared_kv_group_count = 0;
    for (int group = 0; group < 2; ++group) {
        const bool present = std::any_of(topology.attention_layouts.begin(),
            topology.attention_layouts.end(), [group](const AttentionSpec& spec) {
                return spec.kv_sharing.group == group;
            });
        if (present) ++topology.shared_kv_group_count;
    }
    topology.validate();
    return topology;
}

void build_gemma_weight_plan(ResolvedModel& model) {
    const RuntimeTopology& t = model.topology;
    const int per_layer_size = model.graph.layers.empty()
        ? 0 : model.graph.layers.front().per_layer_input.input_size;
    add_request(model, {TensorRole::TokenEmbedding, -1, -1,
                        {t.vocab_size, t.hidden}});
    add_request(model, {TensorRole::LanguageModelHead, -1, -1,
                        {t.vocab_size, t.hidden}});
    add_request(model, {TensorRole::FinalNorm, -1, -1, {t.hidden}});
    if (per_layer_size > 0) {
        add_request(model, {TensorRole::PerLayerEmbedding, -1, -1,
                            {t.vocab_size, t.num_hidden_layers * per_layer_size}});
        add_request(model, {TensorRole::PerLayerContextProjection, -1, -1,
                            {t.num_hidden_layers * per_layer_size, t.hidden}});
        add_request(model, {TensorRole::PerLayerProjectionNorm, -1, -1,
                            {per_layer_size}});
    }
    for (int layer = 0; layer < t.num_hidden_layers; ++layer) {
        const AttentionSpec& attention = t.attention_layout(layer);
        const int q_width = attention.query_heads * attention.head_dim;
        const int kv_width = attention.key_value_heads * attention.head_dim;
        const LayerSpec& graph_layer = model.graph.layers[static_cast<size_t>(layer)];
        add_request(model, {TensorRole::AttentionInputNorm, layer, -1, {t.hidden}});
        add_request(model, {TensorRole::AttentionQuery, layer, -1, {q_width, t.hidden}});
        add_request(model, {TensorRole::AttentionQueryNorm, layer, -1,
                            {attention.head_dim}});
        if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
            add_request(model, {TensorRole::AttentionKey, layer, -1, {kv_width, t.hidden}});
            add_request(model, {TensorRole::AttentionKeyNorm, layer, -1,
                                {attention.head_dim}});
            add_request(model, {TensorRole::AttentionValue, layer, -1,
                                {kv_width, t.hidden}});
        }
        add_request(model, {TensorRole::AttentionOutput, layer, -1,
                            {t.hidden, q_width}});
        add_request(model, {TensorRole::AttentionPostNorm, layer, -1, {t.hidden}});
        add_request(model, {TensorRole::FfnInputNorm, layer, -1, {t.hidden}});
        const auto& dense = std::get<DenseFeedForwardSpec>(graph_layer.feed_forward);
        add_request(model, {TensorRole::FfnGate, layer, -1,
                            {dense.intermediate_size, t.hidden}});
        add_request(model, {TensorRole::FfnUp, layer, -1,
                            {dense.intermediate_size, t.hidden}});
        add_request(model, {TensorRole::FfnDown, layer, -1,
                            {t.hidden, dense.intermediate_size}});
        add_request(model, {TensorRole::FfnOutputNorm, layer, -1, {t.hidden}});
        if (per_layer_size > 0) {
            add_request(model, {TensorRole::PerLayerInputGate, layer, -1,
                                {per_layer_size, t.hidden}});
            add_request(model, {TensorRole::PerLayerProjection, layer, -1,
                                {t.hidden, per_layer_size}});
            add_request(model, {TensorRole::PerLayerInputNorm, layer, -1, {t.hidden}});
            add_request(model, {TensorRole::LayerScalar, layer, -1, {1}});
        }
    }
}

class Gemma4Architecture final : public IArchitecture {
public:
    std::string_view id() const override { return "gemma4"; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        if (metadata.is_gguf()) return {false, 0, "Gemma 4 GGUF is not supported"};
        const bool match = metadata.architecture_type() == "gemma4" &&
            metadata.string_or(text_key("model_type"), {}) == "gemma4_text";
        return {match, match ? 200 : 0,
                match ? "Gemma 4 text configuration" : "not a Gemma 4 text configuration"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        const ProbeResult result = probe(checkpoint.metadata);
        if (!result.supported) throw std::runtime_error(result.reason);
        RuntimeTopology topology = decode_topology(checkpoint.metadata);
        ResolvedModel model;
        model.topology = topology;
        model.architecture_id = "gemma4";
        model.tensor_naming = naming_policy();
        model.chat_profile_id = "gemma4-instruct";
        model.checkpoint_profile_id = checkpoint.metadata.repository_hint.empty()
            ? "gemma4" : checkpoint.metadata.repository_hint;
        model.profile = {"gemma4", "", {}, model.chat_profile_id};
        model.identity = model.checkpoint_profile_id + "-" + topology.fingerprint();
        model.capabilities = {true, true, false, true};
        const AttentionSpec& attention = topology.attention_layouts.front();
        model.definition.dimensions = {topology.hidden, topology.intermediate,
            topology.num_hidden_layers, attention.query_heads,
            attention.key_value_heads, attention.head_dim, topology.vocab_size,
            topology.max_position_embeddings};
        model.definition.rope = {PositionalEncodingKind::Rope, attention.rope_theta, {}};
        model.definition.numerics = {topology.norm_eps, topology.embedding_multiplier,
            topology.attention_multiplier, 1.0f, topology.residual_multiplier,
            topology.logits_divisor};
        model.definition.tokens = {topology.bos_token_id, topology.eos_token_id,
                                   topology.pad_token_id};
        model.definition.architecture = "gemma4";
        model.definition.source_format = "safetensors";
        model.definition.validate();
        model.graph.embedding_multiplier = topology.embedding_multiplier;
        model.graph.logits_divisor = topology.logits_divisor;
        model.graph.final_norm.epsilon = topology.norm_eps;
        model.graph.final_logit_softcap = static_cast<float>(
            optional_number(checkpoint.metadata, "final_logit_softcapping", 0.0));
        const int per_layer_size = optional_int(checkpoint.metadata,
                                                "hidden_size_per_layer_input", 0);
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            LayerSpec spec;
            spec.operator_norm.epsilon = topology.norm_eps;
            spec.post_attention_norm.epsilon = topology.norm_eps;
            spec.pre_feed_forward_norm.epsilon = topology.norm_eps;
            spec.post_feed_forward_norm.epsilon = topology.norm_eps;
            spec.per_layer_input_norm.epsilon = topology.norm_eps;
            spec.mixer = topology.attention_layout(layer);
        const bool wide = layer >= topology.num_hidden_layers -
                optional_int(checkpoint.metadata, "num_kv_shared_layers", 0) &&
                checkpoint.metadata.boolean_or(text_key("use_double_wide_mlp"), false);
        spec.feed_forward = DenseFeedForwardSpec{
                topology.intermediate * (wide ? 2 : 1), ActivationKind::GeluTanh};
            topology.max_feed_forward_intermediate = std::max(
                topology.max_feed_forward_intermediate,
                topology.intermediate * (wide ? 2 : 1));
            topology.feed_forward_intermediates[static_cast<size_t>(layer)] =
                topology.intermediate * (wide ? 2 : 1);
            spec.per_layer_input = {per_layer_size, ActivationKind::GeluTanh,
                                    per_layer_size > 0};
            spec.layer_scalar = 1.0f;
            model.graph.layers.push_back(std::move(spec));
        }
        model.topology = topology;
        build_gemma_weight_plan(model);
        return model;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_gemma4_architecture() {
    return std::make_unique<Gemma4Architecture>();
}

} // namespace celeg::detail

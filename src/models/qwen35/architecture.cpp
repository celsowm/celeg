#include "celeg/models/qwen35/architecture.hpp"

#include "celeg/model/weight_plan.hpp"
#include "naming_policy.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace celeg::detail {
namespace {

constexpr std::string_view kText = "text_config.";
std::string key(std::string_view suffix) { return std::string(kText) + std::string(suffix); }

int required_int(const CheckpointMetadata& metadata, std::string_view name) {
    return static_cast<int>(metadata.integer(key(name)));
}

int optional_int(const CheckpointMetadata& metadata, std::string_view name, int fallback) {
    return static_cast<int>(metadata.integer_or(key(name), fallback));
}

double optional_number(const CheckpointMetadata& metadata, std::string_view name, double fallback) {
    return metadata.number_or(key(name), fallback);
}

bool is_qwen35(const CheckpointMetadata& metadata) {
    const std::string outer = metadata.string_or("model_type", {});
    const std::string nested = metadata.string_or(key("model_type"), {});
    return !metadata.is_gguf() &&
        ((outer == "qwen3_5" && nested == "qwen3_5_text") ||
         (outer == "qwen3_5_moe" && nested == "qwen3_5_moe_text"));
}

bool is_qwen35_moe(const CheckpointMetadata& metadata) {
    return !metadata.is_gguf() &&
        metadata.string_or("model_type", {}) == "qwen3_5_moe" &&
        metadata.string_or(key("model_type"), {}) == "qwen3_5_moe_text";
}

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const auto policy = std::make_shared<const Qwen35TensorNamingPolicy>();
    return policy;
}

RuntimeTopology decode_topology(const CheckpointMetadata& metadata) {
    if (!is_qwen35(metadata)) throw std::runtime_error("Qwen 3.5 requires Qwen 3.5 text metadata");
    RuntimeTopology result;
    result.hidden = required_int(metadata, "hidden_size");
    const int moe_intermediate = optional_int(metadata, "moe_intermediate_size", 0);
    result.intermediate = optional_int(metadata, "intermediate_size",
        moe_intermediate > 0 ? moe_intermediate : 0);
    if (result.intermediate <= 0) throw std::runtime_error("Qwen 3.5 has no FFN intermediate size");
    result.dense_intermediate = result.intermediate;
    result.moe_intermediate = moe_intermediate;
    result.shared_expert_intermediate = optional_int(metadata,
        "shared_expert_intermediate_size", 0);
    result.max_feed_forward_intermediate = std::max(result.intermediate,
        std::max(result.moe_intermediate, result.shared_expert_intermediate));
    result.num_hidden_layers = required_int(metadata, "num_hidden_layers");
    result.mtp_num_hidden_layers = optional_int(metadata, "mtp_num_hidden_layers", 0);
    result.vocab_size = required_int(metadata, "vocab_size");
    result.max_position_embeddings = required_int(metadata, "max_position_embeddings");
    result.token_policy.bos_token_id = optional_int(metadata, "bos_token_id", 1);
    result.token_policy.eos_token_ids = {required_int(metadata, "eos_token_id")};
    if (std::find(result.token_policy.eos_token_ids.begin(),
                  result.token_policy.eos_token_ids.end(), 248046) ==
        result.token_policy.eos_token_ids.end()) {
        result.token_policy.eos_token_ids.push_back(248046);
    }
    result.token_policy.pad_token_id = optional_int(metadata, "pad_token_id", 248046);
    result.numerical_policy.norm_eps = static_cast<float>(optional_number(metadata, "rms_norm_eps", 1.0e-6));
    result.numerical_policy.embedding_multiplier = 1.0f;
    result.numerical_policy.attention_multiplier = 1.0f / std::sqrt(static_cast<float>(required_int(metadata, "head_dim")));
    result.numerical_policy.residual_multiplier = 1.0f;
    result.numerical_policy.logits_divisor = 1.0f;
    result.numerical_policy.rms_norm_add_one = true;
    const bool moe_model = is_qwen35_moe(metadata);
    result.num_experts = optional_int(metadata, "num_experts", 0);
    result.experts_per_token = optional_int(metadata, "num_experts_per_tok", 0);
    result.normalize_topk = moe_model;
    result.moe_router_softmax = moe_model;
    result.feed_forward_kinds.assign(static_cast<size_t>(result.num_hidden_layers),
        moe_model ? FeedForwardKind::MixtureOfExperts : FeedForwardKind::Dense);
    result.feed_forward_intermediates.assign(static_cast<size_t>(result.num_hidden_layers),
        moe_model ? result.moe_intermediate : result.intermediate);
    result.feed_forward_activations.assign(static_cast<size_t>(result.num_hidden_layers), ActivationKind::SwiGLU);
    result.attention_layouts.resize(static_cast<size_t>(result.num_hidden_layers));
    result.gated_delta_net_layouts.resize(static_cast<size_t>(result.num_hidden_layers));
    result.attention_slot_for_layer.assign(static_cast<size_t>(result.num_hidden_layers), -1);

    const auto kinds = metadata.strings(key("layer_types"));
    if (static_cast<int>(kinds.size()) != result.num_hidden_layers) {
        throw std::runtime_error("Qwen 3.5 layer_types length does not match num_hidden_layers");
    }
    const int q_heads = required_int(metadata, "num_attention_heads");
    const int kv_heads = required_int(metadata, "num_key_value_heads");
    const int head_dim = required_int(metadata, "head_dim");
    const double rope_theta = optional_number(metadata, "rope_parameters.rope_theta", 1.0e7);
    result.mrope_interleaved = metadata.boolean_or(
        key("rope_parameters.mrope_interleaved"), false);
    if (result.mrope_interleaved) {
        const auto sections = metadata.integers(key("rope_parameters.mrope_section"));
        if (sections.size() != 3) {
            throw std::runtime_error("Qwen 3.5 M-RoPE requires three rope sections");
        }
        for (size_t index = 0; index < sections.size(); ++index) {
            result.mrope_section[index] = static_cast<int>(sections[index]);
        }
    }
    const int linear_key_heads = required_int(metadata, "linear_num_key_heads");
    const int linear_value_heads = required_int(metadata, "linear_num_value_heads");
    const int linear_key_dim = required_int(metadata, "linear_key_head_dim");
    const int linear_value_dim = required_int(metadata, "linear_value_head_dim");
    const int conv_kernel = required_int(metadata, "linear_conv_kernel_dim");
    result.mixer_kinds.resize(static_cast<size_t>(result.num_hidden_layers));
    for (int layer = 0; layer < result.num_hidden_layers; ++layer) {
        if (kinds[static_cast<size_t>(layer)] == "full_attention") {
            result.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::Attention;
            result.attention_slot_for_layer[static_cast<size_t>(layer)] = result.attention_layer_count;
            result.layer_for_attention_slot.push_back(layer);
            ++result.attention_layer_count;
            result.attention_layouts[static_cast<size_t>(layer)] = AttentionSpec{
                q_heads, kv_heads, head_dim, true, AttentionMaskKind::Causal, 0,
                rope_theta, 0.25, {}, result.numerical_policy.attention_multiplier,
                PositionalEncodingKind::Rope, true};
        } else if (kinds[static_cast<size_t>(layer)] == "linear_attention") {
            result.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::GatedDeltaNet;
            result.gated_delta_net_layouts[static_cast<size_t>(layer)] =
                GatedDeltaNetSpec{conv_kernel, linear_key_dim, linear_value_dim,
                                  linear_key_heads, linear_value_heads};
            ++result.gated_delta_net_layer_count;
        } else {
            throw std::runtime_error("unsupported Qwen 3.5 layer type: " + kinds[static_cast<size_t>(layer)]);
        }
    }
    result.validate();
    return result;
}

void build_graph(ResolvedModel& model, const CheckpointMetadata& metadata) {
    const auto& topology = model.topology;
    const int key_heads = required_int(metadata, "linear_num_key_heads");
    const int value_heads = required_int(metadata, "linear_num_value_heads");
    const int key_dim = required_int(metadata, "linear_key_head_dim");
    const int value_dim = required_int(metadata, "linear_value_head_dim");
    const int conv_kernel = required_int(metadata, "linear_conv_kernel_dim");
    model.graph.final_norm.epsilon = topology.numerical_policy.norm_eps;
    model.graph.layers.reserve(static_cast<size_t>(topology.num_hidden_layers));
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        LayerSpec spec;
        spec.operator_norm.epsilon = topology.numerical_policy.norm_eps;
        spec.post_attention_norm.epsilon = topology.numerical_policy.norm_eps;
        spec.feed_forward_norm.epsilon = topology.numerical_policy.norm_eps;
        spec.residual.multiplier = 1.0f;
        if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            spec.mixer = topology.attention_layout(layer);
        } else {
            spec.mixer = GatedDeltaNetSpec{conv_kernel, key_dim, value_dim, key_heads, value_heads};
        }
        if (topology.num_experts > 0) {
            spec.feed_forward = MixtureOfExpertsSpec{
                topology.moe_intermediate, topology.num_experts,
                topology.experts_per_token, true, false, 1.0f,
                0, 0, true, topology.shared_expert_intermediate, false, true};
        } else {
            spec.feed_forward = DenseFeedForwardSpec{topology.intermediate, ActivationKind::SwiGLU};
        }
        model.graph.layers.push_back(std::move(spec));
    }
}

void build_weights(ResolvedModel& model) {
    const auto& topology = model.topology;
    const auto add = [&model](TensorRole role, int layer, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back({role, layer, -1, std::move(shape)});
    };
    add(TensorRole::TokenEmbedding, -1, {topology.vocab_size, topology.hidden});
    add(TensorRole::LanguageModelHead, -1, {topology.vocab_size, topology.hidden});
    add(TensorRole::FinalNorm, -1, {topology.hidden});
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        add(TensorRole::AttentionInputNorm, layer, {topology.hidden});
        if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            const auto& a = topology.attention_layout(layer);
            add(TensorRole::AttentionQuery, layer, {a.query_projection_width(), topology.hidden});
            add(TensorRole::AttentionQueryNorm, layer, {a.head_dim});
            add(TensorRole::AttentionKey, layer, {a.key_value_width(), topology.hidden});
            add(TensorRole::AttentionKeyNorm, layer, {a.head_dim});
            add(TensorRole::AttentionValue, layer, {a.key_value_width(), topology.hidden});
            add(TensorRole::AttentionOutput, layer, {topology.hidden, a.query_width()});
        } else {
            const auto& d = std::get<GatedDeltaNetSpec>(model.graph.layers[static_cast<size_t>(layer)].mixer);
            const int qkv = 2 * d.key_heads * d.key_head_dim + d.value_heads * d.value_head_dim;
            add(TensorRole::GatedDeltaNetQkv, layer, {qkv, topology.hidden});
            add(TensorRole::GatedDeltaNetZ, layer, {d.value_heads * d.value_head_dim, topology.hidden});
            add(TensorRole::GatedDeltaNetAlpha, layer, {d.value_heads, topology.hidden});
            add(TensorRole::GatedDeltaNetBeta, layer, {d.value_heads, topology.hidden});
            add(TensorRole::GatedDeltaNetDtBias, layer, {d.value_heads});
            add(TensorRole::GatedDeltaNetALog, layer, {d.value_heads});
            add(TensorRole::GatedDeltaNetConv, layer, {qkv, 1, d.conv_kernel});
            add(TensorRole::GatedDeltaNetNorm, layer, {d.value_head_dim});
            add(TensorRole::GatedDeltaNetOutput, layer, {topology.hidden, d.value_heads * d.value_head_dim});
        }
        add(TensorRole::FfnInputNorm, layer, {topology.hidden});
        if (topology.num_experts > 0) {
            add(TensorRole::MoeRouter, layer, {topology.num_experts, topology.hidden});
            add(TensorRole::MoePackedGateUp, layer,
                {topology.num_experts, 2 * topology.moe_intermediate, topology.hidden});
            add(TensorRole::MoePackedDown, layer,
                {topology.num_experts, topology.hidden, topology.moe_intermediate});
            add(TensorRole::MoeSharedGate, layer,
                {topology.shared_expert_intermediate, topology.hidden});
            add(TensorRole::MoeSharedUp, layer,
                {topology.shared_expert_intermediate, topology.hidden});
            add(TensorRole::MoeSharedDown, layer,
                {topology.hidden, topology.shared_expert_intermediate});
            add(TensorRole::MoeSharedGateWeight, layer, {1, topology.hidden});
        } else {
            add(TensorRole::FfnGate, layer, {topology.intermediate, topology.hidden});
            add(TensorRole::FfnUp, layer, {topology.intermediate, topology.hidden});
            add(TensorRole::FfnDown, layer, {topology.hidden, topology.intermediate});
        }
    }
    resolve_weight_plan(model, *naming_policy());
}

class Qwen35Architecture final : public IArchitecture {
public:
    std::string_view id() const override { return "qwen35"; }
    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const bool supported = is_qwen35(metadata);
        return {supported, supported ? 200 : 0, supported ? "Qwen 3.5 checkpoint" : "not Qwen 3.5"};
    }
    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) throw std::runtime_error("Qwen 3.5 architecture cannot resolve checkpoint");
        ArchitectureResolutionStages stages;
        stages.topology = [](const CheckpointView& view) { return decode_topology(view.metadata); };
        stages.graph = [metadata = checkpoint.metadata](ResolvedModel& model, const CheckpointView&) { build_graph(model, metadata); };
        stages.weights = [](ResolvedModel& model, const CheckpointView&) { build_weights(model); };
        stages.capabilities = {true, true, is_qwen35_moe(checkpoint.metadata),
                               checkpoint.metadata.boolean_or("tie_word_embeddings", true)};
        stages.provenance = {"qwen35", "safetensors", {"qwen35", "", {}, "qwen35-instruct"},
                             "qwen35", "qwen35-instruct", {}};
        ResolvedModel model = resolve_architecture_stages(checkpoint, std::move(stages));
        model.provenance.identity = "qwen35-" + model.topology.fingerprint();
        return model;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_qwen35_architecture() { return std::make_unique<Qwen35Architecture>(); }
void register_qwen35_architecture(ArchitectureCatalog& catalog) { catalog.add(make_qwen35_architecture()); }

} // namespace celeg::detail

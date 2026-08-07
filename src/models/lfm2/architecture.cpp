#include "celeg/models/lfm2/architecture.hpp"
#include "celeg/detail/models/lfm2_layer_decoder.hpp"
#include "celeg/detail/models/lfm2_metadata_decoder.hpp"
#include "naming_policy.hpp"

#include "celeg/model/weights/roles.hpp"
#include "celeg/model/weight_plan.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace celeg::detail {
namespace {

int read_int(const CheckpointMetadata& m, std::string_view json_key,
             std::string_view gguf_key) {
    return static_cast<int>(m.integer_for(json_key, gguf_key));
}

int read_int_or(const CheckpointMetadata& m, std::string_view json_key,
                std::string_view gguf_key, int fallback) {
    return static_cast<int>(m.integer_for_or(json_key, gguf_key, fallback));
}

double read_number(const CheckpointMetadata& m, std::string_view json_key,
                   std::string_view gguf_key, double fallback) {
    return m.number_for_or(json_key, gguf_key, fallback);
}

std::string read_string(const CheckpointMetadata& m, std::string_view json_key,
                        std::string_view gguf_key, std::string fallback = {}) {
    return m.string_for_or(json_key, gguf_key, std::move(fallback));
}

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const auto policy = std::make_shared<const CelegTensorNamingPolicy>();
    return policy;
}

void add_request(ResolvedModel& model, TensorRequest request) {
    model.weight_plan.requests.push_back(std::move(request));
}

void build_weight_plan(ResolvedModel& model) {
    const RuntimeTopology& t = model.topology;
    add_request(model, {TensorRole::TokenEmbedding, -1, -1,
                        {t.vocab_size, t.hidden}});
    add_request(model, {TensorRole::FinalNorm, -1, -1, {t.hidden}});
    add_request(model, {TensorRole::LanguageModelHead, -1, -1,
                        {t.vocab_size, t.hidden}});
    for (int layer = 0; layer < t.num_hidden_layers; ++layer) {
        add_request(model, {TensorRole::AttentionInputNorm, layer, -1, {t.hidden}});
        if (t.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            add_request(model, {TensorRole::AttentionQuery, layer, -1,
                                {t.attention_layout(layer).query_width(), t.hidden}});
            add_request(model, {TensorRole::AttentionKey, layer, -1,
                                {t.attention_layout(layer).key_value_width(), t.hidden}});
            add_request(model, {TensorRole::AttentionValue, layer, -1,
                                {t.attention_layout(layer).key_value_width(), t.hidden}});
            add_request(model, {TensorRole::AttentionOutput, layer, -1,
                                {t.hidden, t.hidden}});
        } else {
            add_request(model, {TensorRole::ShortConvInput, layer, -1,
                                {3 * t.hidden, t.hidden}});
            add_request(model, {TensorRole::ShortConvKernel, layer, -1,
                                {t.hidden, 1, t.conv_cache}});
            add_request(model, {TensorRole::ShortConvOutput, layer, -1,
                                {t.hidden, t.hidden}});
        }
        add_request(model, {TensorRole::FfnInputNorm, layer, -1, {t.hidden}});
        if (t.layer_uses_moe(layer)) {
            add_request(model, {TensorRole::MoeRouter, layer, -1,
                                {t.num_experts, t.hidden}});
            for (int expert = 0; expert < t.num_experts; ++expert) {
                add_request(model, {TensorRole::MoeExpertGate, layer, expert,
                                    {t.moe_intermediate, t.hidden}});
                add_request(model, {TensorRole::MoeExpertUp, layer, expert,
                                    {t.moe_intermediate, t.hidden}});
                add_request(model, {TensorRole::MoeExpertDown, layer, expert,
                                    {t.hidden, t.moe_intermediate}});
            }
        } else {
            add_request(model, {TensorRole::FfnGate, layer, -1,
                                {t.intermediate, t.hidden}});
            add_request(model, {TensorRole::FfnUp, layer, -1,
                                {t.intermediate, t.hidden}});
            add_request(model, {TensorRole::FfnDown, layer, -1,
                                {t.hidden, t.intermediate}});
        }
    }
}

CheckpointMetadata normalized_lfm2_metadata(const CheckpointView& checkpoint) {
    CheckpointProfile profile{"lfm2", "", {}, "lfm2-instruct"};
    return CheckpointProfileResolver::matches(
        profile, checkpoint.metadata)
        ? CheckpointProfileResolver::apply(profile, checkpoint.metadata)
        : checkpoint.metadata;
}

RuntimeTopology decode_lfm2_topology(const CheckpointMetadata& m) {
    const Lfm2Metadata decoded = decode_lfm2_metadata(m);
    const bool gguf = decoded.gguf;
    const bool moe = decoded.moe;
    const std::string& prefix = decoded.tensor_prefix;

    RuntimeTopology t;
    t.hidden = decoded.hidden;
    t.intermediate = decoded.intermediate;
    t.dense_intermediate = t.intermediate;
    t.num_hidden_layers = decoded.num_hidden_layers;
    const int query_heads = decoded.num_attention_heads;
    t.vocab_size = decoded.vocab_size;
    t.conv_cache = decoded.conv_cache;
    t.conv_dim = decoded.conv_dim;
    t.max_position_embeddings = decoded.max_position_embeddings;
    t.numerical_policy.norm_eps = decoded.norm_eps;
    t.token_policy.bos_token_id = decoded.bos_token_id;
    t.token_policy.eos_token_ids = {decoded.eos_token_id};
    t.token_policy.pad_token_id = decoded.pad_token_id;
    t.numerical_policy.embedding_multiplier = 1.0f;
    t.numerical_policy.attention_multiplier = 0.0f;
    t.numerical_policy.residual_multiplier = 1.0f;
    t.numerical_policy.logits_divisor = 1.0f;

    const auto decoded_layers = decode_lfm2_layer_types(m, prefix, decoded.num_key_value_heads);
    t.mixer_kinds = decoded_layers.mixer_kinds;
    const int key_value_heads = decoded_layers.num_key_value_heads;
    const int head_dim = decoded.head_dim;
    if (key_value_heads == 0) throw std::runtime_error("checkpoint has no attention layers");

    if (moe) {
        t.moe_intermediate = decoded.moe_intermediate;
        t.num_dense_layers = decoded.num_dense_layers;
        t.num_experts = decoded.num_experts;
        t.experts_per_token = decoded.experts_per_token;
        t.normalize_topk = decoded.normalize_topk;
        t.use_expert_bias = decoded.use_expert_bias;
        t.routed_scaling_factor = decoded.routed_scaling_factor;
        if (gguf) t.use_expert_bias = false;
    }
    t.feed_forward_kinds.assign(static_cast<size_t>(t.num_hidden_layers),
                                FeedForwardKind::Dense);
    if (moe) {
        for (int layer = t.num_dense_layers; layer < t.num_hidden_layers; ++layer) {
            t.feed_forward_kinds[static_cast<size_t>(layer)] =
                FeedForwardKind::MixtureOfExperts;
        }
    }
    t.attention_layer_count = 0;
    t.conv_layer_count = 0;
    for (int i = 0; i < t.num_hidden_layers; ++i) {
        if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::Attention) {
            t.attention_slot_for_layer.push_back(t.attention_layer_count++);
            t.layer_for_attention_slot.push_back(i);
        } else {
            t.attention_slot_for_layer.push_back(-1);
            ++t.conv_layer_count;
        }
    }
    t.max_feed_forward_intermediate = t.intermediate;
    t.feed_forward_intermediates.assign(static_cast<size_t>(t.num_hidden_layers), t.intermediate);
    t.feed_forward_activations.assign(static_cast<size_t>(t.num_hidden_layers),
                                       ActivationKind::SwiGLU);
    t.attention_layouts.assign(static_cast<size_t>(t.num_hidden_layers),
        AttentionSpec{query_heads, key_value_heads, head_dim,
                      false, AttentionMaskKind::Causal, 0,
                      decoded.rope_theta, 1.0, {}});
    t.validate();
    return t;
}

void build_lfm2_graph(ResolvedModel& result,
                      const CheckpointMetadata& metadata) {
    const RuntimeTopology& t = result.topology;
    result.graph.embedding_multiplier = t.numerical_policy.embedding_multiplier;
    result.graph.logits_divisor = t.numerical_policy.logits_divisor;
    result.graph.final_norm.epsilon = t.numerical_policy.norm_eps;
    for (int i = 0; i < t.num_hidden_layers; ++i) {
        LayerSpec layer;
        layer.operator_norm.epsilon = t.numerical_policy.norm_eps;
        layer.feed_forward_norm.epsilon = t.numerical_policy.norm_eps;
        layer.residual.multiplier = t.numerical_policy.residual_multiplier;
        if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::Attention) {
            layer.mixer = t.attention_layout(i);
        } else {
            layer.mixer = ShortConvolutionSpec{t.conv_cache, t.conv_dim, false};
        }
        if (t.layer_uses_moe(i)) {
            layer.feed_forward = MixtureOfExpertsSpec{
                t.moe_intermediate, t.num_experts, t.experts_per_token,
                t.normalize_topk, t.use_expert_bias, t.routed_scaling_factor};
        } else {
            layer.feed_forward = DenseFeedForwardSpec{t.intermediate};
        }
        result.graph.layers.push_back(std::move(layer));
    }
    (void)metadata;
}

ResolvedModel resolve_lfm2(const CheckpointView& checkpoint) {
    const CheckpointMetadata metadata = normalized_lfm2_metadata(checkpoint);
    const Lfm2Metadata decoded = decode_lfm2_metadata(metadata);
    CheckpointProfile profile{"lfm2", "", {}, "lfm2-instruct"};
    ArchitectureResolutionStages stages;
    stages.topology = [metadata](const CheckpointView&) {
        return decode_lfm2_topology(metadata);
    };
    stages.graph = [metadata](ResolvedModel& model, const CheckpointView&) {
        build_lfm2_graph(model, metadata);
    };
    stages.weights = [](ResolvedModel& model, const CheckpointView&) {
        build_weight_plan(model);
        resolve_weight_plan(model, *naming_policy());
    };
    stages.capabilities = {true, true, decoded.moe, true};
    stages.provenance.architecture_id = "lfm2";
    stages.provenance.source_format = metadata.is_gguf() ? "gguf" : "safetensors";
    stages.provenance.chat_profile_id = "lfm2-instruct";
    stages.provenance.checkpoint_profile_id = metadata.repository_hint.empty()
        ? "lfm2" : metadata.repository_hint;
    stages.provenance.profile = std::move(profile);
    ResolvedModel result = resolve_architecture_stages(checkpoint, std::move(stages));
    result.provenance.identity = result.provenance.checkpoint_profile_id + "-" +
                                 result.topology.fingerprint();
    return result;
}

class Lfm2Architecture final : public IArchitecture {
public:
    std::string_view id() const override { return "lfm2"; }
    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const std::string type = metadata.architecture_type();
        const bool supported = type == "lfm2" || type == "lfm2_moe" || type == "lfm2moe";
        return {supported, supported ? 100 : 0, supported ? "LFM2 checkpoint" : "not LFM2"};
    }
    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        const ProbeResult result = probe(checkpoint.metadata);
        if (!result.supported) throw std::runtime_error("LFM2 architecture cannot resolve checkpoint");
        return resolve_lfm2(checkpoint);
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_lfm2_architecture() {
    return std::make_unique<Lfm2Architecture>();
}

void register_lfm2_architecture(ArchitectureCatalog& catalog) {
    catalog.add(make_lfm2_architecture());
}

} // namespace celeg::detail

#include "celeg/detail/model/builtin_architectures.hpp"

#include "celeg/model/weights/roles.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace celeg::detail {
namespace {

bool contains_ci(std::string_view text, std::string_view needle) {
    std::string lower;
    lower.reserve(text.size());
    for (const char c : text) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    std::string target;
    target.reserve(needle.size());
    for (const char c : needle) target.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return lower.find(target) != std::string::npos;
}

int read_int(const CheckpointMetadata& m, std::string_view json_key,
             std::string_view gguf_key) {
    return static_cast<int>(m.source_format == CheckpointSourceFormat::Gguf
        ? m.integer(gguf_key) : m.integer(json_key));
}

int read_int_or(const CheckpointMetadata& m, std::string_view json_key,
                std::string_view gguf_key, int fallback) {
    const std::string_view key = m.source_format == CheckpointSourceFormat::Gguf
        ? gguf_key : json_key;
    return m.contains(key) ? static_cast<int>(m.integer(key)) : fallback;
}

double read_number(const CheckpointMetadata& m, std::string_view json_key,
                   std::string_view gguf_key, double fallback) {
    const std::string_view key = m.source_format == CheckpointSourceFormat::Gguf
        ? gguf_key : json_key;
    return m.number_or(key, fallback);
}

std::string read_string(const CheckpointMetadata& m, std::string_view json_key,
                        std::string_view gguf_key, std::string fallback = {}) {
    const std::string_view key = m.source_format == CheckpointSourceFormat::Gguf
        ? gguf_key : json_key;
    return m.string_or(key, std::move(fallback));
}

std::vector<int64_t> read_layer_heads(const CheckpointMetadata& m, std::string_view prefix) {
    if (m.source_format == CheckpointSourceFormat::Gguf) {
        return std::get<std::vector<int64_t>>(m.value(std::string(prefix) + ".attention.head_count_kv"));
    }
    const auto values = m.strings("layer_types");
    std::vector<int64_t> heads;
    heads.reserve(values.size());
    for (const std::string& value : values) heads.push_back(value == "conv" ? 0 : 1);
    return heads;
}

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const CelegTensorNamingPolicy policy;
    return std::shared_ptr<const ITensorNamingPolicy>(&policy, [](const ITensorNamingPolicy*) {});
}

void add_request(ResolvedModel& model, TensorRequest request) {
    if (model.tensor_naming) {
        const auto names = model.tensor_naming->candidates(request);
        if (!names.empty()) model.tensor_bindings.source_names.push_back(names.front());
    }
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
                                {t.q_width, t.hidden}});
            add_request(model, {TensorRole::AttentionKey, layer, -1,
                                {t.kv_width, t.hidden}});
            add_request(model, {TensorRole::AttentionValue, layer, -1,
                                {t.kv_width, t.hidden}});
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

ModelDefinition make_definition(const RuntimeTopology& topology,
                                const CheckpointMetadata& metadata) {
    ModelDefinition definition;
    definition.dimensions = {
        topology.hidden, topology.intermediate, topology.num_hidden_layers,
        topology.num_attention_heads, topology.num_key_value_heads, topology.head_dim,
        topology.vocab_size, topology.max_position_embeddings};
    definition.rope.kind = PositionalEncodingKind::Rope;
    definition.rope.theta = topology.rope_theta;
    definition.numerics.norm_epsilon = topology.norm_eps;
    definition.numerics.embedding_multiplier = topology.embedding_multiplier;
    definition.numerics.attention_multiplier = topology.attention_multiplier;
    definition.numerics.residual_multiplier = topology.residual_multiplier;
    definition.numerics.logits_divisor = topology.logits_divisor;
    definition.tokens = {topology.bos_token_id, topology.eos_token_id, topology.pad_token_id};
    definition.architecture = "lfm2";
    definition.source_format = metadata.source_format == CheckpointSourceFormat::Gguf
        ? "gguf" : "safetensors";
    definition.validate();
    return definition;
}

ResolvedModel resolve_lfm2(const CheckpointView& checkpoint) {
    CheckpointProfile profile{"lfm2", "", {}, "lfm2-instruct"};
    if (contains_ci(checkpoint.metadata.repository_hint, "1.2b")) {
        profile = {"lfm2-1.2b", "1.2b",
                   {{"intermediate_size", int64_t{8192}}}, "lfm2-instruct"};
    }
    const CheckpointMetadata metadata = CheckpointProfileResolver::matches(
        profile, checkpoint.metadata)
        ? CheckpointProfileResolver::apply(profile, checkpoint.metadata)
        : checkpoint.metadata;
    const auto& m = metadata;
    const bool gguf = m.source_format == CheckpointSourceFormat::Gguf;
    const std::string model_type = gguf ? m.string("general.architecture")
                                        : m.string("model_type");
    const bool moe = model_type == "lfm2moe" || model_type == "lfm2_moe";
    const std::string prefix = gguf ? (moe ? "lfm2moe" : "lfm2") : "";
    auto key = [&](std::string_view json, std::string_view suffix) {
        return gguf ? prefix + "." + std::string(suffix) : std::string(json);
    };
    auto integer = [&](std::string_view json, std::string_view suffix) {
        return static_cast<int>(m.integer(key(json, suffix)));
    };
    auto integer_or = [&](std::string_view json, std::string_view suffix, int fallback) {
        const std::string full = key(json, suffix);
        return m.contains(full) ? static_cast<int>(m.integer(full)) : fallback;
    };
    auto number = [&](std::string_view json, std::string_view suffix, double fallback) {
        return m.number_or(key(json, suffix), fallback);
    };

    RuntimeTopology t;
    t.hidden = integer("hidden_size", "embedding_length");
    t.intermediate = integer("intermediate_size", "feed_forward_length");
    t.dense_intermediate = t.intermediate;
    t.num_hidden_layers = integer("num_hidden_layers", "block_count");
    t.num_attention_heads = integer("num_attention_heads", "attention.head_count");
    t.num_key_value_heads = gguf
        ? 0 : integer("num_key_value_heads", "attention.head_count_kv");
    t.vocab_size = integer_or("vocab_size", "vocab_size",
                              static_cast<int>(m.integer_or("llama.vocab_size", 0)));
    t.conv_cache = integer("conv_L_cache", "shortconv.l_cache");
    t.conv_dim = integer_or("conv_dim", "embedding_length", t.hidden);
    t.max_position_embeddings = integer("max_position_embeddings", "context_length");
    t.norm_eps = static_cast<float>(number("norm_eps", "attention.layer_norm_rms_epsilon", 1.0e-5));
    t.rope_theta = static_cast<float>(number("rope_theta", "rope.freq_base", 1.0e6));
    t.bos_token_id = integer_or("bos_token_id", "", 1);
    t.eos_token_id = integer_or("eos_token_id", "", 7);
    t.pad_token_id = integer_or("pad_token_id", "", 0);
    t.embedding_multiplier = 1.0f;
    t.attention_multiplier = 0.0f;
    t.residual_multiplier = 1.0f;
    t.logits_divisor = 1.0f;
    t.query_key_norm = true;

    if (gguf) {
        const auto heads = read_layer_heads(m, prefix);
        t.mixer_kinds.reserve(heads.size());
        for (const int64_t value : heads) {
            if (value == 0) t.mixer_kinds.push_back(MixerKind::ShortConvolution);
            else {
                t.mixer_kinds.push_back(MixerKind::Attention);
                if (t.num_key_value_heads == 0) t.num_key_value_heads = static_cast<int>(value);
                else if (t.num_key_value_heads != value) throw std::runtime_error("heterogeneous KV heads");
            }
        }
        t.head_dim = integer_or("head_dim", "attention.key_length",
                                t.hidden / t.num_attention_heads);
    } else {
        const auto layers = m.strings("layer_types");
        for (const std::string& layer : layers) {
            if (layer == "conv") t.mixer_kinds.push_back(MixerKind::ShortConvolution);
            else if (layer == "full_attention") t.mixer_kinds.push_back(MixerKind::Attention);
            else throw std::runtime_error("unsupported LFM2 layer type: " + layer);
        }
        t.head_dim = integer_or("head_dim", "attention.key_length",
                                t.hidden / t.num_attention_heads);
    }
    if (t.num_key_value_heads == 0) throw std::runtime_error("checkpoint has no attention layers");

    if (moe) {
        t.moe_intermediate = integer_or("moe_intermediate_size", "expert_feed_forward_length", 0);
        t.num_dense_layers = integer_or("num_dense_layers", "num_dense_layers", 0);
        t.num_experts = integer_or("num_experts", "expert_count", 0);
        t.experts_per_token = integer_or("num_experts_per_tok", "expert_used_count", 0);
        t.normalize_topk = gguf ? true : m.boolean_or("norm_topk_prob", true);
        t.use_expert_bias = gguf ? false : m.boolean_or("use_expert_bias", false);
        t.routed_scaling_factor = static_cast<float>(m.number_or(
            "routed_scaling_factor", 1.0));
        if (gguf) {
            if (t.num_dense_layers == 0 && checkpoint.repository) {
                while (checkpoint.repository->contains(
                    "model.layers." + std::to_string(t.num_dense_layers) +
                    ".feed_forward.w1.weight")) {
                    ++t.num_dense_layers;
                }
            }
            t.use_expert_bias = checkpoint.repository && checkpoint.repository->contains(
                "model.layers." + std::to_string(t.num_dense_layers) + ".feed_forward.expert_bias.weight");
        }
    }
    if (profile.id == "lfm2-1.2b" && checkpoint.repository && checkpoint.repository->contains(
            "model.layers.0.feed_forward.w1.weight")) {
        const auto shape = checkpoint.repository->tensor(
            "model.layers.0.feed_forward.w1.weight").shape;
        if (!shape.empty()) t.intermediate = static_cast<int>(shape.front());
    } else if (t.intermediate == 12288 && t.hidden == 2048 && t.num_hidden_layers == 16) {
        if (checkpoint.repository && checkpoint.repository->contains(
                "model.layers.0.feed_forward.w1.weight")) {
            const auto shape = checkpoint.repository->tensor(
                "model.layers.0.feed_forward.w1.weight").shape;
            if (!shape.empty()) t.intermediate = static_cast<int>(shape.front());
        } else if (contains_ci(m.repository_hint, "1.2b")) {
            t.intermediate = 8192;
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
    t.layer_types = t.mixer_kinds;
    t.q_width = t.num_attention_heads * t.head_dim;
    t.kv_width = t.num_key_value_heads * t.head_dim;
    t.qkv_width = t.q_width + 2 * t.kv_width;
    t.rope_pairs = t.head_dim / 2;
    t.validate();

    ResolvedModel result;
    result.definition = make_definition(t, m);
    result.topology = t;
    result.architecture_id = "lfm2";
    result.is_gguf = gguf;
    result.tensor_naming = naming_policy();
    result.chat_profile_id = "lfm2-instruct";
    result.checkpoint_profile_id = m.repository_hint.empty() ? "lfm2" : m.repository_hint;
    result.profile = std::move(profile);
    result.identity = result.checkpoint_profile_id + "-" + t.fingerprint();
    result.capabilities = {true, true, moe, true};
    result.graph.embedding_multiplier = t.embedding_multiplier;
    result.graph.logits_divisor = t.logits_divisor;
    result.graph.final_norm.epsilon = t.norm_eps;
    for (int i = 0; i < t.num_hidden_layers; ++i) {
        LayerSpec layer;
        layer.operator_norm.epsilon = t.norm_eps;
        layer.feed_forward_norm.epsilon = t.norm_eps;
        layer.residual.multiplier = t.residual_multiplier;
        if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::Attention) {
            layer.mixer = AttentionSpec{t.num_attention_heads, t.num_key_value_heads,
                                        t.head_dim, t.query_key_norm};
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
    build_weight_plan(result);
    return result;
}

class Lfm2Architecture final : public IArchitecture {
public:
    std::string_view id() const override { return "lfm2"; }
    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const std::string type = metadata.string_or(
            metadata.source_format == CheckpointSourceFormat::Gguf
                ? "general.architecture" : "model_type", {});
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

} // namespace celeg::detail

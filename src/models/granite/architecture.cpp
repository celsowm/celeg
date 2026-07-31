#include "celeg/detail/model/builtin_architectures.hpp"

#include "celeg/model/weights/roles.hpp"

#include <stdexcept>

namespace celeg::detail {
namespace {

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const GraniteTensorNamingPolicy policy;
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
        add_request(model, {TensorRole::AttentionQuery, layer, -1,
                            {t.q_width, t.hidden}});
        add_request(model, {TensorRole::AttentionKey, layer, -1,
                            {t.kv_width, t.hidden}});
        add_request(model, {TensorRole::AttentionValue, layer, -1,
                            {t.kv_width, t.hidden}});
        add_request(model, {TensorRole::AttentionOutput, layer, -1,
                            {t.hidden, t.hidden}});
        add_request(model, {TensorRole::FfnInputNorm, layer, -1, {t.hidden}});
        add_request(model, {TensorRole::FfnGate, layer, -1,
                            {t.intermediate, t.hidden}});
        add_request(model, {TensorRole::FfnUp, layer, -1,
                            {t.intermediate, t.hidden}});
        add_request(model, {TensorRole::FfnDown, layer, -1,
                            {t.hidden, t.intermediate}});
    }
}

class GraniteArchitecture final : public IArchitecture {
public:
    std::string_view id() const override { return "granite"; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const std::string type = metadata.source_format == CheckpointSourceFormat::Gguf
            ? metadata.string_or("general.architecture", {})
            : metadata.string_or("model_type", {});
        return {type == "granite", type == "granite" ? 100 : 0,
                type == "granite" ? "Granite checkpoint" : "not Granite"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("Granite architecture cannot resolve checkpoint");
        }
        const auto& source = checkpoint.metadata;
        const bool gguf = source.source_format == CheckpointSourceFormat::Gguf;
        const std::string prefix = gguf ? "granite." : "";
        auto key = [&](std::string_view json_key, std::string_view gguf_suffix) {
            return gguf ? prefix + std::string(gguf_suffix) : std::string(json_key);
        };
        auto integer = [&](std::string_view json_key, std::string_view gguf_suffix) {
            const std::string full = key(json_key, gguf_suffix);
            return static_cast<int>(source.integer(full));
        };
        auto integer_or = [&](std::string_view json_key, std::string_view gguf_suffix,
                              int fallback) {
            const std::string full = key(json_key, gguf_suffix);
            return source.contains(full) ? static_cast<int>(source.integer(full)) : fallback;
        };
        auto number_or = [&](std::string_view json_key, std::string_view gguf_suffix,
                             double fallback) {
            return source.number_or(key(json_key, gguf_suffix), fallback);
        };
        RuntimeTopology t;
        t.hidden = integer("hidden_size", "embedding_length");
        t.intermediate = integer("intermediate_size", "feed_forward_length");
        t.dense_intermediate = t.intermediate;
        t.num_hidden_layers = integer("num_hidden_layers", "block_count");
        t.num_attention_heads = integer("num_attention_heads", "attention.head_count");
        t.num_key_value_heads = integer("num_key_value_heads", "attention.head_count_kv");
        t.head_dim = integer_or("head_dim", "attention.key_length",
                                t.hidden / t.num_attention_heads);
        t.vocab_size = integer_or("vocab_size", "vocab_size", 0);
        if (gguf && t.vocab_size == 0 && source.contains("tokenizer.ggml.tokens")) {
            t.vocab_size = static_cast<int>(source.strings("tokenizer.ggml.tokens").size());
        }
        t.conv_cache = 1;
        t.conv_dim = t.hidden;
        t.max_position_embeddings = integer("max_position_embeddings", "context_length");
        t.bos_token_id = integer_or("bos_token_id", "", 1);
        t.eos_token_id = integer_or("eos_token_id", "", 2);
        t.pad_token_id = integer_or("pad_token_id", "", 0);
        if (gguf) {
            t.bos_token_id = static_cast<int>(source.integer_or(
                "tokenizer.ggml.bos_token_id", t.bos_token_id));
            t.eos_token_id = static_cast<int>(source.integer_or(
                "tokenizer.ggml.eos_token_id", t.eos_token_id));
            t.pad_token_id = static_cast<int>(source.integer_or(
                "tokenizer.ggml.padding_token_id", t.pad_token_id));
        }
        t.norm_eps = static_cast<float>(number_or(
            "rms_norm_eps", "attention.layer_norm_rms_epsilon",
            source.number_or("norm_eps", 1.0e-5)));
        t.rope_theta = static_cast<float>(number_or("rope_theta", "rope.freq_base", 1.0e6));
        t.embedding_multiplier = static_cast<float>(number_or(
            "embedding_multiplier", "embedding_multiplier", 1.0));
        t.attention_multiplier = static_cast<float>(number_or(
            "attention_multiplier", "attention_multiplier", 1.0));
        t.residual_multiplier = static_cast<float>(number_or(
            "residual_multiplier", "residual_multiplier", 1.0));
        t.logits_divisor = static_cast<float>(number_or(
            "logits_scaling", "logits_scaling", 1.0));
        t.query_key_norm = false;
        t.mixer_kinds.assign(static_cast<size_t>(t.num_hidden_layers), MixerKind::Attention);
        t.attention_layer_count = t.num_hidden_layers;
        t.layer_for_attention_slot.resize(static_cast<size_t>(t.num_hidden_layers));
        t.attention_slot_for_layer.resize(static_cast<size_t>(t.num_hidden_layers));
        for (int i = 0; i < t.num_hidden_layers; ++i) {
            t.layer_for_attention_slot[static_cast<size_t>(i)] = i;
            t.attention_slot_for_layer[static_cast<size_t>(i)] = i;
        }
        t.layer_types = t.mixer_kinds;
        t.q_width = t.num_attention_heads * t.head_dim;
        t.kv_width = t.num_key_value_heads * t.head_dim;
        t.qkv_width = t.q_width + 2 * t.kv_width;
        t.rope_pairs = t.head_dim / 2;
        t.validate();

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
        result.definition.dimensions = {t.hidden, t.intermediate, t.num_hidden_layers,
            t.num_attention_heads, t.num_key_value_heads, t.head_dim, t.vocab_size,
            t.max_position_embeddings};
        result.definition.rope = {PositionalEncodingKind::Rope, t.rope_theta, {}};
        result.definition.numerics = {t.norm_eps, t.embedding_multiplier,
            t.attention_multiplier, 1.0f, t.residual_multiplier, t.logits_divisor};
        result.definition.tokens = {t.bos_token_id, t.eos_token_id, t.pad_token_id};
        result.definition.architecture = "granite";
        result.definition.source_format = gguf ? "gguf" : "safetensors";
        result.definition.validate();
        result.graph.embedding_multiplier = t.embedding_multiplier;
        result.graph.logits_divisor = t.logits_divisor;
        result.graph.final_norm.epsilon = t.norm_eps;
        for (int i = 0; i < t.num_hidden_layers; ++i) {
            LayerSpec layer;
            layer.operator_norm.epsilon = t.norm_eps;
            layer.feed_forward_norm.epsilon = t.norm_eps;
            layer.mixer = AttentionSpec{t.num_attention_heads, t.num_key_value_heads,
                                        t.head_dim, false};
            layer.feed_forward = DenseFeedForwardSpec{t.intermediate};
            layer.residual.multiplier = t.residual_multiplier;
            result.graph.layers.push_back(std::move(layer));
        }
        build_weight_plan(result);
        return result;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_granite_architecture() {
    return std::make_unique<GraniteArchitecture>();
}

} // namespace celeg::detail

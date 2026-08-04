#include "celeg/detail/model/builtin_architectures.hpp"

#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weight_plan.hpp"
#include "naming_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace celeg::detail {
namespace {

bool contains_case_insensitive(std::string_view value, std::string_view needle) {
    if (needle.empty() || value.size() < needle.size()) return false;
    for (std::size_t start = 0; start + needle.size() <= value.size(); ++start) {
        bool match = true;
        for (std::size_t index = 0; index < needle.size(); ++index) {
            const auto lower = [](char ch) {
                return static_cast<char>(ch >= 'A' && ch <= 'Z' ? ch - 'A' + 'a' : ch);
            };
            if (lower(value[start + index]) != lower(needle[index])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

bool is_smollm3(const CheckpointMetadata& metadata) {
    if (!metadata.is_gguf() && metadata.string_or("model_type", {}) == "smollm3") return true;
    if (metadata.is_gguf() && metadata.architecture_type() == "smollm3") return true;
    if (contains_case_insensitive(metadata.repository_hint, "smollm3")) return true;

    const bool dimensions_match = metadata.integer_for_or(
        "hidden_size", "llama.embedding_length", 0) == 2048 &&
        metadata.integer_for_or("intermediate_size", "llama.feed_forward_length", 0) == 11008 &&
        metadata.integer_for_or("num_hidden_layers", "llama.block_count", 0) == 36 &&
        metadata.integer_for_or("num_attention_heads", "llama.attention.head_count", 0) == 16 &&
        metadata.integer_for_or("num_key_value_heads", "llama.attention.head_count_kv", 0) == 4 &&
        metadata.integer_for_or("vocab_size", "llama.vocab_size", 0) == 128256;
    return dimensions_match && (metadata.contains("no_rope_layers") ||
                                metadata.contains("llama.no_rope_layers") ||
                                metadata.contains("llama.no_rope_layer_interval"));
}

std::string gguf_namespace(const CheckpointMetadata& metadata) {
    if (!metadata.is_gguf()) return {};
    if (metadata.contains("smollm3.embedding_length")) return "smollm3.";
    return "llama.";
}

int required_int(const CheckpointMetadata& metadata,
                 std::string_view json_key, std::string_view gguf_suffix) {
    return static_cast<int>(metadata.integer_for(json_key,
        gguf_namespace(metadata) + std::string(gguf_suffix)));
}

int optional_int(const CheckpointMetadata& metadata,
                 std::string_view json_key, std::string_view gguf_suffix, int fallback) {
    return static_cast<int>(metadata.integer_for_or(json_key,
        gguf_namespace(metadata) + std::string(gguf_suffix), fallback));
}

double optional_number(const CheckpointMetadata& metadata,
                       std::string_view json_key, std::string_view gguf_suffix,
                       double fallback) {
    return metadata.number_for_or(json_key,
        gguf_namespace(metadata) + std::string(gguf_suffix), fallback);
}

std::vector<int> eos_ids(const CheckpointMetadata& metadata) {
    if (!metadata.is_gguf() && metadata.contains("eos_token_id")) {
        const auto& value = metadata.value("eos_token_id");
        if (const auto* ids = std::get_if<std::vector<int64_t>>(&value)) {
            std::vector<int> result;
            result.reserve(ids->size());
            for (const int64_t id : *ids) result.push_back(static_cast<int>(id));
            return result;
        }
        return {static_cast<int>(metadata.integer("eos_token_id"))};
    }
    return {static_cast<int>(metadata.integer_or("tokenizer.ggml.eos_token_id", 128012))};
}

bool no_rope_layer(const CheckpointMetadata& metadata, int layer, int interval) {
    const std::string key = metadata.is_gguf()
        ? gguf_namespace(metadata) + "no_rope_layers" : "no_rope_layers";
    if (metadata.contains(key)) {
        const auto values = metadata.integers(key);
        if (layer >= 0 && layer < static_cast<int>(values.size())) {
            return values[static_cast<size_t>(layer)] == 0;
        }
    }
    if (interval <= 1) return false;
    return layer % interval == interval - 1;
}

RuntimeTopology resolve_topology(const CheckpointMetadata& metadata) {
    RuntimeTopology topology;
    topology.hidden = required_int(metadata, "hidden_size", "embedding_length");
    topology.intermediate = required_int(metadata, "intermediate_size", "feed_forward_length");
    topology.dense_intermediate = topology.intermediate;
    topology.max_feed_forward_intermediate = topology.intermediate;
    topology.num_hidden_layers = required_int(metadata, "num_hidden_layers", "block_count");
    const int query_heads = required_int(metadata, "num_attention_heads", "attention.head_count");
    const int key_value_heads = required_int(metadata, "num_key_value_heads", "attention.head_count_kv");
    const int head_dim = optional_int(metadata, "head_dim", "attention.key_length",
                                      topology.hidden / query_heads);
    topology.vocab_size = optional_int(metadata, "vocab_size", "vocab_size", 0);
    if (metadata.is_gguf() && topology.vocab_size == 0 &&
        metadata.contains("tokenizer.ggml.tokens")) {
        topology.vocab_size = static_cast<int>(metadata.strings("tokenizer.ggml.tokens").size());
    }
    topology.max_position_embeddings = required_int(metadata, "max_position_embeddings", "context_length");
    topology.bos_token_id = optional_int(metadata, "bos_token_id", "", 128000);
    topology.eos_token_ids = eos_ids(metadata);
    topology.pad_token_id = optional_int(metadata, "pad_token_id", "", 128004);
    if (metadata.is_gguf()) {
        topology.bos_token_id = static_cast<int>(metadata.integer_or(
            "tokenizer.ggml.bos_token_id", topology.bos_token_id));
        topology.pad_token_id = static_cast<int>(metadata.integer_or(
            "tokenizer.ggml.padding_token_id", topology.pad_token_id));
    }
    topology.norm_eps = static_cast<float>(optional_number(
        metadata, "rms_norm_eps", "attention.layer_norm_rms_epsilon", 1.0e-6));
    const double rope_theta = optional_number(metadata, "rope_theta", "rope.freq_base", 5.0e6);
    const int no_rope_interval = optional_int(
        metadata, "no_rope_layer_interval", "no_rope_layer_interval", 4);
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    topology.embedding_multiplier = 1.0f;
    topology.attention_multiplier = attention_scale;
    topology.residual_multiplier = 1.0f;
    topology.logits_divisor = 1.0f;
    topology.conv_cache = 0;
    topology.conv_dim = 0;
    topology.mixer_kinds.assign(static_cast<size_t>(topology.num_hidden_layers), MixerKind::Attention);
    topology.feed_forward_kinds.assign(static_cast<size_t>(topology.num_hidden_layers), FeedForwardKind::Dense);
    topology.attention_layer_count = topology.num_hidden_layers;
    topology.layer_for_attention_slot.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.attention_slot_for_layer.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.feed_forward_intermediates.assign(static_cast<size_t>(topology.num_hidden_layers), topology.intermediate);
    topology.feed_forward_activations.assign(static_cast<size_t>(topology.num_hidden_layers), ActivationKind::SwiGLU);
    topology.attention_layouts.reserve(static_cast<size_t>(topology.num_hidden_layers));
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        AttentionSpec layout{query_heads, key_value_heads, head_dim, false,
            AttentionMaskKind::Causal, 0, rope_theta, 1.0, {}, 1.0f};
        layout.positional_encoding = no_rope_layer(metadata, layer, no_rope_interval)
            ? PositionalEncodingKind::None : PositionalEncodingKind::Rope;
        topology.attention_layouts.push_back(layout);
        topology.layer_for_attention_slot[static_cast<size_t>(layer)] = layer;
        topology.attention_slot_for_layer[static_cast<size_t>(layer)] = layer;
    }
    topology.validate();
    return topology;
}

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const auto policy = std::make_shared<const SmolLm3TensorNamingPolicy>();
    return policy;
}

class SmolLm3Architecture final : public IArchitecture {
public:
    std::string_view id() const override { return "smollm3"; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const bool supported = is_smollm3(metadata);
        return {supported, supported ? 160 : 0,
                supported ? "SmolLM3 checkpoint" : "not SmolLM3"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("SmolLM3 architecture cannot resolve checkpoint");
        }
        const RuntimeTopology topology = resolve_topology(checkpoint.metadata);
        ResolvedModel result;
        result.topology = topology;
        result.architecture_id = "smollm3";
        result.source_format = checkpoint.metadata.is_gguf() ? "gguf" : "safetensors";
        result.checkpoint_profile_id = "smollm3";
        result.chat_profile_id = "smollm3-instruct";
        result.profile = {"smollm3", "", {}, result.chat_profile_id};
        result.identity = "smollm3-" + topology.fingerprint();
        result.capabilities = {true, true, false, true};
        build_dense_transformer_graph(result);
        build_dense_weight_plan(result, *naming_policy());
        return result;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_smollm3_architecture() {
    return std::make_unique<SmolLm3Architecture>();
}

void register_smollm3_architecture(ArchitectureCatalog& catalog) {
    catalog.add(make_smollm3_architecture());
}

} // namespace celeg::detail

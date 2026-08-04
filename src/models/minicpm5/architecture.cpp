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

namespace celeg::detail {
namespace {

bool contains_case_insensitive(std::string_view value, std::string_view needle) {
    if (needle.empty() || value.size() < needle.size()) return false;
    for (size_t start = 0; start + needle.size() <= value.size(); ++start) {
        bool match = true;
        for (size_t index = 0; index < needle.size(); ++index) {
            const auto lower = [](char c) {
                return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
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

bool is_minicpm5(const CheckpointMetadata& metadata) {
    if (metadata.architecture_type() != "llama") return false;
    if (contains_case_insensitive(metadata.repository_hint, "minicpm5")) return true;

    // GGUF exports can omit the repository name. The official checkpoint's
    // dimensions provide a stable fallback without treating every Llama model
    // as MiniCPM5.
    const bool gguf_shape = metadata.is_gguf() &&
        metadata.integer_or("llama.embedding_length", 0) == 1536 &&
        metadata.integer_or("llama.feed_forward_length", 0) == 4608 &&
        metadata.integer_or("llama.block_count", 0) == 24 &&
        metadata.integer_or("llama.attention.head_count", 0) == 16 &&
        metadata.integer_or("llama.attention.head_count_kv", 0) == 2;
    const bool json_shape = !metadata.is_gguf() &&
        metadata.integer_or("hidden_size", 0) == 1536 &&
        metadata.integer_or("intermediate_size", 0) == 4608 &&
        metadata.integer_or("num_hidden_layers", 0) == 24 &&
        metadata.integer_or("num_attention_heads", 0) == 16 &&
        metadata.integer_or("num_key_value_heads", 0) == 2 &&
        metadata.integer_or("vocab_size", 0) == 130560;
    return gguf_shape || json_shape;
}

int required_int(const CheckpointMetadata& metadata,
                 std::string_view json_key, std::string_view gguf_key) {
    return static_cast<int>(metadata.integer_for(json_key,
        "llama." + std::string(gguf_key)));
}

int optional_int(const CheckpointMetadata& metadata,
                 std::string_view json_key, std::string_view gguf_key, int fallback) {
    return static_cast<int>(metadata.integer_for_or(json_key,
        "llama." + std::string(gguf_key), fallback));
}

double optional_number(const CheckpointMetadata& metadata,
                       std::string_view json_key, std::string_view gguf_key,
                       double fallback) {
    return metadata.number_for_or(json_key, "llama." + std::string(gguf_key), fallback);
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

    std::vector<int> result = {static_cast<int>(metadata.integer_or(
        "tokenizer.ggml.eos_token_id", 1))};
    if (metadata.contains("tokenizer.ggml.eot_token_id")) {
        result.push_back(static_cast<int>(metadata.integer("tokenizer.ggml.eot_token_id")));
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
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
        topology.vocab_size = static_cast<int>(
            metadata.strings("tokenizer.ggml.tokens").size());
    }
    topology.max_position_embeddings = required_int(
        metadata, "max_position_embeddings", "context_length");
    topology.bos_token_id = optional_int(metadata, "bos_token_id", "", 0);
    topology.eos_token_ids = eos_ids(metadata);
    topology.pad_token_id = optional_int(metadata, "pad_token_id", "", 1);
    if (metadata.is_gguf()) {
        topology.bos_token_id = static_cast<int>(metadata.integer_or(
            "tokenizer.ggml.bos_token_id", topology.bos_token_id));
        topology.pad_token_id = static_cast<int>(metadata.integer_or(
            "tokenizer.ggml.padding_token_id", topology.pad_token_id));
    }
    topology.norm_eps = static_cast<float>(optional_number(
        metadata, "rms_norm_eps", "attention.layer_norm_rms_epsilon", 1.0e-6));
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const double rope_theta = optional_number(metadata, "rope_theta", "rope.freq_base", 5.0e6);
    topology.embedding_multiplier = 1.0f;
    topology.attention_multiplier = attention_scale;
    topology.residual_multiplier = 1.0f;
    topology.logits_divisor = 1.0f;
    topology.conv_cache = 0;
    topology.conv_dim = 0;
    topology.mixer_kinds.assign(static_cast<size_t>(topology.num_hidden_layers),
                                MixerKind::Attention);
    topology.feed_forward_kinds.assign(static_cast<size_t>(topology.num_hidden_layers),
                                       FeedForwardKind::Dense);
    topology.attention_layer_count = topology.num_hidden_layers;
    topology.layer_for_attention_slot.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.attention_slot_for_layer.resize(static_cast<size_t>(topology.num_hidden_layers));
    topology.feed_forward_intermediates.assign(static_cast<size_t>(topology.num_hidden_layers),
                                               topology.intermediate);
    topology.feed_forward_activations.assign(static_cast<size_t>(topology.num_hidden_layers),
                                             ActivationKind::SwiGLU);
    topology.attention_layouts.assign(static_cast<size_t>(topology.num_hidden_layers),
        AttentionSpec{query_heads, key_value_heads, head_dim, false,
                      AttentionMaskKind::Causal, 0, rope_theta, 1.0, {}, attention_scale});
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        topology.layer_for_attention_slot[static_cast<size_t>(layer)] = layer;
        topology.attention_slot_for_layer[static_cast<size_t>(layer)] = layer;
    }
    topology.validate();
    return topology;
}

std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const auto policy = std::make_shared<const MiniCpm5TensorNamingPolicy>();
    return policy;
}

class MiniCpm5Architecture final : public IArchitecture {
public:
    std::string_view id() const override { return "minicpm5"; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        const bool supported = is_minicpm5(metadata);
        return {supported, supported ? 150 : 0,
                supported ? "MiniCPM5 checkpoint" : "not MiniCPM5"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("MiniCPM5 architecture cannot resolve checkpoint");
        }
        const RuntimeTopology topology = resolve_topology(checkpoint.metadata);
        ResolvedModel result;
        result.topology = topology;
        result.architecture_id = "minicpm5";
        result.source_format = checkpoint.metadata.is_gguf() ? "gguf" : "safetensors";
        result.checkpoint_profile_id = "minicpm5";
        result.chat_profile_id = "minicpm5-instruct";
        result.profile = {"minicpm5", "", {}, result.chat_profile_id};
        result.identity = "minicpm5-" + topology.fingerprint();
        result.capabilities = {true, true, false, false};
        build_dense_transformer_graph(result);
        build_dense_weight_plan(result, *naming_policy());
        return result;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_minicpm5_architecture() {
    return std::make_unique<MiniCpm5Architecture>();
}

void register_minicpm5_architecture(ArchitectureCatalog& catalog) {
    catalog.add(make_minicpm5_architecture());
}

} // namespace celeg::detail

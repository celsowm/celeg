#include "lfm/config.hpp"
#include "lfm/json.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace lfm {
namespace {

int read_int(const Json& root, const char* name) {
    const int64_t value = root[name].as_i64();
    if (value < 0 || value > 2'000'000'000LL) {
        throw std::runtime_error(std::string("invalid config integer: ") + name);
    }
    return static_cast<int>(value);
}

int read_int_or(const Json& root, const char* primary, const char* fallback) {
    if (root.contains(primary)) return read_int(root, primary);
    if (root.contains(fallback)) return read_int(root, fallback);
    throw std::runtime_error(std::string("missing config integer: ") + primary);
}

bool read_bool(const Json& root, const char* name) {
    return root[name].as_bool();
}

bool read_bool_or(const Json& root, const char* primary, const char* fallback) {
    if (root.contains(primary)) return root[primary].as_bool();
    if (root.contains(fallback)) return root[fallback].as_bool();
    return false;
}

} // namespace

ModelConfig ModelConfig::load(const std::string& path) {
    const Json root = Json::parse_file(path);
    ModelConfig config;
    config.model_type = root["model_type"].as_string();
    if (config.model_type == "lfm2") {
        config.architecture = ArchitectureKind::DenseLfm2;
    } else if (config.model_type == "lfm2_moe") {
        config.architecture = ArchitectureKind::MoeLfm2;
    } else {
        throw std::runtime_error("unsupported model architecture: " + config.model_type);
    }
    config.dtype = root["dtype"].as_string();
    config.hidden_size = read_int(root, "hidden_size");
    config.intermediate_size = read_int(root, "intermediate_size");
    config.num_hidden_layers = read_int(root, "num_hidden_layers");
    // LiquidAI publishes "num_attention_heads" but some checkpoints ship the
    // alias "num_heads". Accept either to stay forward-compatible.
    config.num_attention_heads = read_int_or(root, "num_attention_heads", "num_heads");
    config.num_key_value_heads = read_int(root, "num_key_value_heads");
    config.vocab_size = read_int(root, "vocab_size");
    config.conv_cache = read_int(root, "conv_L_cache");
    // MoE configurations omit conv_dim and use_pos_enc; the official
    // architecture always uses conv_dim == hidden_size and positional
    // encoding. Default those only when absent (validated against the
    // official Transformers implementation).
    config.conv_dim = root.contains("conv_dim")
        ? read_int(root, "conv_dim")
        : config.hidden_size;
    config.max_position_embeddings = read_int(root, "max_position_embeddings");
    config.bos_token_id = read_int(root, "bos_token_id");
    config.eos_token_id = read_int(root, "eos_token_id");
    config.pad_token_id = read_int(root, "pad_token_id");
    config.norm_eps = static_cast<float>(root["norm_eps"].as_number());
    config.conv_bias = read_bool(root, "conv_bias");
    // 1.2B-Instruct uses "tie_embedding"; 230M uses "tie_word_embeddings".
    config.tie_word_embeddings = read_bool_or(root, "tie_word_embeddings", "tie_embedding");
    config.use_pos_enc = root.contains("use_pos_enc")
        ? read_bool(root, "use_pos_enc")
        : true;
    // rope_theta / rope_type may be top-level (1.2B-Instruct) or nested under
    // "rope_parameters" (230M). Accept either layout.
    if (root.contains("rope_parameters") && !root.contains("rope_theta")) {
        config.rope_theta = static_cast<float>(root["rope_parameters"]["rope_theta"].as_number());
        if (root["rope_parameters"].contains("rope_type")) {
            config.rope_type = root["rope_parameters"]["rope_type"].as_string();
        } else {
            config.rope_type = "default";
        }
    } else {
        config.rope_theta = static_cast<float>(root["rope_theta"].as_number());
        if (root.contains("rope_type")) {
            config.rope_type = root["rope_type"].as_string();
        } else {
            config.rope_type = "default";
        }
    }

    if (root.contains("_name")) {
        config.repo_hint = root["_name"].as_string();
    }

    if (root.contains("head_dim")) {
        config.head_dim = read_int(root, "head_dim");
    } else {
        if (config.num_attention_heads == 0 || config.hidden_size % config.num_attention_heads != 0) {
            throw std::runtime_error("hidden_size must be divisible by num_attention_heads");
        }
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }

    if (config.architecture == ArchitectureKind::MoeLfm2) {
        MoeConfig moe;
        moe.intermediate_size = config.intermediate_size;
        moe.moe_intermediate_size = read_int(root, "moe_intermediate_size");
        moe.num_dense_layers = read_int(root, "num_dense_layers");
        moe.num_experts = read_int(root, "num_experts");
        moe.experts_per_token = read_int(root, "num_experts_per_tok");
        moe.normalize_topk = read_bool(root, "norm_topk_prob");
        moe.use_expert_bias = read_bool(root, "use_expert_bias");
        moe.routed_scaling_factor = root.contains("routed_scaling_factor")
            ? static_cast<float>(root["routed_scaling_factor"].as_number())
            : 1.0f;
        config.moe = moe;
    }

    for (const Json& item : root["layer_types"].as_array()) {
        const std::string& value = item.as_string();
        if (value == "conv") {
            config.layer_types.push_back(LayerType::Convolution);
        } else if (value == "full_attention") {
            config.layer_types.push_back(LayerType::FullAttention);
        } else {
            throw std::runtime_error("unsupported layer type in config: " + value);
        }
    }

    config.validate();
    return config;
}

void ModelConfig::validate() const {
    if (architecture == ArchitectureKind::DenseLfm2) {
        if (model_type != "lfm2") throw std::runtime_error("config model_type is not lfm2");
    } else if (architecture == ArchitectureKind::MoeLfm2) {
        if (model_type != "lfm2_moe") {
            throw std::runtime_error("config model_type is not lfm2_moe");
        }
        const MoeConfig& m = *moe;
        if (m.intermediate_size <= 0) {
            throw std::runtime_error("invalid MoE dense intermediate size");
        }
        if (m.num_experts <= 0) throw std::runtime_error("invalid MoE expert count");
        if (m.experts_per_token <= 0) {
            throw std::runtime_error("invalid MoE experts_per_token");
        }
        if (m.experts_per_token > m.num_experts) {
            throw std::runtime_error("experts_per_token exceeds num_experts");
        }
        if (m.num_dense_layers < 0) {
            throw std::runtime_error("invalid MoE dense layer count");
        }
        if (m.num_dense_layers > num_hidden_layers) {
            throw std::runtime_error("MoE dense layer count exceeds total layers");
        }
        if (!(m.routed_scaling_factor > 0.0f) || !std::isfinite(m.routed_scaling_factor)) {
            throw std::runtime_error("invalid MoE routed_scaling_factor");
        }
    } else {
        throw std::runtime_error("unsupported model architecture");
    }
    if (!is_gguf && dtype != "bfloat16") {
        throw std::runtime_error("only bfloat16 checkpoints are supported");
    }
    if (hidden_size <= 0 || intermediate_size <= 0 || vocab_size <= 0) {
        throw std::runtime_error("invalid non-positive model dimensions");
    }
    if (num_hidden_layers <= 0 || static_cast<int>(layer_types.size()) != num_hidden_layers) {
        throw std::runtime_error("layer_types length does not match num_hidden_layers");
    }
    if (num_attention_heads <= 0 || num_key_value_heads <= 0 ||
        num_attention_heads % num_key_value_heads != 0) {
        throw std::runtime_error("invalid GQA head configuration");
    }
    if (head_dim <= 0 || head_dim * num_attention_heads != hidden_size || (head_dim % 2) != 0) {
        throw std::runtime_error("invalid attention head_dim");
    }
    if (conv_dim != hidden_size || conv_cache <= 0) {
        throw std::runtime_error("unsupported convolution dimensions");
    }
    if (max_position_embeddings <= 0) throw std::runtime_error("invalid max_position_embeddings");
    if (!(norm_eps > 0.0f) || !std::isfinite(norm_eps)) throw std::runtime_error("invalid norm_eps");
    if (!(rope_theta > 0.0f) || !std::isfinite(rope_theta)) throw std::runtime_error("invalid rope_theta");
    if (rope_type != "default") throw std::runtime_error("only default RoPE is supported");
    if (conv_bias) throw std::runtime_error("convolution bias is not implemented");
    if (!use_pos_enc) throw std::runtime_error("checkpoint disables positional encoding");
    if (bos_token_id < 0 || eos_token_id < 0 || pad_token_id < 0 ||
        bos_token_id >= vocab_size || eos_token_id >= vocab_size || pad_token_id >= vocab_size) {
        throw std::runtime_error("invalid special token IDs in config");
    }
}

std::string ModelConfig::summary() const {
    int attention_layers = 0;
    for (LayerType type : layer_types) {
        if (type == LayerType::FullAttention) ++attention_layers;
    }
    std::ostringstream out;
    out << "model_type=" << model_type
        << " architecture=" << (architecture == ArchitectureKind::MoeLfm2 ? "lfm2_moe" : "lfm2")
        << " dtype=" << dtype
        << " hidden=" << hidden_size;
    if (architecture == ArchitectureKind::MoeLfm2 && moe) {
        out << " moe_intermediate=" << moe->intermediate_size
            << " num_dense_layers=" << moe->num_dense_layers
            << " num_experts=" << moe->num_experts
            << " experts_per_token=" << moe->experts_per_token
            << " normalize_topk=" << (moe->normalize_topk ? 1 : 0)
            << " use_expert_bias=" << (moe->use_expert_bias ? 1 : 0)
            << " routed_scaling_factor=" << moe->routed_scaling_factor;
    }
    out << " intermediate=" << intermediate_size
        << " layers=" << num_hidden_layers
        << " attention_layers=" << attention_layers
        << " conv_layers=" << (num_hidden_layers - attention_layers)
        << " q_heads=" << num_attention_heads
        << " kv_heads=" << num_key_value_heads
        << " head_dim=" << head_dim
        << " vocab=" << vocab_size
        << " max_positions=" << max_position_embeddings
        << " rope_theta=" << rope_theta;
    return out.str();
}

} // namespace lfm

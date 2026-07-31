#include "celeg/model/config/config.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/repositories/gguf.hpp"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {

// Derives a ModelConfig from an LFM2 GGUF checkpoint's metadata. GGUF stores the
// architecture hyper-parameters under the `lfm2.*` and `general.*` key spaces,
// and encodes the per-layer conv/attention pattern in the
// `lfm2.attention.head_count_kv` array (0 -> convolution, non-zero -> attention).
ModelConfig ModelConfig::from_gguf(const GgufFile& gguf) {
    const std::string arch = gguf.str("general.architecture");
    if (arch != "lfm2" && arch != "lfm2moe") {
        throw std::runtime_error("unsupported GGUF architecture: " + arch);
    }

    ModelConfig config;
    config.is_gguf = true;
    config.dtype = "gguf";

    const bool is_moe = (arch == "lfm2moe");
    const std::string prefix = is_moe ? "lfm2moe" : "lfm2";

    if (is_moe) {
        config.architecture = ArchitectureKind::MoeLfm2;
        config.model_type = "lfm2_moe";
    } else {
        config.architecture = ArchitectureKind::DenseLfm2;
        config.model_type = "lfm2";
    }

    config.hidden_size = static_cast<int>(gguf.u32(prefix + ".embedding_length"));
    config.intermediate_size = static_cast<int>(gguf.u32(prefix + ".feed_forward_length"));
    config.num_hidden_layers = static_cast<int>(gguf.u32(prefix + ".block_count"));
    config.num_attention_heads = static_cast<int>(gguf.u32(prefix + ".attention.head_count"));
    config.vocab_size = static_cast<int>(
        gguf.has(prefix + ".vocab_size") ? gguf.u32(prefix + ".vocab_size")
                                         : gguf.u32("llama.vocab_size"));
    config.conv_cache = static_cast<int>(gguf.u32(prefix + ".shortconv.l_cache"));
    config.conv_dim = config.hidden_size;
    config.max_position_embeddings =
        static_cast<int>(gguf.u32(prefix + ".context_length"));
    config.norm_eps = gguf.f32(prefix + ".attention.layer_norm_rms_epsilon");
    config.rope_theta = gguf.f32(prefix + ".rope.freq_base");
    config.rope_type = "default";
    config.conv_bias = false;
    config.use_pos_enc = true;
    config.tie_word_embeddings = true;  // LFM2 ties the LM head to the embedding.
    config.repo_hint = gguf.str_or("general.name", "");

    // Per-layer head_count_kv: 0 => convolution layer, else full attention.
    const GgufValue& kv_heads = gguf.value(prefix + ".attention.head_count_kv");
    if (kv_heads.kind != GgufValueKind::Array) {
        throw std::runtime_error(prefix + ".attention.head_count_kv is not an array");
    }
    const std::vector<int64_t>& per_layer = kv_heads.array_integers;
    if (static_cast<int>(per_layer.size()) != config.num_hidden_layers) {
        throw std::runtime_error(
            prefix + ".attention.head_count_kv length does not match block_count");
    }
    int kv_heads_value = 0;
    config.layer_types.reserve(per_layer.size());
    for (int64_t v : per_layer) {
        if (v == 0) {
            config.layer_types.push_back(LayerType::Convolution);
        } else {
            config.layer_types.push_back(LayerType::FullAttention);
            if (kv_heads_value == 0) kv_heads_value = static_cast<int>(v);
            else if (kv_heads_value != static_cast<int>(v)) {
                throw std::runtime_error(
                    "heterogeneous KV head counts are not supported");
            }
        }
    }
    if (kv_heads_value == 0) {
        throw std::runtime_error("GGUF checkpoint has no attention layers");
    }
    config.num_key_value_heads = kv_heads_value;

    // head_dim: prefer explicit key, else derive from hidden / heads.
    if (gguf.has(prefix + ".attention.key_length")) {
        config.head_dim = static_cast<int>(gguf.u32(prefix + ".attention.key_length"));
    } else {
        if (config.num_attention_heads == 0 ||
            config.hidden_size % config.num_attention_heads != 0) {
            throw std::runtime_error(
                "hidden_size not divisible by num_attention_heads");
        }
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }

    // GGUF omits explicit special-token config for the model; take the tokenizer
    // ids so validate()'s bounds checks pass. These mirror LFM2.5 defaults.
    config.bos_token_id = gguf.has("tokenizer.ggml.bos_token_id")
        ? static_cast<int>(gguf.i64("tokenizer.ggml.bos_token_id")) : 1;
    config.eos_token_id = gguf.has("tokenizer.ggml.eos_token_id")
        ? static_cast<int>(gguf.i64("tokenizer.ggml.eos_token_id")) : 7;
    config.pad_token_id = gguf.has("tokenizer.ggml.padding_token_id")
        ? static_cast<int>(gguf.i64("tokenizer.ggml.padding_token_id")) : 0;

    // MoE configuration (only present in lfm2moe GGUF checkpoints).
    if (is_moe) {
        MoeConfig moe_cfg;
        if (gguf.has(prefix + ".expert_count")) {
            moe_cfg.num_experts = static_cast<int>(gguf.u32(prefix + ".expert_count"));
        }
        if (gguf.has(prefix + ".expert_used_count")) {
            moe_cfg.experts_per_token = static_cast<int>(gguf.u32(prefix + ".expert_used_count"));
        }
        if (gguf.has(prefix + ".expert_feed_forward_length")) {
            moe_cfg.moe_intermediate_size = static_cast<int>(gguf.u32(prefix + ".expert_feed_forward_length"));
        }
        if (gguf.has(prefix + ".num_dense_layers")) {
            moe_cfg.num_dense_layers = static_cast<int>(gguf.u32(prefix + ".num_dense_layers"));
        } else {
            // GGUF exports omit this configuration key. Infer the dense prefix
            // from its un-routed FFN tensors instead of treating every block as
            // an MoE block. LFM2.5-8B-A1B has two such prefix blocks.
            while (gguf.contains_tensor("blk." +
                    std::to_string(moe_cfg.num_dense_layers) +
                    ".ffn_gate.weight")) {
                ++moe_cfg.num_dense_layers;
            }
        }
        const std::string first_moe = "blk." +
            std::to_string(moe_cfg.num_dense_layers) + ".ffn_expert_bias.weight";
        moe_cfg.use_expert_bias = gguf.contains_tensor(first_moe);
        moe_cfg.normalize_topk = true;
        moe_cfg.routed_scaling_factor = 1.0f;
        moe_cfg.intermediate_size = config.intermediate_size;
        config.moe = moe_cfg;
        config.validate();
        return config;
    }

    config.validate();
    return config;
}

// ---------------------------------------------------------------------------
// GgufRepository.
// ---------------------------------------------------------------------------

namespace {

// Extracts the "{i}" from "model.layers.{i}." and the trailing suffix.
// Returns false if `name` is not a per-layer tensor.
bool split_layer(std::string_view name, int& layer, std::string& suffix) {
    constexpr std::string_view prefix = "model.layers.";
    if (name.substr(0, prefix.size()) != prefix) return false;
    size_t pos = prefix.size();
    size_t start = pos;
    while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') ++pos;
    if (pos == start || pos >= name.size() || name[pos] != '.') return false;
    layer = std::stoi(std::string(name.substr(start, pos - start)));
    suffix = std::string(name.substr(pos + 1));
    return true;
}

std::string blk(int layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

struct ExpertTensorRequest {
    int layer = 0;
    int expert = 0;
    std::string gguf_name;
};

std::optional<ExpertTensorRequest> parse_expert_tensor(std::string_view hf_name) {
    int layer = 0;
    std::string suffix;
    if (!split_layer(hf_name, layer, suffix)) return std::nullopt;
    constexpr std::string_view prefix = "feed_forward.experts.";
    if (!suffix.starts_with(prefix)) return std::nullopt;
    const std::string rest = suffix.substr(prefix.size());
    const size_t dot = rest.find('.');
    if (dot == std::string::npos || dot == 0) return std::nullopt;
    for (size_t i = 0; i < dot; ++i) {
        if (rest[i] < '0' || rest[i] > '9') return std::nullopt;
    }
    const int expert = std::stoi(rest.substr(0, dot));
    const std::string_view weight = std::string_view(rest).substr(dot + 1);
    const char* name = nullptr;
    if (weight == "w1.weight") name = "ffn_gate_exps.weight";
    if (weight == "w3.weight") name = "ffn_up_exps.weight";
    if (weight == "w2.weight") name = "ffn_down_exps.weight";
    if (name == nullptr) return std::nullopt;
    return ExpertTensorRequest{layer, expert, blk(layer, name)};
}

HostTensorView host_view(const GgufTensorView& gt) {
    HostTensorView view;
    view.shape = gt.shape;
    view.data = gt.data;
    view.bytes = gt.bytes;
    view.ggml_type = gt.type;
    switch (gt.type) {
        case GgmlType::F32: view.dtype = TensorDType::F32; break;
        case GgmlType::F16: view.dtype = TensorDType::F16; break;
        case GgmlType::BF16: view.dtype = TensorDType::BF16; break;
        default: view.dtype = TensorDType::Quantized; break;
    }
    return view;
}

} // namespace

GgufRepository::GgufRepository(std::shared_ptr<GgufFile> gguf)
    : gguf_(std::move(gguf)) {
    if (!gguf_) throw std::invalid_argument("GgufRepository requires a GgufFile");
}

std::string GgufRepository::translate(std::string_view hf_name) const {
    if (hf_name == "model.embed_tokens.weight") return "token_embd.weight";
    if (hf_name == "model.embedding_norm.weight") return "token_embd_norm.weight";
    if (hf_name == "model.lm_head.weight") return "output.weight";

    int layer = 0;
    std::string s;
    if (!split_layer(hf_name, layer, s)) return {};

    if (s == "operator_norm.weight") return blk(layer, "attn_norm.weight");
    if (s == "ffn_norm.weight") return blk(layer, "ffn_norm.weight");

    // Dense feed-forward.
    if (s == "feed_forward.w1.weight") return blk(layer, "ffn_gate.weight");
    if (s == "feed_forward.w3.weight") return blk(layer, "ffn_up.weight");
    if (s == "feed_forward.w2.weight") return blk(layer, "ffn_down.weight");

    // Attention.
    if (s == "self_attn.q_proj.weight") return blk(layer, "attn_q.weight");
    if (s == "self_attn.k_proj.weight") return blk(layer, "attn_k.weight");
    if (s == "self_attn.v_proj.weight") return blk(layer, "attn_v.weight");
    if (s == "self_attn.out_proj.weight") return blk(layer, "attn_output.weight");
    if (s == "self_attn.q_layernorm.weight") return blk(layer, "attn_q_norm.weight");
    if (s == "self_attn.k_layernorm.weight") return blk(layer, "attn_k_norm.weight");

    // Short convolution.
    if (s == "conv.in_proj.weight") return blk(layer, "shortconv.in_proj.weight");
    if (s == "conv.conv.weight") return blk(layer, "shortconv.conv.weight");
    if (s == "conv.out_proj.weight") return blk(layer, "shortconv.out_proj.weight");

    // MoE feed-forward (expert weights + router).
    if (s == "feed_forward.gate.weight") return blk(layer, "ffn_gate_inp.weight");
    if (s.substr(0, strlen("feed_forward.experts.")) == "feed_forward.experts.") {
        // Parse "feed_forward.experts.{e}.w{1,2,3}.weight"
        std::string rest = s.substr(strlen("feed_forward.experts."));
        size_t dot = rest.find('.');
        if (dot != std::string::npos) {
            std::string expert_id = rest.substr(0, dot);
            std::string weight_suffix = rest.substr(dot + 1);
            if (weight_suffix == "w1.weight")
                return blk(layer, ("ffn_gate_exps." + expert_id + ".weight").c_str());
            if (weight_suffix == "w3.weight")
                return blk(layer, ("ffn_up_exps." + expert_id + ".weight").c_str());
            if (weight_suffix == "w2.weight")
                return blk(layer, ("ffn_down_exps." + expert_id + ".weight").c_str());
        }
    }
    if (s == "feed_forward.expert_bias.weight")
        return blk(layer, "ffn_expert_bias.weight");

    return {};
}

bool GgufRepository::contains(std::string_view name) const {
    if (const auto expert = parse_expert_tensor(name)) {
        return gguf_->contains_tensor(expert->gguf_name);
    }
    const std::string gguf_name = translate(name);
    return !gguf_name.empty() && gguf_->contains_tensor(gguf_name);
}

HostTensorView GgufRepository::tensor(std::string_view name) const {
    if (const auto expert = parse_expert_tensor(name)) {
        const GgufTensorView packed = gguf_->tensor(expert->gguf_name);
        if (packed.shape.size() != 3 || packed.shape[0] <= 0 ||
            expert->expert < 0 || expert->expert >= packed.shape[0] ||
            packed.bytes % static_cast<size_t>(packed.shape[0]) != 0) {
            throw std::runtime_error("invalid packed GGUF expert tensor: " +
                                     expert->gguf_name);
        }
        HostTensorView view = host_view(packed);
        const size_t expert_bytes = packed.bytes /
            static_cast<size_t>(packed.shape[0]);
        view.shape.erase(view.shape.begin());
        view.data += static_cast<size_t>(expert->expert) * expert_bytes;
        view.bytes = expert_bytes;
        return view;
    }
    const std::string gguf_name = translate(name);
    if (gguf_name.empty()) {
        throw std::out_of_range("gguf: no mapping for tensor " + std::string(name));
    }
    const GgufTensorView gt = gguf_->tensor(gguf_name);

    return host_view(gt);
}

std::vector<std::string> GgufRepository::names() const {
    return gguf_->tensor_names();
}

} // namespace celeg

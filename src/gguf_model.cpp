#include "lfm/config.hpp"
#include "lfm/gguf.hpp"
#include "lfm/gguf_repo.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm {

// Derives a ModelConfig from an LFM2 GGUF checkpoint's metadata. GGUF stores the
// architecture hyper-parameters under the `lfm2.*` and `general.*` key spaces,
// and encodes the per-layer conv/attention pattern in the
// `lfm2.attention.head_count_kv` array (0 -> convolution, non-zero -> attention).
ModelConfig ModelConfig::from_gguf(const GgufFile& gguf) {
    const std::string arch = gguf.str("general.architecture");
    if (arch != "lfm2") {
        throw std::runtime_error("unsupported GGUF architecture: " + arch);
    }

    ModelConfig config;
    config.is_gguf = true;
    config.model_type = "lfm2";
    config.architecture = ArchitectureKind::DenseLfm2;
    config.dtype = "gguf";

    config.hidden_size = static_cast<int>(gguf.u32("lfm2.embedding_length"));
    config.intermediate_size = static_cast<int>(gguf.u32("lfm2.feed_forward_length"));
    config.num_hidden_layers = static_cast<int>(gguf.u32("lfm2.block_count"));
    config.num_attention_heads = static_cast<int>(gguf.u32("lfm2.attention.head_count"));
    config.vocab_size = static_cast<int>(
        gguf.has("lfm2.vocab_size") ? gguf.u32("lfm2.vocab_size")
                                    : gguf.u32("llama.vocab_size"));
    config.conv_cache = static_cast<int>(gguf.u32("lfm2.shortconv.l_cache"));
    config.conv_dim = config.hidden_size;
    config.max_position_embeddings =
        static_cast<int>(gguf.u32("lfm2.context_length"));
    config.norm_eps = gguf.f32("lfm2.attention.layer_norm_rms_epsilon");
    config.rope_theta = gguf.f32("lfm2.rope.freq_base");
    config.rope_type = "default";
    config.conv_bias = false;
    config.use_pos_enc = true;
    config.tie_word_embeddings = true;  // LFM2 ties the LM head to the embedding.
    config.repo_hint = gguf.str_or("general.name", "");

    // Per-layer head_count_kv: 0 => convolution layer, else full attention.
    const GgufValue& kv_heads = gguf.value("lfm2.attention.head_count_kv");
    if (kv_heads.kind != GgufValueKind::Array) {
        throw std::runtime_error("lfm2.attention.head_count_kv is not an array");
    }
    const std::vector<int64_t>& per_layer = kv_heads.array_integers;
    if (static_cast<int>(per_layer.size()) != config.num_hidden_layers) {
        throw std::runtime_error(
            "lfm2.attention.head_count_kv length does not match block_count");
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
    if (gguf.has("lfm2.attention.key_length")) {
        config.head_dim = static_cast<int>(gguf.u32("lfm2.attention.key_length"));
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

    return {};
}

bool GgufRepository::contains(std::string_view name) const {
    const std::string gguf_name = translate(name);
    return !gguf_name.empty() && gguf_->contains_tensor(gguf_name);
}

HostTensorView GgufRepository::tensor(std::string_view name) const {
    const std::string gguf_name = translate(name);
    if (gguf_name.empty()) {
        throw std::out_of_range("gguf: no mapping for tensor " + std::string(name));
    }
    const GgufTensorView gt = gguf_->tensor(gguf_name);

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

std::vector<std::string> GgufRepository::names() const {
    return gguf_->tensor_names();
}

} // namespace lfm

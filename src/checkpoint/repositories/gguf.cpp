#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/repositories/gguf.hpp"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {

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
    view.block_encoding = block_encoding_from_ggml_type(gt.type);
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
    if (hf_name == "model.norm.weight") return "output_norm.weight";
    if (hf_name == "model.lm_head.weight") return "output.weight";

    int layer = 0;
    std::string s;
    if (!split_layer(hf_name, layer, s)) return {};

    if (s == "operator_norm.weight") return blk(layer, "attn_norm.weight");
    if (s == "input_layernorm.weight") return blk(layer, "attn_norm.weight");
    if (s == "ffn_norm.weight") return blk(layer, "ffn_norm.weight");
    if (s == "post_attention_layernorm.weight") return blk(layer, "ffn_norm.weight");

    // Dense feed-forward.
    if (s == "feed_forward.w1.weight") return blk(layer, "ffn_gate.weight");
    if (s == "feed_forward.w3.weight") return blk(layer, "ffn_up.weight");
    if (s == "feed_forward.w2.weight") return blk(layer, "ffn_down.weight");

    // Attention.
    if (s == "self_attn.q_proj.weight") return blk(layer, "attn_q.weight");
    if (s == "self_attn.k_proj.weight") return blk(layer, "attn_k.weight");
    if (s == "self_attn.v_proj.weight") return blk(layer, "attn_v.weight");
    if (s == "self_attn.out_proj.weight") return blk(layer, "attn_output.weight");
    if (s == "self_attn.o_proj.weight") return blk(layer, "attn_output.weight");
    if (s == "self_attn.q_layernorm.weight") return blk(layer, "attn_q_norm.weight");
    if (s == "self_attn.k_layernorm.weight") return blk(layer, "attn_k_norm.weight");

    // Granite names its MLP projections gate/up/down; GGUF stores the same
    // semantic roles using the common ffn_* block names.
    if (s == "mlp.gate_proj.weight") return blk(layer, "ffn_gate.weight");
    if (s == "mlp.up_proj.weight") return blk(layer, "ffn_up.weight");
    if (s == "mlp.down_proj.weight") return blk(layer, "ffn_down.weight");

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

#include "gguf_tensor_mapper.hpp"

#include <optional>

namespace celeg {
namespace {

bool split_layer(std::string_view name, int& layer, std::string& suffix) {
    std::string_view prefix;
    if (name.starts_with("model.layers.")) prefix = "model.layers.";
    else if (name.starts_with("backbone.layers.")) prefix = "backbone.layers.";
    else if (name.starts_with("model.backbone.layers.")) {
        prefix = "model.backbone.layers.";
    } else {
        return false;
    }

    size_t position = prefix.size();
    const size_t start = position;
    while (position < name.size() && name[position] >= '0' && name[position] <= '9') {
        ++position;
    }
    if (position == start || position >= name.size() || name[position] != '.') {
        return false;
    }
    layer = std::stoi(std::string(name.substr(start, position - start)));
    suffix = std::string(name.substr(position + 1));
    return true;
}

std::string block_name(int layer, std::string_view suffix) {
    return "blk." + std::to_string(layer) + "." + std::string(suffix);
}

std::optional<GgufTensorReference> resolve_expert(std::string_view canonical_name) {
    int layer = 0;
    std::string suffix;
    if (!split_layer(canonical_name, layer, suffix)) return std::nullopt;

    constexpr std::string_view prefix = "feed_forward.experts.";
    if (!suffix.starts_with(prefix)) return std::nullopt;
    const std::string rest = suffix.substr(prefix.size());
    const size_t dot = rest.find('.');
    if (dot == std::string::npos || dot == 0) return std::nullopt;
    for (size_t i = 0; i < dot; ++i) {
        if (rest[i] < '0' || rest[i] > '9') return std::nullopt;
    }

    std::string_view weight = std::string_view(rest).substr(dot + 1);
    std::string_view native_suffix;
    if (weight == "w1.weight") native_suffix = "ffn_gate_exps.weight";
    else if (weight == "w3.weight") native_suffix = "ffn_up_exps.weight";
    else if (weight == "w2.weight") native_suffix = "ffn_down_exps.weight";
    else return std::nullopt;

    return GgufTensorReference{
        block_name(layer, native_suffix),
        std::stoi(rest.substr(0, dot))};
}

GgufTensorReference layer_tensor(int layer, std::string_view suffix) {
    return GgufTensorReference{block_name(layer, suffix), -1};
}

}

GgufTensorReference GgufTensorNameMapper::resolve(std::string_view canonical_name) {
    constexpr std::string_view language_model_prefix = "model.language_model.";
    std::string normalized;
    if (canonical_name.starts_with(language_model_prefix)) {
        normalized = "model." + std::string(canonical_name.substr(language_model_prefix.size()));
        canonical_name = normalized;
    }

    if (canonical_name == "backbone.embeddings.weight" ||
        canonical_name == "model.backbone.embeddings.weight" ||
        canonical_name == "model.embed_tokens.weight") {
        return {"token_embd.weight", -1};
    }
    if (canonical_name == "model.embedding_norm.weight") {
        return {"token_embd_norm.weight", -1};
    }
    if (canonical_name == "backbone.norm_f.weight" ||
        canonical_name == "backbone.norm.weight" ||
        canonical_name == "model.backbone.norm_f.weight" ||
        canonical_name == "model.backbone.norm.weight" ||
        canonical_name == "model.norm.weight") {
        return {"output_norm.weight", -1};
    }
    if (canonical_name == "lm_head.weight" ||
        canonical_name == "model.lm_head.weight") {
        return {"output.weight", -1};
    }

    if (const auto expert = resolve_expert(canonical_name)) return *expert;

    int layer = 0;
    std::string suffix;
    if (!split_layer(canonical_name, layer, suffix)) return {};

    if (suffix == "operator_norm.weight" || suffix == "input_layernorm.weight" ||
        suffix == "norm.weight") return layer_tensor(layer, "attn_norm.weight");
    if (suffix == "ffn_norm.weight" || suffix == "post_attention_layernorm.weight") {
        return layer_tensor(layer, "ffn_norm.weight");
    }

    if (suffix == "mixer.in_proj.weight") return layer_tensor(layer, "ssm_in.weight");
    if (suffix == "mixer.conv1d.weight") return layer_tensor(layer, "ssm_conv1d.weight");
    if (suffix == "mixer.conv1d.bias") return layer_tensor(layer, "ssm_conv1d.bias");
    if (suffix == "mixer.dt_bias") return layer_tensor(layer, "ssm_dt.bias");
    if (suffix == "mixer.A_log") return layer_tensor(layer, "ssm_a");
    if (suffix == "mixer.D") return layer_tensor(layer, "ssm_d");
    if (suffix == "mixer.norm.weight") return layer_tensor(layer, "ssm_norm.weight");
    if (suffix == "mixer.out_proj.weight") return layer_tensor(layer, "ssm_out.weight");
    if (suffix == "mixer.q_proj.weight") return layer_tensor(layer, "attn_q.weight");
    if (suffix == "mixer.k_proj.weight") return layer_tensor(layer, "attn_k.weight");
    if (suffix == "mixer.v_proj.weight") return layer_tensor(layer, "attn_v.weight");
    if (suffix == "mixer.o_proj.weight") return layer_tensor(layer, "attn_output.weight");
    if (suffix == "mixer.up_proj.weight") return layer_tensor(layer, "ffn_up.weight");
    if (suffix == "mixer.down_proj.weight") return layer_tensor(layer, "ffn_down.weight");

    if (suffix == "feed_forward.w1.weight" || suffix == "mlp.gate_proj.weight") {
        return layer_tensor(layer, "ffn_gate.weight");
    }
    if (suffix == "feed_forward.w3.weight" || suffix == "mlp.up_proj.weight") {
        return layer_tensor(layer, "ffn_up.weight");
    }
    if (suffix == "feed_forward.w2.weight" || suffix == "mlp.down_proj.weight") {
        return layer_tensor(layer, "ffn_down.weight");
    }

    if (suffix == "self_attn.q_proj.weight") return layer_tensor(layer, "attn_q.weight");
    if (suffix == "self_attn.k_proj.weight") return layer_tensor(layer, "attn_k.weight");
    if (suffix == "self_attn.v_proj.weight") return layer_tensor(layer, "attn_v.weight");
    if (suffix == "self_attn.out_proj.weight" || suffix == "self_attn.o_proj.weight") {
        return layer_tensor(layer, "attn_output.weight");
    }
    if (suffix == "self_attn.q_layernorm.weight") {
        return layer_tensor(layer, "attn_q_norm.weight");
    }
    if (suffix == "self_attn.k_layernorm.weight") {
        return layer_tensor(layer, "attn_k_norm.weight");
    }

    if (suffix == "conv.in_proj.weight") return layer_tensor(layer, "shortconv.in_proj.weight");
    if (suffix == "conv.conv.weight") return layer_tensor(layer, "shortconv.conv.weight");
    if (suffix == "conv.out_proj.weight") return layer_tensor(layer, "shortconv.out_proj.weight");
    if (suffix == "feed_forward.gate.weight") {
        return layer_tensor(layer, "ffn_gate_inp.weight");
    }
    if (suffix == "feed_forward.expert_bias.weight") {
        return layer_tensor(layer, "ffn_expert_bias.weight");
    }

    return {};
}

}

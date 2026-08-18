#include "support.hpp"

#include "canonical_internal.hpp"

#include <stdexcept>
#include <stdexcept>
#include <stdexcept>
#include <string>

namespace celeg::inference_detail {

[[noreturn]] void fail(ResolutionFailureKind kind, std::string message,
                       std::vector<EvidenceItem> evidence) {
    throw ResolutionError(kind, std::move(message), std::move(evidence));
}

bool shape_is(const TensorInventoryEntry& entry,
              std::initializer_list<std::int64_t> expected) {
    const std::vector<std::int64_t> wanted(expected);
    if (entry.shape == wanted) return true;

    return entry.shape.size() == 2 && wanted.size() == 3 && wanted[1] == 1 &&
           entry.shape[0] == wanted[0] && entry.shape[1] == wanted[2];
}

const TensorInventoryEntry* find_unique(const TensorInventory& inventory,
                                        const std::vector<std::string>& candidates,
                                        TensorRole role, int layer,
                                        std::initializer_list<std::int64_t> shape,
                                        std::vector<EvidenceItem> evidence) {
    std::vector<const TensorInventoryEntry*> matches;
    for (const std::string& candidate : candidates) {
        if (const auto* entry = inventory.find(candidate)) matches.push_back(entry);
    }
    if (matches.empty()) {
        fail(ResolutionFailureKind::MissingTensorRole,
             "automatic resolution could not bind " +
                 std::string(tensor_role_name(role)) +
                 (layer >= 0 ? " for layer " + std::to_string(layer) : ""),
             std::move(evidence));
    }
    if (matches.size() != 1) {
        std::string message = "automatic resolution found multiple bindings for " +
            std::string(tensor_role_name(role));
        if (layer >= 0) message += " for layer " + std::to_string(layer);
        for (const auto* entry : matches) message += "\n  " + entry->name;
        fail(ResolutionFailureKind::AmbiguousTensorBinding, std::move(message),
             std::move(evidence));
    }
    if (!shape_is(*matches.front(), shape)) {
        fail(ResolutionFailureKind::ShapeConstraintViolation,
             "tensor " + matches.front()->name + " has a shape inconsistent with " +
                 std::string(tensor_role_name(role)));
    }
    evidence.push_back({EvidenceKind::TensorName, matches.front()->name,
                        std::string(tensor_role_name(role))});
    return matches.front();
}

std::vector<std::string> attention_tensor_candidates(int layer,
                                                      std::string_view suffix) {
    const std::string index = std::to_string(layer);
    std::string gguf_suffix;
    if (suffix == "q_proj.weight") gguf_suffix = "attn_q.weight";
    if (suffix == "k_proj.weight") gguf_suffix = "attn_k.weight";
    if (suffix == "v_proj.weight") gguf_suffix = "attn_v.weight";
    if (suffix == "o_proj.weight") gguf_suffix = "attn_output.weight";
    if (suffix == "q_norm.weight") gguf_suffix = "attn_q_norm.weight";
    if (suffix == "k_norm.weight") gguf_suffix = "attn_k_norm.weight";
    if (suffix == "q_norm.weight") gguf_suffix = "attn_q_norm.weight";
    if (suffix == "k_norm.weight") gguf_suffix = "attn_k_norm.weight";
    if (suffix == "q_norm.weight") gguf_suffix = "attn_q_norm.weight";
    if (suffix == "k_norm.weight") gguf_suffix = "attn_k_norm.weight";
    std::vector<std::string> result = {
        "transformer.h." + index + ".attn." + std::string(suffix),
        "model.language_model.layers." + index + ".self_attn." + std::string(suffix),
        "model.layers." + index + ".self_attn." + std::string(suffix),
        "model.layers." + index + ".attention." + std::string(suffix),
        "layers." + index + ".attention." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
    if (suffix == "o_proj.weight") {
        result.push_back("model.language_model.layers." + index + ".self_attn.out_proj.weight");
        result.push_back("model.layers." + index + ".self_attn.out_proj.weight");
    } else if (suffix == "q_norm.weight" || suffix == "k_norm.weight") {
        result.push_back("model.language_model.layers." + index + ".self_attn." + std::string(suffix));
        result.push_back("model.layers." + index + ".self_attn." + std::string(suffix));
    } else if (suffix == "q_norm.weight" || suffix == "k_norm.weight") {
        result.push_back("model.language_model.layers." + index + ".self_attn." + std::string(suffix));
        result.push_back("model.layers." + index + ".self_attn." + std::string(suffix));
    } else if (suffix == "q_norm.weight" || suffix == "k_norm.weight") {
        result.push_back("model.language_model.layers." + index + ".self_attn." + std::string(suffix));
        result.push_back("model.layers." + index + ".self_attn." + std::string(suffix));
    }
    return result;
}

std::vector<std::string> feed_forward_tensor_candidates(int layer,
                                                        std::string_view suffix) {
    const std::string index = std::to_string(layer);
    std::string gguf_suffix;
    if (suffix == "w_gate.weight") gguf_suffix = "ffn_gate.weight";
    if (suffix == "w_up.weight") gguf_suffix = "ffn_up.weight";
    if (suffix == "w_down.weight") gguf_suffix = "ffn_down.weight";
    std::vector<std::string> result = {
        "transformer.h." + index + ".mlp." + std::string(suffix),
        "model.language_model.layers." + index + ".mlp." + std::string(suffix),
        "model.layers." + index + ".mlp." + std::string(suffix),
        "model.language_model.layers." + index + ".feed_forward." + std::string(suffix),
        "layers." + index + ".feed_forward." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
    if (suffix == "w_gate.weight") {
        result.push_back("model.language_model.layers." + index + ".mlp.gate_proj.weight");
        result.push_back("model.layers." + index + ".mlp.gate_proj.weight");
        result.push_back("model.layers." + index + ".feed_forward.w1.weight");
    } else if (suffix == "w_up.weight") {
        result.push_back("model.language_model.layers." + index + ".mlp.up_proj.weight");
        result.push_back("model.layers." + index + ".mlp.up_proj.weight");
        result.push_back("model.layers." + index + ".feed_forward.w3.weight");
    } else if (suffix == "w_down.weight") {
        result.push_back("model.language_model.layers." + index + ".mlp.down_proj.weight");
        result.push_back("model.layers." + index + ".mlp.down_proj.weight");
        result.push_back("model.layers." + index + ".feed_forward.w2.weight");
    }
    return result;
}

std::vector<std::string> shortconv_tensor_candidates(int layer,
                                                     std::string_view suffix) {
    const std::string index = std::to_string(layer);
    return {
        "model.language_model.layers." + index + ".conv." + std::string(suffix),
        "model.layers." + index + ".conv." + std::string(suffix),
        "layers." + index + ".conv." + std::string(suffix),
        "blk." + index + ".shortconv." + std::string(suffix),
    };
}

std::vector<std::string> mamba2_tensor_candidates(int layer,
                                                  std::string_view suffix) {
    const std::string index = std::to_string(layer);
    std::string gguf_suffix;
    if (suffix == "in_proj.weight") gguf_suffix = "ssm_in.weight";
    else if (suffix == "conv1d.weight") gguf_suffix = "ssm_conv1d.weight";
    else if (suffix == "conv1d.bias") gguf_suffix = "ssm_conv1d.bias";
    else if (suffix == "dt_bias") gguf_suffix = "ssm_dt.bias";
    else if (suffix == "A_log") gguf_suffix = "ssm_a";
    else if (suffix == "D") gguf_suffix = "ssm_d";
    else if (suffix == "norm.weight") gguf_suffix = "ssm_norm.weight";
    else if (suffix == "out_proj.weight") gguf_suffix = "ssm_out.weight";
    else gguf_suffix = std::string(suffix);
    return {
        "model.language_model.layers." + index + ".mixer." + std::string(suffix),
        "model.layers." + index + ".mixer." + std::string(suffix),
        "backbone.layers." + index + ".mixer." + std::string(suffix),
        "layers." + index + ".mixer." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
}

std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role) {
    const std::string index = std::to_string(layer);
    switch (role) {
    case TensorRole::AttentionInputNorm:
        return {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
            "model.language_model.layers." + index + ".input_layernorm.weight",
            "model.language_model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".pre_attn_norm.weight",
            "model.language_model.layers." + index + ".pre_attn_norm.weight",
            "backbone.layers." + index + ".norm.weight",
            "blk." + index + ".attn_norm.weight",
        };
    case TensorRole::AttentionPostNorm:
        return {
            "model.layers." + index + ".post_attn_norm.weight",
            "model.language_model.layers." + index + ".post_attn_norm.weight",
            "model.layers." + index + ".mixer_norm.after.weight",
            "model.language_model.layers." + index + ".mixer_norm.after.weight",
            "blk." + index + ".mixer_norm.after.weight",
        };
    case TensorRole::FfnInputNorm:
        return {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
            "model.language_model.layers." + index + ".post_attention_layernorm.weight",
            "model.layers." + index + ".pre_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".pre_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".pre_mlp_norm.weight",
            "model.language_model.layers." + index + ".pre_mlp_norm.weight",
            "blk." + index + ".ffn_norm.weight",
        };
    case TensorRole::FfnOutputNorm:
        return {
            "model.layers." + index + ".post_mlp_norm.weight",
            "model.language_model.layers." + index + ".post_mlp_norm.weight",
            "model.layers." + index + ".post_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".post_feedforward_layernorm.weight",
            "model.layers." + index + ".feed_forward_norm.after.weight",
            "model.language_model.layers." + index + ".feed_forward_norm.after.weight",
            "blk." + index + ".post_ffw_norm.weight",
        };
    default:
        throw std::invalid_argument("tensor role is not a sublayer norm role");
    }
}

bool has_any_tensor(const TensorInventory& inventory,
                    const std::vector<std::string>& candidates) {
    for (const std::string& candidate : candidates) {
        if (inventory.find(candidate) != nullptr) return true;
    }
    return false;
}

std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role) {
    const std::string index = std::to_string(layer);
    switch (role) {
    case TensorRole::AttentionInputNorm:
        return {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
            "model.language_model.layers." + index + ".input_layernorm.weight",
            "model.language_model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".pre_attn_norm.weight",
            "model.language_model.layers." + index + ".pre_attn_norm.weight",
            "backbone.layers." + index + ".norm.weight",
            "blk." + index + ".attn_norm.weight",
        };
    case TensorRole::AttentionPostNorm:
        return {
            "model.layers." + index + ".post_attn_norm.weight",
            "model.language_model.layers." + index + ".post_attn_norm.weight",
            "model.layers." + index + ".mixer_norm.after.weight",
            "model.language_model.layers." + index + ".mixer_norm.after.weight",
            "blk." + index + ".mixer_norm.after.weight",
        };
    case TensorRole::FfnInputNorm:
        return {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
            "model.language_model.layers." + index + ".post_attention_layernorm.weight",
            "model.layers." + index + ".pre_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".pre_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".pre_mlp_norm.weight",
            "model.language_model.layers." + index + ".pre_mlp_norm.weight",
            "blk." + index + ".ffn_norm.weight",
        };
    case TensorRole::FfnOutputNorm:
        return {
            "model.layers." + index + ".post_mlp_norm.weight",
            "model.language_model.layers." + index + ".post_mlp_norm.weight",
            "model.layers." + index + ".post_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".post_feedforward_layernorm.weight",
            "model.layers." + index + ".feed_forward_norm.after.weight",
            "model.language_model.layers." + index + ".feed_forward_norm.after.weight",
            "blk." + index + ".post_ffw_norm.weight",
        };
    default:
        throw std::invalid_argument("tensor role is not a sublayer norm role");
    }
}

bool has_any_tensor(const TensorInventory& inventory,
                    const std::vector<std::string>& candidates) {
    for (const std::string& candidate : candidates) {
        if (inventory.find(candidate) != nullptr) return true;
    }
    return false;
}

std::vector<std::string> norm_tensor_candidates(int layer, TensorRole role) {
    const std::string index = std::to_string(layer);
    switch (role) {
    case TensorRole::AttentionInputNorm:
        return {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
            "model.language_model.layers." + index + ".input_layernorm.weight",
            "model.language_model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".operator_norm.weight",
            "model.layers." + index + ".pre_attn_norm.weight",
            "model.language_model.layers." + index + ".pre_attn_norm.weight",
            "backbone.layers." + index + ".norm.weight",
            "blk." + index + ".attn_norm.weight",
        };
    case TensorRole::AttentionPostNorm:
        return {
            "model.layers." + index + ".post_attn_norm.weight",
            "model.language_model.layers." + index + ".post_attn_norm.weight",
            "model.layers." + index + ".mixer_norm.after.weight",
            "model.language_model.layers." + index + ".mixer_norm.after.weight",
            "blk." + index + ".mixer_norm.after.weight",
        };
    case TensorRole::FfnInputNorm:
        return {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
            "model.language_model.layers." + index + ".post_attention_layernorm.weight",
            "model.layers." + index + ".pre_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".pre_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".ffn_norm.weight",
            "model.layers." + index + ".pre_mlp_norm.weight",
            "model.language_model.layers." + index + ".pre_mlp_norm.weight",
            "blk." + index + ".ffn_norm.weight",
        };
    case TensorRole::FfnOutputNorm:
        return {
            "model.layers." + index + ".post_mlp_norm.weight",
            "model.language_model.layers." + index + ".post_mlp_norm.weight",
            "model.layers." + index + ".post_feedforward_layernorm.weight",
            "model.language_model.layers." + index + ".post_feedforward_layernorm.weight",
            "model.layers." + index + ".feed_forward_norm.after.weight",
            "model.language_model.layers." + index + ".feed_forward_norm.after.weight",
            "blk." + index + ".post_ffw_norm.weight",
        };
    default:
        throw std::invalid_argument("tensor role is not a sublayer norm role");
    }
}

bool has_any_tensor(const TensorInventory& inventory,
                    const std::vector<std::string>& candidates) {
    for (const std::string& candidate : candidates) {
        if (inventory.find(candidate) != nullptr) return true;
    }
    return false;
}

void add_binding(TensorRoleBindings& bindings, TensorRole role, int layer,
                 const TensorInventoryEntry& tensor,
                 std::vector<EvidenceItem> evidence, int physical_layer) {
    bindings.values.push_back({role, layer, -1, physical_layer, tensor.name, tensor.shape,
                               std::move(evidence)});
}

const TensorInventoryEntry* find_mamba_tensor(const InferenceInput& input,
                                              int layer,
                                              std::string_view suffix) {
    const auto candidates = mamba2_tensor_candidates(layer, suffix);
    const TensorInventoryEntry* found = nullptr;
    for (const auto& candidate : candidates) {
        if (const auto* tensor = input.inventory.find(candidate)) {
            if (found != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple Mamba-2 tensor spellings are present for layer " +
                        std::to_string(layer));
            }
            found = tensor;
        }
    }
    return found;
}

bool layer_has_feed_forward(const CanonicalInferenceContext& context,
                            int layer) {
    const auto& input = context.input;
    const auto has_tensor = [&](std::string_view name) {
        return input.inventory.find(name) != nullptr;
    };
    const std::string index = std::to_string(layer);
    return find_mamba_tensor(input, layer, "in_proj.weight") == nullptr &&
        (has_tensor("blk." + index + ".ffn_up.weight") ||
         has_tensor("model.layers." + index + ".mlp.up_proj.weight") ||
         has_tensor("model.language_model.layers." + index + ".mlp.up_proj.weight") ||
         has_tensor("transformer.h." + index + ".mlp.w_up.weight") ||
         has_tensor("model.layers." + index + ".feed_forward.w1.weight") ||
         (context.moe &&
          has_tensor("model.layers." + index +
                     ".mlp.experts.0.gate_proj.weight")));
}

}

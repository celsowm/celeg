#include "support.hpp"


#include <string>

namespace celeg::inference_detail {

[[noreturn]] void fail(ResolutionFailureKind kind, std::string message,
                       std::vector<EvidenceItem> evidence) {
    throw ResolutionError(kind, std::move(message), std::move(evidence));
}

bool shape_is(const TensorInventoryEntry& entry,
              std::initializer_list<std::int64_t> expected) {
    return entry.shape == std::vector<std::int64_t>(expected);
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
    std::vector<std::string> result = {
        "transformer.h." + index + ".attn." + std::string(suffix),
        "model.language_model.layers." + index + ".self_attn." + std::string(suffix),
        "model.layers." + index + ".self_attn." + std::string(suffix),
        "layers." + index + ".attention." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
    if (suffix == "o_proj.weight") {
        result.push_back("model.layers." + index + ".self_attn.out_proj.weight");
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
        "layers." + index + ".feed_forward." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
    if (suffix == "w_gate.weight") {
        result.push_back("model.layers." + index + ".feed_forward.w1.weight");
    } else if (suffix == "w_up.weight") {
        result.push_back("model.layers." + index + ".feed_forward.w3.weight");
    } else if (suffix == "w_down.weight") {
        result.push_back("model.layers." + index + ".feed_forward.w2.weight");
    }
    return result;
}

std::vector<std::string> shortconv_tensor_candidates(int layer,
                                                     std::string_view suffix) {
    const std::string index = std::to_string(layer);
    return {
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
        "layers." + index + ".mixer." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
}

void add_binding(TensorRoleBindings& bindings, TensorRole role, int layer,
                 const TensorInventoryEntry& tensor,
                 std::vector<EvidenceItem> evidence, int physical_layer) {
    bindings.values.push_back({role, layer, -1, physical_layer, tensor.name, tensor.shape,
                               std::move(evidence)});
}

} // namespace celeg::inference_detail

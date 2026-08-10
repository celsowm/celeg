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
    return {
        "transformer.h." + index + ".attn." + std::string(suffix),
        "model.layers." + index + ".self_attn." + std::string(suffix),
        "layers." + index + ".attention." + std::string(suffix),
    };
}

std::vector<std::string> feed_forward_tensor_candidates(int layer,
                                                        std::string_view suffix) {
    const std::string index = std::to_string(layer);
    return {
        "transformer.h." + index + ".mlp." + std::string(suffix),
        "model.layers." + index + ".mlp." + std::string(suffix),
        "layers." + index + ".feed_forward." + std::string(suffix),
    };
}

void add_binding(TensorRoleBindings& bindings, TensorRole role, int layer,
                 const TensorInventoryEntry& tensor,
                 std::vector<EvidenceItem> evidence, int physical_layer) {
    bindings.values.push_back({role, layer, -1, physical_layer, tensor.name, tensor.shape,
                               std::move(evidence)});
}

} // namespace celeg::inference_detail

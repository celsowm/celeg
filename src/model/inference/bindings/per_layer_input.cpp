#include "../canonical_internal.hpp"
#include "../rules.hpp"
#include "../support.hpp"

#include <string>
#include <utility>
#include <vector>

#include "detail.hpp"

namespace celeg::inference_detail {
namespace {

/// A per-layer-input tower carries one embedding table indexed by token *and*
/// layer, a projection of it into the hidden width, and per-layer gate,
/// projection, norm and output scalar. The tensors are named consistently
/// within a checkpoint but the container prefix varies (multimodal packagings
/// nest the text stack under `model.language_model.`), so every candidate is
/// tried under each known prefix rather than hard-coding one layout.
const std::vector<std::string>& stack_prefixes() {
    static const std::vector<std::string> prefixes = {
        "model.language_model.",
        "model.",
        "",
    };
    return prefixes;
}

const TensorInventoryEntry* find_first(const TensorInventory& inventory,
                                       const std::string& suffix) {
    for (const std::string& prefix : stack_prefixes()) {
        if (const auto* entry = inventory.find(prefix + suffix)) return entry;
    }
    return nullptr;
}

const TensorInventoryEntry* find_layer(const TensorInventory& inventory,
                                       int layer, const std::string& suffix) {
    return find_first(inventory, "layers." + std::to_string(layer) + "." + suffix);
}

void bind(CanonicalInferenceContext& context, TensorRole role, int layer,
          const TensorInventoryEntry& tensor) {
    add_binding(context.facts.bindings, role, layer, tensor,
                {{EvidenceKind::TensorName, tensor.name,
                  std::string(tensor_role_name(role))}});
}

}

/// Enables the per-layer-input tower when the checkpoint actually ships one.
///
/// The kernels for this exist on every backend, but the policy that switches
/// them on was previously reachable only from a JSON descriptor, so the
/// automatic resolver silently dropped the whole tower along with each layer's
/// output scalar. google/gemma-4-E4B carries 129 per-layer-input tensors and 42
/// layer_scalar tensors that were never read; its layer 0 scalar is 0.056, so
/// dropping it alone inflated that layer's output by roughly 18x.
///
/// Detection is by tensor presence, matching how the resolver infers every
/// other structure, so any checkpoint shipping the same tensors gets the same
/// treatment without naming an architecture.
void bind_per_layer_input(CanonicalInferenceContext& context) {
    auto& facts = context.facts;
    const TensorInventory& inventory = context.input.inventory;

    const TensorInventoryEntry* embedding =
        find_first(inventory, "embed_tokens_per_layer.weight");
    const TensorInventoryEntry* projection =
        find_first(inventory, "per_layer_model_projection.weight");
    const TensorInventoryEntry* projection_norm =
        find_first(inventory, "per_layer_projection_norm.weight");
    if (embedding == nullptr && projection == nullptr && projection_norm == nullptr) {
        facts.graph.per_layer_input = std::nullopt;
        return;
    }
    if (embedding == nullptr || projection == nullptr || projection_norm == nullptr) {
        fail(ResolutionFailureKind::MissingTensorRole,
             "checkpoint has a partial per-layer-input tower");
    }

    // The per-layer width is the trailing dimension of the shared projection
    // norm, so it is derived from the checkpoint rather than a config key that
    // multimodal packagings nest inconsistently.
    const int input_size = projection_norm->shape.empty()
        ? 0
        : static_cast<int>(projection_norm->shape.back());
    if (input_size <= 0) {
        fail(ResolutionFailureKind::ConflictingMetadata,
             "per-layer-input width could not be derived from the projection norm");
    }
    if (!context.input.metadata.core.norm_epsilon.has_value()) {
        fail(ResolutionFailureKind::MissingRequiredMetadata,
             "per-layer-input tower requires a normalization epsilon");
    }

    bind(context, TensorRole::PerLayerEmbedding, -1, *embedding);
    bind(context, TensorRole::PerLayerContextProjection, -1, *projection);
    bind(context, TensorRole::PerLayerProjectionNorm, -1, *projection_norm);

    for (int layer = 0; layer < context.layer_count; ++layer) {
        const auto* gate = find_layer(inventory, layer, "per_layer_input_gate.weight");
        const auto* per_layer_projection =
            find_layer(inventory, layer, "per_layer_projection.weight");
        const auto* input_norm =
            find_layer(inventory, layer, "post_per_layer_input_norm.weight");
        const auto* scalar = find_layer(inventory, layer, "layer_scalar");
        if (gate == nullptr || per_layer_projection == nullptr ||
            input_norm == nullptr || scalar == nullptr) {
            fail(ResolutionFailureKind::MissingTensorRole,
                 "per-layer-input tower is missing tensors for layer " +
                     std::to_string(layer));
        }
        bind(context, TensorRole::PerLayerInputGate, layer, *gate);
        bind(context, TensorRole::PerLayerProjection, layer, *per_layer_projection);
        bind(context, TensorRole::PerLayerInputNorm, layer, *input_norm);
        bind(context, TensorRole::LayerScalar, layer, *scalar);
    }

    facts.graph.per_layer_input = PerLayerInputPolicy{
        input_size,
        ActivationKind::GeluTanh,
        NormSpec{*context.input.metadata.core.norm_epsilon, NormWeightKind::Scale}};
}

}

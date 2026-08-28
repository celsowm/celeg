#include "../canonical_internal.hpp"
#include "../rules.hpp"
#include "../support.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"

namespace celeg::inference_detail {
namespace {

void add_global_binding(CanonicalInferenceContext& context,
                        TensorRole role,
                        const TensorInventoryEntry& tensor) {
    add_binding(
        context.facts.bindings,
        role,
        -1,
        tensor,
        {{EvidenceKind::TensorName,
          tensor.name,
          std::string(tensor_role_name(role))}});
}

}

void bind_global_tensors(CanonicalInferenceContext& context) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& facts = context.facts;

    add_global_binding(context, TensorRole::TokenEmbedding, *context.embedding);

    const std::vector<std::string> head_names = {
        "lm_head.weight",
        "transformer.lm_head.weight",
        "output.weight",
    };
    const TensorInventoryEntry* head = nullptr;
    for (const std::string& name : head_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (head != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple language-model heads are present");
            }
            head = candidate;
        }
    }
    if (head == nullptr) {
        const std::vector<std::string> head_suffixes = {
            ".lm_head.weight",
            ".output.weight",
        };
        for (const TensorInventoryEntry& entry : input.inventory.entries()) {
            const std::string& name = entry.name;
            for (const std::string& suffix : head_suffixes) {
                if (name.size() >= suffix.size() &&
                    name.compare(name.size() - suffix.size(),
                                 suffix.size(), suffix) == 0) {
                    if (head != nullptr) {
                        fail(
                            ResolutionFailureKind::AmbiguousTensorBinding,
                            "multiple language-model heads are present");
                    }
                    head = &entry;
                    break;
                }
            }
        }
    }
    if (head == nullptr) {
        if (const bool already_determined =
                m.core.tied_embeddings.has_value();
            !already_determined && context.embedding != nullptr) {
            facts.tied_embeddings = true;
            facts.evidence.push_back({
                EvidenceKind::Derived,
                context.embedding->name,
                "safetensors checkpoint has no independent language-model head; "
                "tied to token embedding"});
        }
        if (!facts.tied_embeddings) {
            fail(
                ResolutionFailureKind::MissingTensorRole,
                "untied checkpoint has no language-model head");
        }
        head = context.embedding;
        facts.evidence.push_back({
            EvidenceKind::Derived,
            context.embedding->name,
            "language-model head is tied to token embedding"});
    } else if (!shape_is(*head, {*m.core.vocab_size, *m.core.hidden_size})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "language-model head shape does not agree with normalized metadata");
    }
    add_global_binding(context, TensorRole::LanguageModelHead, *head);

    const std::vector<std::string> final_norm_names = {
        "transformer.ln_f.weight",
        "model.norm.weight",
        "norm.weight",
        "output_norm.weight",
        "model.language_model.norm.weight",
        "backbone.norm_f.weight",
        "model.embedding_norm.weight",
        "token_embd_norm.weight",
    };
    const TensorInventoryEntry* final_norm = nullptr;
    for (const std::string& name : final_norm_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (final_norm != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple final norms are present");
            }
            final_norm = candidate;
        }
    }
    if (final_norm == nullptr) {
        const std::vector<std::string> final_norm_suffixes = {
            ".embedding_norm.weight",
            ".norm.weight",
            ".ln_f.weight",
            ".output_norm.weight",
            ".norm_f.weight",
            ".token_embd_norm.weight",
        };
        for (const TensorInventoryEntry& entry : input.inventory.entries()) {
            const std::string& name = entry.name;
            for (const std::string& suffix : final_norm_suffixes) {
                if (name.size() >= suffix.size() &&
                    name.compare(name.size() - suffix.size(),
                                 suffix.size(), suffix) == 0) {
                    if (final_norm != nullptr) {
                        fail(
                            ResolutionFailureKind::AmbiguousTensorBinding,
                            "multiple final norms are present");
                    }
                    final_norm = &entry;
                    break;
                }
            }
        }
    }
    if (final_norm == nullptr) {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "automatic resolution could not find final norm");
    }
    if (!shape_is(*final_norm, {*m.core.hidden_size})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "final norm shape mismatch");
    }
    add_global_binding(context, TensorRole::FinalNorm, *final_norm);
}

}

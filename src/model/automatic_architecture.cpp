#include "celeg/model/architecture.hpp"

#include "celeg/model/inference.hpp"

#include <stdexcept>

namespace celeg {
namespace {

class AutomaticArchitecture final : public IArchitecture {
public:
    std::string_view id() const override { return "automatic"; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        // Composite text/vision configurations and explicit architecture
        // descriptors remain owned by their importer. A flat checkpoint can
        // still be inferred without treating its repository name or model
        // type as evidence.
        if (metadata.contains("text_config.model_type")) {
            return {false, 0, "composite checkpoint requires an explicit importer"};
        }
        const bool has_structural_metadata = metadata.contains("hidden_size") &&
            metadata.contains("num_hidden_layers") &&
            metadata.contains("num_attention_heads") &&
            metadata.contains("vocab_size");
        return {has_structural_metadata, 0,
                has_structural_metadata ? "generic structural metadata" :
                                           "required structural metadata is absent"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("automatic resolution is not applicable to this checkpoint");
        }
        const InferenceInput input = build_inference_input(checkpoint);
        CanonicalModelFacts facts = infer_canonical_model_facts(input);
        facts.source_format = checkpoint.metadata.is_gguf() ? "gguf" : "safetensors";
        return ResolutionAssembler{}.assemble(facts);
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_automatic_architecture() {
    return std::make_unique<AutomaticArchitecture>();
}

} // namespace celeg

#include "celeg/model/architecture.hpp"

#include "celeg/model/inference.hpp"


#include <string>
#include <stdexcept>

namespace celeg {
namespace {


class AutomaticArchitecture final : public IArchitecture {
public:
    std::string_view id() const override { return "automatic"; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        // Component-prefixed metadata is normalized by the same semantic
        // resolver as flat metadata. The root may also carry unrelated vision
        // fields; those do not make text resolution identity-specific.
        const auto has = [&metadata](std::string_view key) {
            return metadata.contains(key) ||
                   metadata.contains("text_config." + std::string(key));
        };
        const auto has_gguf = [&metadata](std::string_view suffix) {
            return metadata.is_gguf() && metadata.contains(
                metadata.architecture_type() + "." + std::string(suffix));
        };
        // GGUF tokenizer tables are structural vocabulary evidence.  Several
        // valid GGUF writers deliberately omit <architecture>.vocab_size
        // because the table is the authoritative source (notably hybrid
        // recurrent checkpoints).  Do not make an identity string decide
        // whether the generic resolver gets a chance to inspect them.
        const bool has_vocabulary =
            has("vocab_size") || has_gguf("vocab_size") ||
            metadata.contains("tokenizer.ggml.tokens");
        const bool has_structural_metadata =
            (has("hidden_size") || has_gguf("embedding_length")) &&
            (has("num_hidden_layers") || has_gguf("block_count")) &&
            (has("num_attention_heads") || has("num_heads") ||
             has("mamba_num_heads") || has_gguf("attention.head_count")) &&
            has_vocabulary;
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
        ResolvedModel result = ResolutionAssembler{}.assemble(facts);
        return result;
    }
};

} // namespace

std::unique_ptr<IArchitecture> make_automatic_architecture() {
    return std::make_unique<AutomaticArchitecture>();
}

} // namespace celeg

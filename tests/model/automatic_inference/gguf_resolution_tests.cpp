#include "gguf_resolution_tests.hpp"

#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

namespace celeg::automatic_inference_test {

void run_gguf_resolution_tests(const celeg::ArchitectureCatalog& catalog) {
    celeg::CheckpointView gguf_checkpoint;
    gguf_checkpoint.metadata = gguf_metadata();
    gguf_checkpoint.repository = gguf_repository();
    const auto& gguf_architecture = catalog.select(gguf_checkpoint.metadata);
    const celeg::ResolvedModel gguf_model = gguf_architecture.resolve(gguf_checkpoint);
    CELEG_TEST_CHECK(gguf_model.provenance.source_format == "gguf");
    CELEG_TEST_CHECK(gguf_model.graph.hidden == 8);
    CELEG_TEST_CHECK(gguf_model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(celeg::explain_resolution(gguf_checkpoint).failures.empty());
    /// A GGUF architecture absent from the no-RoPE profile, with active
    /// "rope.freq_base" metadata, must resolve to a real RopePositionSpec:
    /// generic inference does not treat all GGUF checkpoints as position-free,
    /// only the ones the format boundary declares as such.
    CELEG_TEST_CHECK(std::holds_alternative<celeg::RopePositionSpec>(
        std::get<celeg::AttentionSpec>(gguf_model.graph.layers[0].mixer).position));

    celeg::CheckpointView no_rope_checkpoint;
    no_rope_checkpoint.metadata = no_rope_gguf_metadata();
    no_rope_checkpoint.repository = gguf_repository();
    const celeg::ResolvedModel no_rope_model =
        catalog.select(no_rope_checkpoint.metadata).resolve(no_rope_checkpoint);
    /// Same tensor grammar and same "active-looking" rope hparams as
    /// gguf_model above, but a GGUF architecture the format boundary knows
    /// never applies RoPE: the resolved attention layer must carry
    /// NoPositionEncodingSpec regardless of the vestigial rope metadata.
    CELEG_TEST_CHECK(std::holds_alternative<celeg::NoPositionEncodingSpec>(
        std::get<celeg::AttentionSpec>(no_rope_model.graph.layers[0].mixer).position));
    CELEG_TEST_CHECK(celeg::explain_resolution(no_rope_checkpoint).failures.empty());

    auto tokenizer_vocab = gguf_metadata();
    tokenizer_vocab.values.erase("conventional.vocab_size");
    tokenizer_vocab.values["tokenizer.ggml.tokens"] =
        std::vector<std::string>(32, "token");
    celeg::CheckpointView tokenizer_vocab_checkpoint;
    tokenizer_vocab_checkpoint.metadata = std::move(tokenizer_vocab);
    tokenizer_vocab_checkpoint.repository = gguf_repository();
    const celeg::ResolvedModel tokenizer_vocab_model =
        catalog.select(tokenizer_vocab_checkpoint.metadata).resolve(tokenizer_vocab_checkpoint);
    CELEG_TEST_CHECK(tokenizer_vocab_model.topology.dims.vocab_size == 32);
}

}

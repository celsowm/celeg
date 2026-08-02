#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/architecture.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

class TestArchitecture final : public celeg::IArchitecture {
public:
    TestArchitecture(std::string name, int specificity)
        : name_(std::move(name)), specificity_(specificity) {}

    std::string_view id() const override { return name_; }
    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {true, specificity_, "test"};
    }
    celeg::ResolvedModel resolve(const celeg::CheckpointView&) const override {
        return {};
    }

private:
    std::string name_;
    int specificity_;
};


celeg::CheckpointMetadata lfm_metadata(std::string name = "LiquidAI/LFM2.5-1.2B-Instruct") {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::string("lfm2");
    metadata.values["dtype"] = std::string("bfloat16");
    metadata.values["hidden_size"] = int64_t(2048);
    metadata.values["intermediate_size"] = int64_t(12288);
    metadata.values["num_hidden_layers"] = int64_t(16);
    metadata.values["num_attention_heads"] = int64_t(32);
    metadata.values["num_key_value_heads"] = int64_t(8);
    metadata.values["head_dim"] = int64_t(64);
    metadata.values["vocab_size"] = int64_t(65536);
    metadata.values["conv_L_cache"] = int64_t(3);
    metadata.values["conv_dim"] = int64_t(2048);
    metadata.values["max_position_embeddings"] = int64_t(32768);
    metadata.values["bos_token_id"] = int64_t(1);
    metadata.values["eos_token_id"] = int64_t(7);
    metadata.values["pad_token_id"] = int64_t(0);
    metadata.values["norm_eps"] = 1.0e-5;
    metadata.values["rope_theta"] = 1.0e6;
    metadata.values["layer_types"] = std::vector<std::string>{
        "conv", "conv", "full_attention", "conv", "conv", "full_attention",
        "conv", "conv", "full_attention", "conv", "conv", "full_attention",
        "conv", "full_attention", "full_attention", "full_attention"};
    metadata.repository_hint = std::move(name);
    return metadata;
}

} // namespace

int main() {
    const auto catalog = celeg::create_builtin_architecture_catalog();
    CELEG_TEST_CHECK(catalog->ids().size() == 2);
    const auto metadata = lfm_metadata();
    const auto& architecture = catalog->select(metadata);
    CELEG_TEST_CHECK(architecture.id() == "lfm2");

    celeg::CheckpointView checkpoint;
    checkpoint.metadata = metadata;
    const celeg::ResolvedModel model = architecture.resolve(checkpoint);
    CELEG_TEST_CHECK(model.architecture_id == "lfm2");
    CELEG_TEST_CHECK(model.topology.intermediate == 8192);
    CELEG_TEST_CHECK(model.graph.layers.size() == 16);
    CELEG_TEST_CHECK(model.graph.layers[0].mixer_kind() == celeg::MixerKind::ShortConvolution);
    CELEG_TEST_CHECK(model.graph.layers[2].mixer_kind() == celeg::MixerKind::Attention);
    CELEG_TEST_CHECK(!model.graph.has_moe());

    auto granite = metadata;
    granite.values["model_type"] = std::string("granite");
    const auto& granite_architecture = catalog->select(granite);
    CELEG_TEST_CHECK(granite_architecture.id() == "granite");

    celeg::CheckpointMetadata granite_gguf;
    granite_gguf.source_format = celeg::CheckpointSourceFormat::Gguf;
    granite_gguf.values["general.architecture"] = std::string("granite");
    granite_gguf.values["granite.embedding_length"] = int64_t(8);
    granite_gguf.values["granite.feed_forward_length"] = int64_t(16);
    granite_gguf.values["granite.block_count"] = int64_t(1);
    granite_gguf.values["granite.attention.head_count"] = int64_t(2);
    granite_gguf.values["granite.attention.head_count_kv"] = int64_t(1);
    granite_gguf.values["granite.attention.key_length"] = int64_t(4);
    granite_gguf.values["granite.context_length"] = int64_t(64);
    granite_gguf.values["granite.attention.layer_norm_rms_epsilon"] = 1.0e-5;
    granite_gguf.values["granite.rope.freq_base"] = 10000.0;
    granite_gguf.values["granite.vocab_size"] = int64_t(32);
    granite_gguf.values["tokenizer.ggml.bos_token_id"] = int64_t(1);
    granite_gguf.values["tokenizer.ggml.eos_token_id"] = int64_t(2);
    granite_gguf.values["tokenizer.ggml.padding_token_id"] = int64_t(0);
    celeg::CheckpointView granite_gguf_checkpoint;
    granite_gguf_checkpoint.metadata = granite_gguf;
    const celeg::ResolvedModel granite_gguf_model =
        granite_architecture.resolve(granite_gguf_checkpoint);
    CELEG_TEST_CHECK(granite_gguf_model.definition.source_format == "gguf");
    CELEG_TEST_CHECK(granite_gguf_model.chat_profile_id == "granite-instruct");
    CELEG_TEST_CHECK(granite_gguf_model.topology.vocab_size == 32);
    CELEG_TEST_CHECK(granite_gguf_model.graph.layers.size() == 1);

    celeg::ArchitectureCatalog mutable_catalog;
    mutable_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    bool duplicate_rejected = false;
    try {
        mutable_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    CELEG_TEST_CHECK(duplicate_rejected);
    mutable_catalog.freeze();
    bool mutation_rejected = false;
    try {
        mutable_catalog.add(std::make_unique<TestArchitecture>("two", 10));
    } catch (const std::logic_error&) {
        mutation_rejected = true;
    }
    CELEG_TEST_CHECK(mutation_rejected);

    celeg::ArchitectureCatalog ambiguous_catalog;
    ambiguous_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    ambiguous_catalog.add(std::make_unique<TestArchitecture>("two", 10));
    bool ambiguity_rejected = false;
    try {
        ambiguous_catalog.select(metadata);
    } catch (const std::runtime_error&) {
        ambiguity_rejected = true;
    }
    CELEG_TEST_CHECK(ambiguity_rejected);

    std::cout << "architecture_resolution_test: ok\n";
}

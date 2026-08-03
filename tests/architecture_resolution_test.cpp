#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/architecture.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

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

celeg::CheckpointMetadata gemma_metadata(int hidden, int intermediate, int layers,
                                         int kv_heads, int shared, std::string name) {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::string("gemma4");
    metadata.values["text_config.model_type"] = std::string("gemma4_text");
    metadata.values["text_config.hidden_size"] = int64_t(hidden);
    metadata.values["text_config.intermediate_size"] = int64_t(intermediate);
    metadata.values["text_config.num_hidden_layers"] = int64_t(layers);
    metadata.values["text_config.num_attention_heads"] = int64_t(8);
    metadata.values["text_config.num_key_value_heads"] = int64_t(kv_heads);
    metadata.values["text_config.head_dim"] = int64_t(256);
    metadata.values["text_config.global_head_dim"] = int64_t(512);
    metadata.values["text_config.vocab_size"] = int64_t(262144);
    metadata.values["text_config.max_position_embeddings"] = int64_t(131072);
    metadata.values["text_config.num_kv_shared_layers"] = int64_t(shared);
    metadata.values["text_config.sliding_window"] = int64_t(512);
    metadata.values["text_config.hidden_size_per_layer_input"] = int64_t(256);
    metadata.values["text_config.use_double_wide_mlp"] = true;
    metadata.values["text_config.final_logit_softcapping"] = 30.0;
    metadata.values["text_config.rms_norm_eps"] = 1.0e-6;
    metadata.values["text_config.bos_token_id"] = int64_t(2);
    metadata.values["text_config.eos_token_id"] = int64_t(1);
    metadata.values["text_config.pad_token_id"] = int64_t(0);
    std::vector<std::string> schedule(static_cast<size_t>(layers), "sliding_attention");
    for (int i = 5; i < layers; i += 6) schedule[static_cast<size_t>(i)] = "full_attention";
    schedule.back() = "full_attention";
    metadata.values["text_config.layer_types"] = std::move(schedule);
    metadata.values["text_config.rope_parameters.sliding_attention.rope_theta"] = 10000.0;
    metadata.values["text_config.rope_parameters.sliding_attention.rope_type"] =
        std::string("default");
    metadata.values["text_config.rope_parameters.full_attention.rope_theta"] = 1000000.0;
    metadata.values["text_config.rope_parameters.full_attention.rope_type"] =
        std::string("proportional");
    metadata.values["text_config.rope_parameters.full_attention.partial_rotary_factor"] = 0.25;
    metadata.repository_hint = std::move(name);
    return metadata;
}

} // namespace

int main() {
    const auto catalog = celeg::create_builtin_architecture_catalog();
    CELEG_TEST_CHECK(catalog->ids().size() == 4);
    const auto metadata = lfm_metadata();
    const auto& architecture = catalog->select(metadata);
    CELEG_TEST_CHECK(architecture.id() == "lfm2");

    celeg::CheckpointView checkpoint;
    checkpoint.metadata = metadata;
    const celeg::ResolvedModel model = architecture.resolve(checkpoint);
    CELEG_TEST_CHECK(model.architecture_id == "lfm2");
    CELEG_TEST_CHECK(model.topology.intermediate == 12288);
    CELEG_TEST_CHECK(model.graph.layers.size() == 16);
    CELEG_TEST_CHECK(model.graph.layers[0].mixer_kind() == celeg::MixerKind::ShortConvolution);
    CELEG_TEST_CHECK(model.graph.layers[2].mixer_kind() == celeg::MixerKind::Attention);
    CELEG_TEST_CHECK(!model.graph.has_moe());

    celeg::CheckpointMetadata minicpm5;
    minicpm5.repository_hint = "openbmb/MiniCPM5-1B";
    minicpm5.values["model_type"] = std::string("llama");
    minicpm5.values["hidden_size"] = int64_t(1536);
    minicpm5.values["intermediate_size"] = int64_t(4608);
    minicpm5.values["num_hidden_layers"] = int64_t(24);
    minicpm5.values["num_attention_heads"] = int64_t(16);
    minicpm5.values["num_key_value_heads"] = int64_t(2);
    minicpm5.values["head_dim"] = int64_t(128);
    minicpm5.values["vocab_size"] = int64_t(130560);
    minicpm5.values["max_position_embeddings"] = int64_t(131072);
    minicpm5.values["bos_token_id"] = int64_t(0);
    minicpm5.values["eos_token_id"] = std::vector<int64_t>{1, 130073};
    minicpm5.values["pad_token_id"] = int64_t(1);
    minicpm5.values["rms_norm_eps"] = 1.0e-6;
    minicpm5.values["rope_theta"] = 5.0e6;
    const auto& minicpm5_architecture = catalog->select(minicpm5);
    CELEG_TEST_CHECK(minicpm5_architecture.id() == "minicpm5");
    celeg::CheckpointView minicpm5_checkpoint;
    minicpm5_checkpoint.metadata = minicpm5;
    const auto minicpm5_model = minicpm5_architecture.resolve(minicpm5_checkpoint);
    CELEG_TEST_CHECK(minicpm5_model.topology.num_hidden_layers == 24);
    CELEG_TEST_CHECK(minicpm5_model.topology.attention_layouts.front().key_value_heads == 2);
    CELEG_TEST_CHECK(minicpm5_model.topology.eos_token_ids == (std::vector<int>{1, 130073}));
    CELEG_TEST_CHECK(minicpm5_model.definition.tokens.eos == (std::vector<int>{1, 130073}));
    CELEG_TEST_CHECK(minicpm5_model.chat_profile_id == "minicpm5-instruct");

    celeg::CheckpointMetadata minicpm5_gguf;
    minicpm5_gguf.source_format = celeg::CheckpointSourceFormat::Gguf;
    minicpm5_gguf.repository_hint = "openbmb/MiniCPM5-1B-GGUF";
    minicpm5_gguf.values["general.architecture"] = std::string("llama");
    minicpm5_gguf.values["llama.embedding_length"] = int64_t(1536);
    minicpm5_gguf.values["llama.feed_forward_length"] = int64_t(4608);
    minicpm5_gguf.values["llama.block_count"] = int64_t(24);
    minicpm5_gguf.values["llama.attention.head_count"] = int64_t(16);
    minicpm5_gguf.values["llama.attention.head_count_kv"] = int64_t(2);
    minicpm5_gguf.values["llama.attention.key_length"] = int64_t(128);
    minicpm5_gguf.values["llama.context_length"] = int64_t(131072);
    minicpm5_gguf.values["llama.attention.layer_norm_rms_epsilon"] = 1.0e-6;
    minicpm5_gguf.values["llama.rope.freq_base"] = 5.0e6;
    minicpm5_gguf.values["llama.vocab_size"] = int64_t(130560);
    minicpm5_gguf.values["tokenizer.ggml.bos_token_id"] = int64_t(0);
    minicpm5_gguf.values["tokenizer.ggml.eos_token_id"] = int64_t(1);
    minicpm5_gguf.values["tokenizer.ggml.eot_token_id"] = int64_t(130073);
    minicpm5_gguf.values["tokenizer.ggml.padding_token_id"] = int64_t(1);
    const auto& minicpm5_gguf_architecture = catalog->select(minicpm5_gguf);
    CELEG_TEST_CHECK(minicpm5_gguf_architecture.id() == "minicpm5");
    celeg::CheckpointView minicpm5_gguf_checkpoint;
    minicpm5_gguf_checkpoint.metadata = minicpm5_gguf;
    const auto minicpm5_gguf_model = minicpm5_gguf_architecture.resolve(minicpm5_gguf_checkpoint);
    CELEG_TEST_CHECK(minicpm5_gguf_model.definition.source_format == "gguf");
    CELEG_TEST_CHECK(minicpm5_gguf_model.topology.eos_token_ids == (std::vector<int>{1, 130073}));

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

    for (const auto& gemma : {
        gemma_metadata(1536, 6144, 35, 1, 20, "google/gemma-4-E2B-it"),
        gemma_metadata(2560, 10240, 42, 2, 18, "google/gemma-4-E4B-it")}) {
        const auto& gemma_architecture = catalog->select(gemma);
        CELEG_TEST_CHECK(gemma_architecture.id() == "gemma4");
        celeg::CheckpointView gemma_checkpoint;
        gemma_checkpoint.metadata = gemma;
        const auto resolved = gemma_architecture.resolve(gemma_checkpoint);
        CELEG_TEST_CHECK(resolved.topology.attention_layouts.size() ==
                         resolved.topology.num_hidden_layers);
        CELEG_TEST_CHECK(resolved.topology.attention_layouts[0].mask ==
                         celeg::AttentionMaskKind::SlidingCausal);
        CELEG_TEST_CHECK(resolved.topology.attention_layouts[5].mask ==
                         celeg::AttentionMaskKind::Causal);
        const auto& local = resolved.topology.attention_layouts[0];
        const int expected_local_kv_heads = resolved.topology.hidden == 1536 ? 1 : 2;
        CELEG_TEST_CHECK(local.query_heads == 8 && local.key_value_heads ==
                         expected_local_kv_heads);
        CELEG_TEST_CHECK(local.head_dim == 256 && local.query_width() == 2048);
        CELEG_TEST_CHECK(local.key_value_width() == local.key_value_heads * 256);
        CELEG_TEST_CHECK(local.mask == celeg::AttentionMaskKind::SlidingCausal);
        CELEG_TEST_CHECK(local.sliding_window == 512 && local.rope_theta == 10000.0);
        const auto& full = resolved.topology.attention_layouts[5];
        CELEG_TEST_CHECK(full.mask == celeg::AttentionMaskKind::Causal);
        CELEG_TEST_CHECK(full.head_dim == 512 && full.query_width() == 4096);
        CELEG_TEST_CHECK(full.key_value_width() == local.key_value_heads * 512);
        CELEG_TEST_CHECK(full.rotary_fraction == 0.25 && full.rope_theta == 1000000.0);
        const int shared_start = resolved.topology.num_hidden_layers -
            static_cast<int>(gemma.integer("text_config.num_kv_shared_layers"));
        CELEG_TEST_CHECK(resolved.topology.attention_layouts[shared_start]
                             .kv_sharing.shared());
        CELEG_TEST_CHECK(resolved.topology.shared_kv_group_count == 2);
        CELEG_TEST_CHECK(resolved.topology.has_per_layer_input);
        CELEG_TEST_CHECK(resolved.topology.per_layer_input_size == 256);
        CELEG_TEST_CHECK(resolved.graph.layers[0].per_layer_input.enabled);
        CELEG_TEST_CHECK(std::get<celeg::DenseFeedForwardSpec>(
                             resolved.graph.layers.back().feed_forward).intermediate_size ==
                         resolved.topology.intermediate * 2);
        CELEG_TEST_CHECK(resolved.capabilities.tied_embeddings);
        CELEG_TEST_CHECK(resolved.graph.final_logit_softcap == 30.0f);
        CELEG_TEST_CHECK(resolved.chat_profile_id == "gemma4-instruct");
    }

    celeg::CheckpointMetadata gemma_gguf;
    gemma_gguf.source_format = celeg::CheckpointSourceFormat::Gguf;
    gemma_gguf.values["general.architecture"] = std::string("gemma4");
    gemma_gguf.values["gemma4.embedding_length"] = int64_t(1536);
    gemma_gguf.values["gemma4.feed_forward_length"] = int64_t(6144);
    gemma_gguf.values["gemma4.block_count"] = int64_t(35);
    gemma_gguf.values["gemma4.vocab_size"] = int64_t(262144);
    gemma_gguf.values["gemma4.context_length"] = int64_t(131072);
    gemma_gguf.values["gemma4.attention.head_count"] = int64_t(8);
    gemma_gguf.values["gemma4.attention.head_count_kv"] = int64_t(1);
    gemma_gguf.values["gemma4.attention.key_length"] = int64_t(512);
    gemma_gguf.values["gemma4.attention.key_length_swa"] = int64_t(256);
    gemma_gguf.values["gemma4.attention.layer_norm_rms_epsilon"] = 1.0e-6;
    gemma_gguf.values["gemma4.attention.sliding_window"] = int64_t(512);
    gemma_gguf.values["gemma4.attention.shared_kv_layers"] = int64_t(20);
    gemma_gguf.values["gemma4.embedding_length_per_layer_input"] = int64_t(256);
    gemma_gguf.values["gemma4.final_logit_softcapping"] = 30.0;
    gemma_gguf.values["gemma4.rope.freq_base"] = 1000000.0;
    gemma_gguf.values["gemma4.rope.freq_base_swa"] = 10000.0;
    gemma_gguf.values["gemma4.attention.sliding_window_pattern"] =
        std::vector<int64_t>(35, 1);
    auto& gguf_schedule = std::get<std::vector<int64_t>>(
        gemma_gguf.values["gemma4.attention.sliding_window_pattern"]);
    gguf_schedule[5] = 0;
    gguf_schedule.back() = 0;
    gemma_gguf.values["tokenizer.ggml.bos_token_id"] = int64_t(2);
    gemma_gguf.values["tokenizer.ggml.eos_token_id"] = int64_t(1);
    gemma_gguf.values["tokenizer.ggml.padding_token_id"] = int64_t(0);
    celeg::CheckpointView gemma_gguf_checkpoint;
    gemma_gguf_checkpoint.metadata = gemma_gguf;
    const auto& gemma_gguf_architecture = catalog->select(gemma_gguf);
    const auto gemma_gguf_model = gemma_gguf_architecture.resolve(gemma_gguf_checkpoint);
    CELEG_TEST_CHECK(gemma_gguf_model.definition.source_format == "gguf");
    CELEG_TEST_CHECK(gemma_gguf_model.topology.hidden == 1536);
    CELEG_TEST_CHECK(gemma_gguf_model.topology.intermediate == 6144);
    CELEG_TEST_CHECK(gemma_gguf_model.topology.attention_layouts[5].mask ==
                     celeg::AttentionMaskKind::Causal);
    CELEG_TEST_CHECK(gemma_gguf_model.topology.attention_layouts[6].mask ==
                     celeg::AttentionMaskKind::SlidingCausal);
    CELEG_TEST_CHECK(gemma_gguf_model.topology.attention_layouts[5].head_dim == 512);

    auto malformed_gemma = gemma_metadata(1536, 6144, 35, 1, 20, "bad");
    malformed_gemma.values["text_config.model_type"] = std::string("gemma4_vision");
    bool malformed_rejected = false;
    try { (void)catalog->select(malformed_gemma); }
    catch (const std::runtime_error&) { malformed_rejected = true; }
    CELEG_TEST_CHECK(malformed_rejected);

    auto malformed_schedule = gemma_metadata(1536, 6144, 35, 1, 20, "bad-schedule");
    malformed_schedule.values["text_config.layer_types"] =
        std::vector<std::string>{"sliding_attention"};
    malformed_rejected = false;
    try {
        const auto& architecture = catalog->select(malformed_schedule);
        celeg::CheckpointView checkpoint;
        checkpoint.metadata = malformed_schedule;
        (void)architecture.resolve(checkpoint);
    }
    catch (const std::runtime_error&) { malformed_rejected = true; }
    CELEG_TEST_CHECK(malformed_rejected);

    auto malformed_sharing = gemma_metadata(1536, 6144, 35, 1, 20, "bad-sharing");
    malformed_sharing.values["text_config.num_kv_shared_layers"] = int64_t(36);
    malformed_rejected = false;
    try {
        const auto& architecture = catalog->select(malformed_sharing);
        celeg::CheckpointView checkpoint;
        checkpoint.metadata = malformed_sharing;
        (void)architecture.resolve(checkpoint);
    } catch (const std::runtime_error&) { malformed_rejected = true; }
    CELEG_TEST_CHECK(malformed_rejected);

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
